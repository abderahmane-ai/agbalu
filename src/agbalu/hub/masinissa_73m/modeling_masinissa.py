"""Masinissa-73M: a modern deep bidirectional masked language model for Kabyle.

Implements Block Attention Residuals, SwiGLU, and QK RMSNorm for Hugging Face transformers.
No dependencies beyond torch and transformers.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from .configuration_masinissa import MasinissaConfig

IGNORE_INDEX = -100


def log_bucket_indices(offsets: Tensor, buckets: int, context_tokens: int) -> Tensor:
    """Map signed relative token distances to logarithmic buckets."""
    middle = buckets // 2
    distance = offsets.abs().clamp(min=middle, max=context_tokens - 1)
    logarithmic = (
        middle
        + torch.ceil(
            torch.log(distance.float() / middle)
            / math.log((context_tokens - 1) / middle)
            * (middle - 1)
        ).long()
    )
    signed = torch.where(offsets.abs() <= middle, offsets, logarithmic * offsets.sign())
    return signed + buckets - 1


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.epsilon)
        return (x.float() * norm).to(x.dtype) * self.weight.to(x.dtype)


class DepthAttention(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width, epsilon)

    def forward(self, blocks: list[Tensor], partial: Tensor | None) -> Tensor:
        candidates = [*blocks, *([] if partial is None else [partial])]
        stacked = torch.stack(candidates, dim=0)  # [num_candidates, batch, length, width]
        normed = self.norm(stacked)
        weights = (normed.float() * self.query.float()).sum(dim=-1).softmax(dim=0).to(stacked.dtype)
        out: Tensor = (weights.unsqueeze(-1) * stacked).sum(dim=0)
        return out


class Attention(nn.Module):
    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.head_size = config.head_size
        self.scale = 1.0 / math.sqrt(3 * self.head_size)

        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_size, config.layer_norm_eps)
        self.k_norm = RMSNorm(self.head_size, config.layer_norm_eps)

    def forward(
        self,
        hidden: Tensor,
        relative: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden)
        # Reshape to [batch, length, 3, heads, head_size]
        q, k, v = qkv.view(batch, length, 3, self.heads, self.head_size).unbind(2)
        # Permute to [batch, heads, length, head_size]
        q = self.q_norm(q).permute(0, 2, 1, 3)
        k = self.k_norm(k).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        position_q, position_k = relative  # [heads, 2*buckets-1, head_size]

        positions = torch.arange(length, device=hidden.device)
        qi = positions[:, None]
        ki = positions[None, :]
        distance = qi - ki  # [length, length]
        forward_buckets = log_bucket_indices(
            distance, self.config.position_buckets, self.config.max_position_embeddings
        )
        reverse_buckets = log_bucket_indices(
            -distance, self.config.position_buckets, self.config.max_position_embeddings
        )

        valid = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones((batch, length), dtype=torch.bool, device=hidden.device)
        )

        with torch.autocast(hidden.device.type, enabled=False):
            # Content-to-content scores: [batch, heads, length, length]
            c2c = torch.matmul(q.float(), k.float().transpose(-1, -2))

            # Positional score tables per batch item: [batch, heads, length, 2*buckets-1]
            c2p_table = torch.matmul(q.float(), position_k.float().unsqueeze(0).transpose(-1, -2))
            p2c_table = torch.matmul(k.float(), position_q.float().unsqueeze(0).transpose(-1, -2))

            # Gather along query/key indices: [batch, heads, length, length]
            c2p = c2p_table[:, :, qi, forward_buckets]
            p2c = p2c_table[:, :, ki, reverse_buckets]

            scores = (c2c + c2p + p2c) * self.scale

            real_pairs = valid[:, None, :, None] & valid[:, None, None, :]
            safe_pad = (~valid[:, None, :, None]) & (qi == ki)[None, None, :, :]
            allowed = real_pairs | safe_pad
            scores = torch.where(allowed, scores, float("-inf"))

            probs = functional.softmax(scores, dim=-1)
            context = torch.matmul(probs, v.float()).to(hidden.dtype)

        context = context.permute(0, 2, 1, 3).contiguous().reshape(batch, length, -1)
        projected: Tensor = self.output(context)
        return projected * valid.unsqueeze(-1).to(projected.dtype)


class Layer(nn.Module):
    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.attention = Attention(config)
        self.attention_route = DepthAttention(width, config.layer_norm_eps)
        self.feedforward_route = DepthAttention(width, config.layer_norm_eps)
        self.attention_norm = RMSNorm(width, config.layer_norm_eps)
        self.feedforward_norm = RMSNorm(width, config.layer_norm_eps)
        self.gate_up = nn.Linear(width, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, width, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        blocks: list[Tensor],
        partial: Tensor | None,
        relative: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        mixed = self.attention_route(blocks, partial)
        delta = self.dropout(self.attention(self.attention_norm(mixed), relative, attention_mask))
        partial = delta if partial is None else partial + delta
        mixed_ff = self.feedforward_route(blocks, partial)
        projected = self.gate_up(self.feedforward_norm(mixed_ff))
        gate, value = projected.chunk(2, dim=-1)
        change = self.down(functional.silu(gate) * value)
        result: Tensor = partial + self.dropout(change)
        return result


class MasinissaModel(PreTrainedModel):
    config_class = MasinissaConfig
    base_model_prefix = "masinissa"

    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.relative = nn.Parameter(
            torch.empty(2 * config.position_buckets - 1, config.hidden_size)
        )
        self.position_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.position_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.final_route = DepthAttention(config.hidden_size, config.layer_norm_eps)
        self.final_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = math.sqrt(2.0 / (5.0 * self.config.hidden_size))
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
            if getattr(module, "bias", None) is not None and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, MasinissaModel):
            nn.init.trunc_normal_(module.relative, std=std, a=-2 * std, b=2 * std)
        elif isinstance(module, DepthAttention):
            nn.init.zeros_(module.query)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embedding = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput | tuple[Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        embedded = self.embedding(input_ids)
        blocks = [self.dropout(embedded)]
        partial: Tensor | None = None

        all_hidden_states: list[Tensor] | None = [embedded] if output_hidden_states else None

        head_shape = (
            2 * self.config.position_buckets - 1,
            self.config.num_attention_heads,
            self.config.head_size,
        )
        pq = self.position_q(self.relative).view(head_shape).transpose(0, 1)
        pk = self.position_k(self.relative).view(head_shape).transpose(0, 1)
        relative = (pq, pk)

        for index, layer in enumerate(self.layers):
            partial = layer(blocks, partial, relative, attention_mask)
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states.append(partial)
            if (index + 1) % self.config.block_layers == 0:
                blocks.append(partial)
                partial = None

        final_out = self.final_norm(self.final_route(blocks, partial))
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states.append(final_out)

        if not return_dict:
            return (final_out,) + (() if all_hidden_states is None else (tuple(all_hidden_states),))

        return BaseModelOutput(
            last_hidden_state=final_out,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class MasinissaForMaskedLM(PreTrainedModel):
    config_class = MasinissaConfig
    base_model_prefix = "masinissa"

    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__(config)
        self.masinissa = MasinissaModel(config)
        self.head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.head_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.vocabulary_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.masinissa.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.masinissa.embedding = value

    def get_output_embeddings(self) -> nn.Linear | None:
        return None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> MaskedLMOutput | tuple[Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.masinissa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        projected = self.head_norm(functional.gelu(self.head(hidden_states)))
        # Tied prediction logits
        logits = functional.linear(projected, self.masinissa.embedding.weight, self.vocabulary_bias)

        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        if not return_dict:
            output = (logits, *outputs[1:])
            return (loss, *output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


__all__ = ["MasinissaConfig", "MasinissaForMaskedLM", "MasinissaModel"]
