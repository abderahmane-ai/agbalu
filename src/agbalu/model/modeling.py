"""LTG-BERT/GPT-BERT encoder, reimplemented and typed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.model.config import ModelConfig

IGNORE_INDEX: Final = -100
"""Label value excluded from the loss, matching `torch.nn.functional.cross_entropy`."""


@dataclass(frozen=True, slots=True)
class LossOutput:
    loss: Tensor
    z_loss: Tensor
    accuracy: Tensor
    """A tensor, not a float: reading it on the host forces a device sync, and the
    training loop accumulates over micro-batches and syncs once per optimiser step."""
    num_tokens: int


def log_bucket_position(relative_position: Tensor, bucket_size: int, max_position: int) -> Tensor:
    """Relative offsets to bucket indices: linear near the diagonal, logarithmic beyond.

    Verbatim in behaviour from the reference `make_log_bucket_position`. The effect is
    that neighbouring tokens get their own bucket and distant ones share, which is what
    lets a 32-bucket table cover a 512-token window.
    """
    sign = torch.sign(relative_position)
    mid = bucket_size // 2
    absolute = torch.where(
        (relative_position < mid) & (relative_position > -mid),
        torch.full_like(relative_position, mid - 1),
        torch.abs(relative_position).clamp(max=max_position - 1),
    )
    log_position = (
        torch.ceil(
            torch.log(absolute.float() / mid) / math.log((max_position - 1) / mid) * (mid - 1)
        ).int()
        + mid
    )
    return torch.where(absolute <= mid, relative_position, log_position * sign).long()


def _trunc_normal_(weight: Tensor, hidden_size: int) -> None:
    std = math.sqrt(2.0 / (5.0 * hidden_size))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


class GeGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * functional.gelu(gate, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.gate = GeGLU()
        self.norm_mid = nn.LayerNorm(
            config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        _trunc_normal_(self.up.weight, config.hidden_size)
        _trunc_normal_(self.down.weight, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(self.norm_in(x))
        x = self.norm_mid(self.gate(x))
        out: Tensor = self.dropout(self.down(x))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size

        self.in_proj_qk = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.pre_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.post_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)

        self.register_buffer(
            "position_indices", self._position_indices(config.max_position_embeddings)
        )

        for projection in (self.in_proj_qk, self.in_proj_v, self.out_proj):
            _trunc_normal_(projection.weight, config.hidden_size)
            nn.init.zeros_(projection.bias)

    def _position_indices(self, length: int) -> Tensor:
        offsets = torch.arange(length).unsqueeze(1) - torch.arange(length).unsqueeze(0)
        buckets = log_bucket_position(
            offsets, self.config.position_bucket_size, self.config.max_position_embeddings
        )
        return self.config.position_bucket_size - 1 + buckets

    def forward(self, x: Tensor, attention_mask: Tensor, relative_embedding: Tensor) -> Tensor:
        """`x` is `[T, B, D]`; `attention_mask` is `[B, 1, 1, T]` and True where padded."""
        seq_len, batch_size, _ = x.size()
        indices: Tensor = self.get_buffer("position_indices")
        if indices.size(0) < seq_len:
            indices = self._position_indices(seq_len).to(x.device)

        x = self.pre_norm(x)
        query, key = self.in_proj_qk(x).chunk(2, dim=2)
        value = self.in_proj_v(x)

        position = self.in_proj_qk(self.dropout(relative_embedding))
        position = functional.embedding(indices[:seq_len, :seq_len], position)
        query_pos, key_pos = position.chunk(2, dim=-1)
        query_pos = query_pos.view(seq_len, seq_len, self.num_heads, self.head_size)
        key_pos = key_pos.view(seq_len, seq_len, self.num_heads, self.head_size)

        shape = (seq_len, batch_size * self.num_heads, self.head_size)
        query = query.reshape(shape).transpose(0, 1)
        key = key.reshape(shape).transpose(0, 1)
        value = value.reshape(shape).transpose(0, 1)

        scores = torch.bmm(query, key.transpose(1, 2) * self.scale)
        scores = scores.view(batch_size, self.num_heads, seq_len, seq_len)
        query = query.view(batch_size, self.num_heads, seq_len, self.head_size)
        key = key.view(batch_size, self.num_heads, seq_len, self.head_size)
        scores = scores + torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale)
        scores = scores + torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos)

        scores = scores.masked_fill(attention_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        probabilities = self.dropout(probabilities)

        context = torch.bmm(probabilities.flatten(0, 1), value)
        context = context.transpose(0, 1).reshape(seq_len, batch_size, -1)
        projected: Tensor = self.dropout(self.out_proj(self.post_norm(context)))
        return projected


class Embedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relative_embedding = nn.Parameter(
            torch.empty(2 * config.position_bucket_size - 1, config.hidden_size)
        )
        self.relative_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        _trunc_normal_(self.word_embedding.weight, config.hidden_size)
        _trunc_normal_(self.relative_embedding, config.hidden_size)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        words = self.dropout(self.word_norm(self.word_embedding(input_ids)))
        relative: Tensor = self.relative_norm(self.relative_embedding)
        return words, relative


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        feed_forward = [FeedForward(config) for _ in range(config.num_hidden_layers)]
        for depth, layer in enumerate(feed_forward):
            scale = math.sqrt(1.0 / (2.0 * (1 + depth)))
            with torch.no_grad():
                layer.up.weight.mul_(scale)
                layer.down.weight.mul_(scale)
        self.attention_layers = nn.ModuleList(
            Attention(config) for _ in range(config.num_hidden_layers)
        )
        self.feed_forward_layers = nn.ModuleList(feed_forward)

    def forward(self, x: Tensor, attention_mask: Tensor, relative_embedding: Tensor) -> Tensor:
        for attention, feed_forward in zip(
            self.attention_layers, self.feed_forward_layers, strict=True
        ):
            x = x + attention(x, attention_mask, relative_embedding)
            x = x + feed_forward(x)
        return x


class MaskClassifier(nn.Module):
    def __init__(self, config: ModelConfig, word_embedding: Tensor) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.norm_out = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        _trunc_normal_(self.dense.weight, config.hidden_size)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.decoder.bias)
        self.decoder.weight = cast(nn.Parameter, word_embedding)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.dense(self.norm_in(x)))
        logits: Tensor = self.decoder(self.dropout(self.norm_out(x)))
        return logits


class Encoder(nn.Module):
    """The pretraining model: encoder plus the tied masked-token classifier."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.transformer = Transformer(config)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)

    def contextualise(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """`input_ids` `[B, T]`, `attention_mask` `[B, T]` True where real. Returns `[B, T, D]`."""
        words, relative = self.embedding(input_ids)
        padding = ~attention_mask[:, None, None, :]
        hidden: Tensor = self.transformer(words.transpose(0, 1), padding, relative)
        return hidden.transpose(0, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> LossOutput:
        hidden = self.contextualise(input_ids, attention_mask)
        selected = labels != IGNORE_INDEX
        if not bool(selected.any()):
            zero = hidden.sum() * 0.0
            return LossOutput(loss=zero, z_loss=zero, accuracy=zero, num_tokens=0)

        logits = self.classifier(hidden[selected])
        gold = labels[selected]
        loss = functional.cross_entropy(logits, gold)
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        with torch.no_grad():
            accuracy = (logits.argmax(-1) == gold).float().mean()
        return LossOutput(loss=loss, z_loss=z_loss, accuracy=accuracy, num_tokens=int(gold.numel()))

    @torch.no_grad()
    def predict_masked(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
        """Predicted piece id at each masked position, `IGNORE_INDEX` elsewhere.

        Scores only the masked positions, as `forward` does — a full `[B, T, V]` logit
        tensor is not needed to read off an argmax.
        """
        hidden = self.contextualise(input_ids, attention_mask)
        selected = labels != IGNORE_INDEX
        predictions = torch.full_like(labels, IGNORE_INDEX)
        if bool(selected.any()):
            predictions[selected] = self.classifier(hidden[selected]).argmax(-1)
        return predictions

    def parameter_count(self) -> int:
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
        return total
