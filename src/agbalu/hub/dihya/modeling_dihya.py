"""Dihya-5M: byte-level Berber language identification.

Module attribute names are the published checkpoint's state_dict keys. Renaming one
breaks `from_pretrained` for everybody who downloaded the release.

The contrastive projection head the model was trained with is not here. It shapes the
trunk during training and is never read at inference, so shipping it would hand every
downloader 98,304 parameters that no forward pass touches.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_dihya import DihyaConfig

MASK_FILL = -1e4
"""Finite rather than `-inf`: a row that is entirely padding would otherwise softmax to
NaN, and an empty string is a real input to a language identifier."""


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        normed: Tensor = x * torch.rsqrt(variance + self.eps) * self.weight
        return normed


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        projected: Tensor = self.w3(functional.silu(self.w1(x)) * self.w2(x))
        return projected


class ConvStem(nn.Module):
    """Parallel depthwise-separable convolutions over the byte embeddings.

    Three widths because the discriminating evidence sits at three scales: a grapheme
    cluster, an affix, and a clitic chain. One kernel width picks one of the three.
    """

    def __init__(self, config: DihyaConfig) -> None:
        super().__init__()
        dim, branch_dim = config.hidden_size, config.conv_dim
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False),
                nn.Conv1d(dim, branch_dim, kernel_size=1, bias=False),
            )
            for k in config.conv_kernels
        )
        self.branch_norms = nn.ModuleList(
            RMSNorm(branch_dim, eps=config.rms_norm_eps) for _ in config.conv_kernels
        )
        self.proj = nn.Linear(branch_dim * len(config.conv_kernels), dim, bias=False)
        self.norm = RMSNorm(dim, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        transposed = x.transpose(1, 2)
        outputs = [
            functional.silu(norm(branch(transposed).transpose(1, 2)))
            for branch, norm in zip(self.branches, self.branch_norms, strict=True)
        ]
        stemmed: Tensor = x + self.dropout(self.norm(self.proj(torch.cat(outputs, dim=-1))))
        return stemmed


def rope_freqs(head_dim: int, length: int, base: float, device: torch.device) -> Tensor:
    """Complex rotary frequencies, derived on call and never stored.

    A registered non-persistent buffer comes back from `from_pretrained` as uninitialised
    memory, because it is deliberately absent from the checkpoint.
    """
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(length, device=device).float()
    angles = torch.outer(positions, theta)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    batch, heads, length, head_dim = x.shape
    paired = torch.view_as_complex(x.float().reshape(batch, heads, length, -1, 2))
    rotated = torch.view_as_real(paired * freqs[:length].unsqueeze(0).unsqueeze(0))
    return rotated.reshape(batch, heads, length, head_dim).type_as(x)


class Attention(nn.Module):
    def __init__(self, config: DihyaConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_size
        self.dropout = config.dropout_prob
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, length, dim = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = apply_rope(self.q_proj(x).view(shape).transpose(1, 2), freqs)
        key = apply_rope(self.k_proj(x).view(shape).transpose(1, 2), freqs)
        value = self.v_proj(x).view(shape).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask.unsqueeze(1).unsqueeze(2) if mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        merged: Tensor = self.out_proj(attended.transpose(1, 2).reshape(batch, length, dim))
        return merged


class EncoderLayer(nn.Module):
    def __init__(self, config: DihyaConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: Tensor, freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden: Tensor = x + self.dropout(self.attn(self.norm1(x), freqs, mask=mask))
        residual: Tensor = self.dropout(self.ffn(self.norm2(hidden)))
        return hidden + residual


class AttentivePooling(nn.Module):
    """Weighted sum over positions, so a short discriminating affix is not averaged away."""

    def __init__(self, config: DihyaConfig) -> None:
        super().__init__()
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        scores = self.score(x).squeeze(-1) / math.sqrt(x.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, MASK_FILL)
        weights = functional.softmax(scores, dim=-1).unsqueeze(-1)
        pooled: Tensor = (x * weights).sum(dim=1)
        return pooled


class MarginHead(nn.Module):
    """Scaled cosine classifier.

    The per-class additive margins the head was trained with apply to the target logit
    only, so they exist during training and are identity at inference. The margin buffer
    is therefore not part of the release.
    """

    def __init__(self, config: DihyaConfig) -> None:
        super().__init__()
        self.scale = config.logit_scale
        self.weight = nn.Parameter(torch.empty(len(config.classes), config.hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        cosine = functional.linear(
            functional.normalize(x, p=2, dim=1), functional.normalize(self.weight, p=2, dim=1)
        )
        return cosine * self.scale


class DihyaPreTrainedModel(PreTrainedModel):
    config_class = DihyaConfig
    base_model_prefix = "dihya"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, MarginHead):
            nn.init.xavier_uniform_(module.weight)


class DihyaForSequenceClassification(DihyaPreTrainedModel):
    """Byte-level classifier over six Berber varieties and an explicit rejection class."""

    def __init__(self, config: DihyaConfig) -> None:
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.stem = ConvStem(config)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pool = AttentivePooling(config)
        self.head = MarginHead(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed = value  # type: ignore[assignment]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[Tensor, ...]:
        mask = attention_mask.bool() if attention_mask is not None else None
        hidden = self.stem(self.embed(input_ids))
        freqs = rope_freqs(
            self.config.head_size,
            input_ids.size(1),
            self.config.rope_theta,
            input_ids.device,
        )
        for layer in self.layers:
            hidden = layer(hidden, freqs, mask=mask)
        pooled = self.pool(self.final_norm(hidden), mask=mask)
        logits = self.head(pooled)

        loss = None
        if labels is not None:
            loss = functional.cross_entropy(logits, labels)
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=(pooled,))

    def prior_shift(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.tensor(self.config.prior_shift, device=device, dtype=dtype)

    @torch.inference_mode()
    def identify(
        self,
        texts: str | list[str],
        max_length: int | None = None,
        batch_size: int = 128,
    ) -> list[dict[str, Any]]:
        """Classify text. One dict per input: `language`, `confidence`, `prediction_set`.

        The tokenizer is not needed: the vocabulary is the 256 UTF-8 byte values, so the
        encoding is the input's own bytes. `prediction_set` is the split-conformal set
        `{k : p_k >= 1 - q_hat}` when the repository carries a calibrated `q_hat`, and the
        argmax alone when it does not — never a singleton dressed up as a guarantee.
        """
        wanted = [texts] if isinstance(texts, str) else list(texts)
        if not wanted:
            return []
        limit = max_length or self.config.max_position_embeddings
        device = next(self.parameters()).device
        classes = list(self.config.classes)
        threshold = None if self.config.q_hat is None else 1.0 - float(self.config.q_hat)

        results: list[dict[str, Any]] = []
        for start in range(0, len(wanted), batch_size):
            chunk = wanted[start : start + batch_size]
            rows = [
                [b + self.config.byte_offset for b in text.encode("utf-8")[:limit]]
                or [self.config.pad_token_id]
                for text in chunk
            ]
            width = max(len(row) for row in rows)
            input_ids = torch.full(
                (len(rows), width), self.config.pad_token_id, dtype=torch.long, device=device
            )
            attention = torch.zeros((len(rows), width), dtype=torch.long, device=device)
            for i, row in enumerate(rows):
                input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                attention[i, : len(row)] = 1

            logits = self(input_ids, attention_mask=attention).logits
            shifted = logits - self.prior_shift(logits.device, logits.dtype)
            for probabilities in torch.softmax(shifted, dim=-1).tolist():
                top = max(range(len(probabilities)), key=probabilities.__getitem__)
                members = (
                    tuple(c for c, p in zip(classes, probabilities, strict=True) if p >= threshold)
                    if threshold is not None
                    else (classes[top],)
                )
                results.append(
                    {
                        "language": classes[top],
                        "confidence": probabilities[top],
                        "prediction_set": members or (classes[top],),
                        "probabilities": dict(zip(classes, probabilities, strict=True)),
                    }
                )
        return results


__all__ = [
    "DihyaConfig",
    "DihyaForSequenceClassification",
    "DihyaPreTrainedModel",
]
