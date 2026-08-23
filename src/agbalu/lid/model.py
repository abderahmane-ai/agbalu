"""Byte-level classifier for Berber language identification.

Published as `agbalu/Dihya-5M`. Multi-scale character convolutions feed a rotary
transformer encoder and attentive pooling; the head is cosine with per-class additive
margins, and a small projection carries the contrastive objective.

Two heads on one trunk because they want different geometry: the classifier wants the
pooled vector to be linearly separable, and the contrastive term wants it to be
domain-invariant. Applying the contrastive loss to the classifier's own features makes
the two pull on the same vector, which is the ablation the projection head exists to
avoid in every SimCLR-family recipe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.lid.config import ModelConfig

ROPE_BASE: Final[float] = 10_000.0
MASK_FILL: Final = -1e4
COMPLEX_PAIR: Final = 2


@dataclass(frozen=True, slots=True)
class Output:
    """Cosine logits, the pooled trunk vector, and the normalised contrastive view."""

    logits: Tensor
    embeddings: Tensor
    projections: Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.w3(functional.silu(self.w1(x)) * self.w2(x)))


class ConvStem(nn.Module):
    """Parallel depthwise-separable convolutions over the byte embeddings.

    Three widths because the discriminating evidence is at three scales: a single
    grapheme cluster (`ţ` against `t`), an affix (`u-`/`w-` annexation), and a clitic
    chain. One kernel width picks one of the three.
    """

    def __init__(
        self, dim: int, kernels: Sequence[int], branch_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False),
                nn.Conv1d(dim, branch_dim, kernel_size=1, bias=False),
            )
            for k in kernels
        )
        self.branch_norms = nn.ModuleList(RMSNorm(branch_dim) for _ in kernels)
        self.proj = nn.Linear(branch_dim * len(kernels), dim, bias=False)
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        transposed = x.transpose(1, 2)
        outputs = [
            functional.silu(norm(branch(transposed).transpose(1, 2)))
            for branch, norm in zip(self.branches, self.branch_norms, strict=True)
        ]
        return x + cast(Tensor, self.dropout(self.norm(self.proj(torch.cat(outputs, dim=-1)))))


def precompute_rope_freqs(head_dim: int, max_length: int, base: float = ROPE_BASE) -> Tensor:
    """Complex rotary frequencies for every position up to `max_length`."""
    theta = 1.0 / (base ** (torch.arange(0, head_dim, COMPLEX_PAIR).float() / head_dim))
    positions = torch.arange(max_length).float()
    return torch.polar(
        torch.ones_like(torch.outer(positions, theta)), torch.outer(positions, theta)
    )


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary position embedding to a query or key tensor of shape (B, H, L, D)."""
    batch, heads, length, head_dim = x.shape
    paired = torch.view_as_complex(x.float().reshape(batch, heads, length, -1, COMPLEX_PAIR))
    rotated = torch.view_as_real(paired * freqs[:length].unsqueeze(0).unsqueeze(0))
    return rotated.reshape(batch, heads, length, head_dim).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
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
        return cast(Tensor, self.out_proj(attended.transpose(1, 2).reshape(batch, length, dim)))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, intermediate_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, intermediate_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        hidden: Tensor = x + self.dropout(self.attn(self.norm1(x), freqs, mask=mask))
        residual: Tensor = self.dropout(self.ffn(self.norm2(hidden)))
        return hidden + residual


class AttentivePooling(nn.Module):
    """Weighted sum over positions, so a short discriminating affix is not averaged away."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        scores = self.score(x).squeeze(-1) / math.sqrt(x.size(-1))
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask, MASK_FILL)
        weights = functional.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class MarginHead(nn.Module):
    """Cosine classifier with a per-class additive margin.

    The margin is subtracted from the target logit only, so a class with few examples
    is pushed further inside its own decision boundary. `set_margins` writes the LDAM
    values once the class counts are known; until then the margins are zero and the
    head is a plain scaled cosine classifier.
    """

    margins: Tensor

    def __init__(self, in_features: int, num_classes: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.zeros(num_classes))

    def set_margins(self, margins: Tensor) -> None:
        if margins.shape != self.margins.shape:
            msg = f"expected {tuple(self.margins.shape)} margins, got {tuple(margins.shape)}"
            raise ValueError(msg)
        self.margins.copy_(margins.to(self.margins))

    def forward(self, x: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = functional.linear(
            functional.normalize(x, p=2, dim=1), functional.normalize(self.weight, p=2, dim=1)
        )
        if labels is None:
            return cosine * self.scale
        offset = functional.one_hot(labels, num_classes=self.weight.size(0)).type_as(cosine)
        return (cosine - offset * self.margins.to(cosine)) * self.scale


class CharClassifier(nn.Module):
    """The byte-level trunk, classifier head and contrastive projection."""

    rope_freqs: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.stem = ConvStem(
            config.embed_dim, config.conv_kernels, config.conv_dim, dropout=config.dropout
        )
        self.layers = nn.ModuleList(
            EncoderLayer(
                config.embed_dim, config.num_heads, config.intermediate_dim, config.dropout
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.embed_dim)
        self.pool = AttentivePooling(config.embed_dim)
        self.head = MarginHead(config.embed_dim, config.num_classes, config.scale)
        self.projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.projection_dim, bias=False),
        )
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(config.embed_dim // config.num_heads, config.max_length),
            persistent=False,
        )

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, input_ids: Tensor, pad_mask: Tensor | None = None, labels: Tensor | None = None
    ) -> Output:
        hidden = self.stem(self.embed(input_ids))
        for layer in self.layers:
            hidden = layer(hidden, self.rope_freqs, mask=pad_mask)
        pooled = self.pool(self.final_norm(hidden), pad_mask=pad_mask)
        return Output(
            logits=self.head(pooled, labels=labels),
            embeddings=pooled,
            projections=functional.normalize(self.projection(pooled), p=2, dim=1),
        )
