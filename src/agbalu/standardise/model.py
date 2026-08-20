"""Character-level encoder-decoder for orthography standardisation.

Published as `agbalu/Boulifa-48M`. The release name is not a code identifier (§3.9), so
the classes here are named for what they configure.

Features:
- Depthwise convolutional stem (kernels 3, 5) for local multi-character digram/trigram context
- Rotary Position Embeddings (RoPE)
- SwiGLU non-linearities and RMSNorm
- Tied embeddings and output projection
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.standardise.config import ModelConfig

ROPE_BASE = 10_000.0


@dataclass(frozen=True, slots=True)
class LossOutput:
    """A step's loss and token counts."""

    loss: Tensor
    correct: Tensor
    tokens: Tensor


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
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        projected: Tensor = self.down_proj(gate * up)
        return projected


def precompute_rope_freqs(dim: int, max_len: int, base: float = ROPE_BASE) -> Tensor:
    half = dim // 2
    theta = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    positions = torch.arange(max_len, dtype=torch.float32)
    return torch.outer(positions, theta)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    _batch, seq, _heads, dim = x.shape
    half = dim // 2
    x_1, x_2 = x[..., :half], x[..., half:]
    f = freqs[:seq].unsqueeze(0).unsqueeze(2)
    cos = f.cos()
    sin = f.sin()
    rotated_1 = x_1 * cos - x_2 * sin
    rotated_2 = x_1 * sin + x_2 * cos
    return torch.cat([rotated_1, rotated_2], dim=-1)


class ConvStem(nn.Module):
    """1D depthwise convolutions over characters before the first transformer layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        xt = x.transpose(1, 2)
        c3 = self.conv3(xt).transpose(1, 2)
        c5 = self.conv5(xt).transpose(1, 2)
        out = self.proj1(c3) + self.proj2(c5)
        stemmed: Tensor = self.norm(out + residual)
        return stemmed


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, *, is_cross: bool = False) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_size
        self.is_cross = is_cross

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor | None = None,
        *,
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch, q_seq, _ = query.shape
        kv = query if key_value is None else key_value
        _, kv_seq, _ = kv.shape

        q = self.q_proj(query).view(batch, q_seq, self.num_heads, self.head_dim)
        k = self.k_proj(kv).view(batch, kv_seq, self.num_heads, self.head_dim)
        v = self.v_proj(kv).view(batch, kv_seq, self.num_heads, self.head_dim)

        if freqs is not None and not self.is_cross:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / (self.head_dim**0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        probs = functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(probs, v).transpose(1, 2).contiguous()
        context = context.view(batch, q_seq, -1)
        out: Tensor = self.o_proj(context)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Attention(config, is_cross=False)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor, freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        h = x + self.attn(self.norm1(x), freqs=freqs, mask=mask)
        out: Tensor = h + self.ffn(self.norm2(h))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Attention(config, is_cross=False)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = Attention(config, is_cross=True)
        self.norm3 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        *,
        freqs: Tensor,
        self_mask: Tensor | None = None,
        cross_mask: Tensor | None = None,
    ) -> Tensor:
        h = x + self.self_attn(self.norm1(x), freqs=freqs, mask=self_mask)
        h = h + self.cross_attn(self.norm2(h), memory, mask=cross_mask)
        out: Tensor = h + self.ffn(self.norm3(h))
        return out


class CharTransformer(nn.Module):
    """Character-level sequence-to-sequence standardisation transformer (47.8M parameters)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.stem = ConvStem(self.config.hidden_size)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self.config) for _ in range(self.config.num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(self.config) for _ in range(self.config.num_decoder_layers)]
        )

        self.enc_norm = RMSNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dec_norm = RMSNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

        freqs = precompute_rope_freqs(
            self.config.head_size,
            self.config.max_position_embeddings,
        )
        self.register_buffer("rope_freqs", freqs, persistent=False)

    @property
    def rope(self) -> Tensor:
        # `get_buffer` rather than the attribute: `nn.Module.__getattr__` is typed as
        # `Tensor | Module`, and a registered buffer is always the former.
        return self.get_buffer("rope_freqs")

    def encode(self, input_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.stem(self.embedding(input_ids))
        for layer in self.encoder_layers:
            x = layer(x, self.rope, mask=mask)
        encoded: Tensor = self.enc_norm(x)
        return encoded

    def decode(
        self,
        target_ids: Tensor,
        memory: Tensor,
        *,
        self_mask: Tensor | None = None,
        cross_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(target_ids)
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                freqs=self.rope,
                self_mask=self_mask,
                cross_mask=cross_mask,
            )
        decoded: Tensor = self.dec_norm(x)
        return decoded

    def output_projection(self, x: Tensor) -> Tensor:
        # Tied embedding output
        logits: Tensor = functional.linear(x, self.embedding.weight)
        return logits

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        pad_id: int | None = None,
    ) -> LossOutput:
        pad = self.config.pad_token_id if pad_id is None else pad_id

        # Target shifted right for teacher forcing
        dec_in = target_ids[:, :-1]
        dec_out = target_ids[:, 1:]

        # Causal mask for decoder
        seq_len = dec_in.shape[1]
        causal_mask = (
            torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Padding masks
        enc_pad_mask = (input_ids == pad).unsqueeze(1).unsqueeze(2)
        enc_mask = torch.zeros(
            (input_ids.shape[0], 1, 1, input_ids.shape[1]),
            device=input_ids.device,
        ).masked_fill(enc_pad_mask, float("-inf"))

        cross_mask = torch.zeros(
            (input_ids.shape[0], 1, 1, input_ids.shape[1]),
            device=input_ids.device,
        ).masked_fill(enc_pad_mask, float("-inf"))

        memory = self.encode(input_ids, mask=enc_mask)
        decoded = self.decode(
            dec_in,
            memory,
            self_mask=causal_mask,
            cross_mask=cross_mask,
        )
        logits = self.output_projection(decoded)

        # Loss calculation with label smoothing
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            dec_out.reshape(-1),
            ignore_index=pad,
            label_smoothing=self.config.label_smoothing,
        )

        preds = logits.argmax(dim=-1)
        valid_mask = dec_out != pad
        correct = ((preds == dec_out) & valid_mask).sum()
        tokens = valid_mask.sum()

        return LossOutput(loss=loss, correct=correct, tokens=tokens)
