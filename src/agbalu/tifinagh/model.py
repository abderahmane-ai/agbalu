"""Character-level encoder-decoder for script conversion, published as `agbalu/Juba-27M`.

Three choices are load-bearing and the rest is a standard transformer.

The **convolutional stem** runs before the encoder because Tifinagh writes consonant
clusters with the vowel deleted, so the evidence for where a schwa belongs is the
adjacent two to five characters. Depthwise kernels of 3 and 5 hand the first encoder
layer that window already computed instead of spending an attention head on it.

**Rotary positions** rather than learned absolute ones, because the cue is the distance
between consonants and not where in the sentence the cluster sits.

**Tied embedding and output projection.** Input and output alphabets are the same
alphabet, so one matrix is the correct constraint rather than a saving.

Attribute names here are the published checkpoint's `state_dict` keys. Renaming one
silently breaks `load_state_dict` for everybody who downloaded the release.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.tifinagh.config import ModelConfig

ROPE_BASE = 10_000.0


@dataclass(frozen=True, slots=True)
class LossOutput:
    """A step's loss and the counts its accuracy is computed from.

    `correct` and `tokens` stay on the device. Dividing them here would be a host
    synchronisation inside `forward`, which is the defect that cost the encoder run its
    throughput: the caller syncs once when it logs, not once per micro-batch.
    """

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
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        projected: Tensor = self.w3(functional.silu(self.w1(x)) * self.w2(x))
        return projected


class ConvStem(nn.Module):
    """Depthwise separable convolutions of width 3 and 5, added to the embedding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        self.norm = RMSNorm(dim, eps=config.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        channels = x.transpose(1, 2)
        window = torch.cat(
            [self.conv3(channels).transpose(1, 2), self.conv5(channels).transpose(1, 2)],
            dim=-1,
        )
        stemmed: Tensor = self.norm(x + self.proj(window))
        return stemmed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        # `get_buffer` rather than the attribute: `nn.Module.__getattr__` is typed as
        # `Tensor | Module`, and a registered buffer is always the former.
        inv_freq = self.get_buffer("inv_freq")
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        doubled = torch.cat((angles, angles), dim=-1)
        return doubled.cos(), doubled.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Self- or cross-attention. Rotary positions apply to self-attention only.

    Cross-attention relates two different sequences, so a shared position index would
    claim character `i` of the Tifinagh source aligns with character `i` of the Latin
    target — which is exactly the claim schwa restoration falsifies.
    """

    def __init__(self, config: ModelConfig, *, is_cross: bool = False) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_size
        self.hidden_size = config.hidden_size
        self.is_cross = is_cross

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _heads(self, projected: Tensor, length: int) -> Tensor:
        batch = projected.shape[0]
        return projected.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        batch, seq_len, _ = x.shape
        source = context if (self.is_cross and context is not None) else x
        kv_len = source.shape[1]

        q = self._heads(self.q_proj(x), seq_len)
        k = self._heads(self.k_proj(source), kv_len)
        v = self._heads(self.v_proj(source), kv_len)

        if rope_cos is not None and rope_sin is not None and not self.is_cross:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        attended = functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        projected: Tensor = self.out_proj(merged)
        return projected


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        attended = x + self.self_attn(self.norm1(x), rope_cos=cos, rope_sin=sin)
        hidden: Tensor = attended + self.mlp(self.norm2(attended))
        return hidden


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = Attention(config, is_cross=True)
        self.norm3 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor, context: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        attended = x + self.self_attn(self.norm1(x), rope_cos=cos, rope_sin=sin, is_causal=True)
        crossed = attended + self.cross_attn(self.norm2(attended), context=context)
        hidden: Tensor = crossed + self.mlp(self.norm3(crossed))
        return hidden


class CharTransformer(nn.Module):
    """Encoder-decoder over characters. 26,901,888 parameters at the release config."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.conv_stem = ConvStem(config)
        self.rope = RotaryEmbedding(config.head_size)

        self.encoder_layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        )

        self.enc_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=0.02)

    def encode(self, src_ids: Tensor) -> Tensor:
        x = self.conv_stem(self.embed(src_ids))
        cos, sin = self.rope(src_ids.shape[1], src_ids.device)
        for layer in self.encoder_layers:
            x = layer(x, cos, sin)
        encoded: Tensor = self.enc_norm(x)
        return encoded

    def decode(self, tgt_ids: Tensor, context: Tensor) -> Tensor:
        x = self.embed(tgt_ids)
        cos, sin = self.rope(tgt_ids.shape[1], tgt_ids.device)
        for layer in self.decoder_layers:
            x = layer(x, context, cos, sin)
        logits: Tensor = self.lm_head(self.dec_norm(x))
        return logits

    def forward(self, src_ids: Tensor, tgt_input_ids: Tensor) -> Tensor:
        """Teacher-forced logits. `loss_with_logits` is the training entry point."""
        return self.decode(tgt_input_ids, self.encode(src_ids))

    def loss_with_logits(
        self, src_ids: Tensor, tgt_input_ids: Tensor, labels: Tensor
    ) -> tuple[LossOutput, Tensor]:
        """The training objective, returning the logits it was computed from.

        Returned rather than recomputed: an evaluation pass that calls `forward` for the
        loss and `decode` again for the predictions runs the model twice over the same
        batch, which is what the first test-split pass did.
        """
        logits = self.decode(tgt_input_ids, self.encode(src_ids))
        loss = functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            labels.reshape(-1),
            ignore_index=self.config.pad_token_id,
            label_smoothing=self.config.label_smoothing,
        )
        with torch.no_grad():
            scored = labels != self.config.pad_token_id
            correct = ((logits.argmax(dim=-1) == labels) & scored).sum()
        return LossOutput(loss=loss, correct=correct, tokens=scored.sum()), logits
