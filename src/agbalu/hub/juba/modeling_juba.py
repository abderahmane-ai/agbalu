"""Juba-27M: a character encoder-decoder for Neo-Tifinagh to Kabyle Latin.

Module attribute names are the published checkpoint's `state_dict` keys. Renaming one
breaks `from_pretrained` for everybody who downloaded the release.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .configuration_juba import JubaConfig

ROPE_BASE = 10_000.0


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
    """Depthwise separable convolutions of width 3 and 5, added to the embedding.

    Tifinagh writes consonant clusters with the vowel deleted, so the evidence for where a
    schwa belongs is the adjacent two to five characters.
    """

    def __init__(self, config: JubaConfig) -> None:
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
    """The inverse frequencies live on a plain attribute, not a registered buffer.

    They are derived from `head_size` and absent from `model.safetensors` by design, and
    `from_pretrained` allocates buffers empty and fills them from the checkpoint — so as a
    non-persistent buffer this comes back as uninitialised memory, which rotates every
    query and key by a garbage angle and raises nothing.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self._inv_freq: Tensor | None = None

    def inv_freq(self, device: torch.device) -> Tensor:
        cached = self._inv_freq
        if cached is None or cached.device != device:
            steps = torch.arange(0, self.dim, 2, device=device).float()
            cached = 1.0 / (ROPE_BASE ** (steps / self.dim))
            self._inv_freq = cached
        return cached

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq(device)
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

    Cross-attention relates two sequences of different lengths, so a shared position index
    would assert an alignment that schwa restoration falsifies.
    """

    def __init__(self, config: JubaConfig, *, is_cross: bool = False) -> None:
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
    def __init__(self, config: JubaConfig) -> None:
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
    def __init__(self, config: JubaConfig) -> None:
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


class JubaEncoderAdapter:
    """The callable `generate` fetches from `get_encoder`.

    Not an `nn.Module`: the encoder weights are the parent's, registered at the names the
    published checkpoint uses, and a module holding a reference back to the parent would
    publish every one of them a second time under this attribute.
    """

    def __init__(self, model: JubaForSeq2SeqLM) -> None:
        self._model = model

    def forward(self, input_ids: Tensor, **kwargs: object) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self._model.encode(input_ids))

    def __call__(self, input_ids: Tensor, **kwargs: object) -> BaseModelOutput:
        return self.forward(input_ids, **kwargs)


class JubaPreTrainedModel(PreTrainedModel):
    config_class = JubaConfig
    base_model_prefix = "juba"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        for parameter in module.parameters(recurse=False):
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=0.02)


class JubaForSeq2SeqLM(JubaPreTrainedModel, GenerationMixin):
    """26,901,888 parameters at the release config. The output projection is the input
    embedding, so `lm_head.weight` is absent from `model.safetensors` and re-tied here."""

    _tied_weights_keys = {"lm_head.weight": "embed.weight"}

    def __init__(self, config: JubaConfig) -> None:
        super().__init__(config)
        # Annotated `nn.Module` rather than `nn.Embedding` because `set_input_embeddings`
        # is `transformers`' resize-and-retie hook and hands back whatever it built.
        self.embed: nn.Module = nn.Embedding(
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
        self.lm_head: nn.Module = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed = value

    def get_output_embeddings(self) -> nn.Module:
        head: nn.Module = self.lm_head
        return head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def get_encoder(self, modality: str | None = None) -> JubaEncoderAdapter:
        return JubaEncoderAdapter(self)

    def get_decoder(self) -> JubaForSeq2SeqLM:
        return self

    def encode(self, input_ids: Tensor) -> Tensor:
        x = self.conv_stem(self.embed(input_ids))
        cos, sin = self.rope(input_ids.shape[1], input_ids.device)
        for layer in self.encoder_layers:
            x = layer(x, cos, sin)
        encoded: Tensor = self.enc_norm(x)
        return encoded

    def decode(self, decoder_input_ids: Tensor, context: Tensor) -> Tensor:
        x = self.embed(decoder_input_ids)
        cos, sin = self.rope(decoder_input_ids.shape[1], decoder_input_ids.device)
        for layer in self.decoder_layers:
            x = layer(x, context, cos, sin)
        logits: Tensor = self.lm_head(self.dec_norm(x))
        return logits

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        labels: Tensor | None = None,
        **kwargs: object,
    ) -> Seq2SeqLMOutput:
        """`attention_mask` is accepted and ignored.

        The published model attends over padding: `Transliterator._encode_batch` pads with
        `pad_token_id` and passes no mask, so honouring one here would return different
        logits from the weights' own evaluation. Batch rows of equal length, or decode one
        at a time, to reproduce the card's numbers exactly.
        """
        if encoder_outputs is None:
            if input_ids is None:
                message = "one of input_ids or encoder_outputs is required"
                raise ValueError(message)
            encoder_outputs = BaseModelOutput(last_hidden_state=self.encode(input_ids))
        if decoder_input_ids is None:
            message = "decoder_input_ids is required; `generate` supplies it"
            raise ValueError(message)

        logits = self.decode(decoder_input_ids, encoder_outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
                label_smoothing=self.config.label_smoothing,
            )
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: Tensor,
        encoder_outputs: BaseModelOutput | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Overridden rather than inherited: this model has no key-value cache, so every
        step re-reads the whole prefix and the inherited cache-slicing default would hand
        the decoder one token."""
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "use_cache": False,
        }


__all__ = ["JubaConfig", "JubaForSeq2SeqLM", "JubaPreTrainedModel"]
