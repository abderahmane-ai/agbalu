"""Masinissa-31M: an LTG-BERT masked language model for Kabyle.

Module attribute names are the published checkpoint's `state_dict` keys. The relative
position index tables are derived, not learned — twelve identical 512x512 int64 lookups
that would add 25.2 MB to a 124.5 MB download — so they are absent from
`model.safetensors` on purpose and rebuilt here on first use.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from .configuration_masinissa import MasinissaConfig

IGNORE_INDEX = -100


def log_bucket_position(relative_position: Tensor, bucket_size: int, max_position: int) -> Tensor:
    """Relative offsets to bucket indices: linear near the diagonal, logarithmic beyond.

    Neighbouring tokens get their own bucket and distant ones share, which is what lets a
    32-bucket table cover a 512-token window.
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


class GeGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * functional.gelu(gate, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, config: MasinissaConfig) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(self.norm_in(x))
        x = self.norm_mid(self.gate(x))
        out: Tensor = self.dropout(self.down(x))
        return out


class Attention(nn.Module):
    def __init__(self, config: MasinissaConfig) -> None:
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

        self._indices: Tensor | None = None

    def _position_indices(self, length: int) -> Tensor:
        offsets = torch.arange(length).unsqueeze(1) - torch.arange(length).unsqueeze(0)
        buckets = log_bucket_position(
            offsets, self.config.position_bucket_size, self.config.max_position_embeddings
        )
        return self.config.position_bucket_size - 1 + buckets

    def position_indices(self, seq_len: int, device: torch.device) -> Tensor:
        """The bucket table, built on first use and kept on a plain attribute.

        Not a registered buffer. `from_pretrained` allocates buffers empty and fills them
        from the checkpoint, and this table is derived rather than stored — twelve
        identical 512x512 int64 lookups would add 25.2 MB to a 124.5 MB download — so as a
        non-persistent buffer it comes back as uninitialised memory and indexes out of
        range. A plain attribute is invisible to that machinery.
        """
        cached = self._indices
        if cached is None or cached.size(0) < seq_len or cached.device != device:
            width = max(seq_len, self.config.max_position_embeddings)
            cached = self._position_indices(width).to(device)
            self._indices = cached
        return cached

    def forward(self, x: Tensor, attention_mask: Tensor, relative_embedding: Tensor) -> Tensor:
        """`x` is `[T, B, D]`; `attention_mask` is `[B, 1, 1, T]` and True where padded."""
        seq_len, batch_size, _ = x.size()
        indices = self.position_indices(seq_len, x.device)

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
    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__()
        self.word_embedding: nn.Module = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relative_embedding = nn.Parameter(
            torch.empty(2 * config.position_bucket_size - 1, config.hidden_size)
        )
        self.relative_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        words = self.dropout(self.word_norm(self.word_embedding(input_ids)))
        relative: Tensor = self.relative_norm(self.relative_embedding)
        return words, relative


class Transformer(nn.Module):
    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__()
        self.attention_layers = nn.ModuleList(
            Attention(config) for _ in range(config.num_hidden_layers)
        )
        self.feed_forward_layers = nn.ModuleList(
            FeedForward(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, x: Tensor, attention_mask: Tensor, relative_embedding: Tensor) -> Tensor:
        for attention, feed_forward in zip(
            self.attention_layers, self.feed_forward_layers, strict=True
        ):
            x = x + attention(x, attention_mask, relative_embedding)
            x = x + feed_forward(x)
        return x


class MaskClassifier(nn.Module):
    def __init__(self, config: MasinissaConfig) -> None:
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
        self.decoder: nn.Module = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.dense(self.norm_in(x)))
        logits: Tensor = self.decoder(self.dropout(self.norm_out(x)))
        return logits


class MasinissaPreTrainedModel(PreTrainedModel):
    config_class = MasinissaConfig
    base_model_prefix = "masinissa"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        std = math.sqrt(2.0 / (5.0 * self.config.hidden_size))
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            nn.init.trunc_normal_(
                module.relative_embedding, mean=0.0, std=std, a=-2 * std, b=2 * std
            )


def _padding_mask(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    """`[B, 1, 1, T]`, True where the position is padding.

    `~` on an int tensor is a bitwise complement, not a logical one, so the cast is
    load-bearing: `attention_mask` arrives from a tokenizer as 1/0 integers.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return ~attention_mask.bool()[:, None, None, :]


class MasinissaModel(MasinissaPreTrainedModel):
    """The encoder alone, returning contextualised token representations."""

    _keys_to_ignore_on_load_unexpected = [r"^classifier\."]

    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__(config)
        self.embedding = Embedding(config)
        self.transformer = Transformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding.word_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embedding.word_embedding = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> BaseModelOutput:
        words, relative = self.embedding(input_ids)
        hidden = self.transformer(
            words.transpose(0, 1), _padding_mask(input_ids, attention_mask), relative
        )
        return BaseModelOutput(last_hidden_state=hidden.transpose(0, 1))


class MasinissaForMaskedLM(MasinissaPreTrainedModel):
    """The pretraining model: the encoder plus the tied masked-token classifier."""

    _tied_weights_keys = {"classifier.decoder.weight": "embedding.word_embedding.weight"}

    def __init__(self, config: MasinissaConfig) -> None:
        super().__init__(config)
        self.embedding = Embedding(config)
        self.transformer = Transformer(config)
        self.classifier = MaskClassifier(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding.word_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embedding.word_embedding = value

    def get_output_embeddings(self) -> nn.Module:
        return self.classifier.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.classifier.decoder = new_embeddings

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs: object,
    ) -> MaskedLMOutput:
        words, relative = self.embedding(input_ids)
        hidden = self.transformer(
            words.transpose(0, 1), _padding_mask(input_ids, attention_mask), relative
        )
        logits = self.classifier(hidden.transpose(0, 1))
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return MaskedLMOutput(loss=loss, logits=logits)


__all__ = [
    "MasinissaConfig",
    "MasinissaForMaskedLM",
    "MasinissaModel",
    "MasinissaPreTrainedModel",
]
