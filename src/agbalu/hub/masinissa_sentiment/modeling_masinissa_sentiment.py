"""Masinissa-Sentiment: three-class Kabyle sentiment over the Masinissa-31M encoder.

Module attribute names are the published checkpoint's `state_dict` keys. Renaming one
breaks `from_pretrained` for everybody who downloaded the release.

The trunk is Masinissa-31M unchanged, carried here rather than imported because the
published repository has neither this project nor the encoder's repository beside it. It
is a copy, and `tests/unit/test_hub_masinissa_sentiment.py` is what holds it to the
original: it runs this module and the training modules over one set of weights and
compares the logits. The relative position index tables are derived, not learned, and are
kept on plain attributes so they stay out of the download.

The masked-language-model head the encoder was pretrained with is not here. It shapes the
trunk during pretraining and no forward pass reads it now, so shipping it would hand every
downloader six million parameters nothing touches.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_masinissa_sentiment import MasinissaSentimentConfig


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
    def __init__(self, config: MasinissaSentimentConfig) -> None:
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
    def __init__(self, config: MasinissaSentimentConfig) -> None:
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
        identical 512x512 int64 lookups would add 25.2 MB to the download — so as a
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
    def __init__(self, config: MasinissaSentimentConfig) -> None:
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
    def __init__(self, config: MasinissaSentimentConfig) -> None:
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


class Encoder(nn.Module):
    """The trunk, under the attribute names the fine-tuned checkpoint was saved with."""

    def __init__(self, config: MasinissaSentimentConfig) -> None:
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Transformer(config)

    def contextualise(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """`input_ids` `[B, T]`, `attention_mask` `[B, T]` True where real. Returns `[B, T, D]`."""
        words, relative = self.embedding(input_ids)
        padding = ~attention_mask[:, None, None, :]
        hidden: Tensor = self.transformer(words.transpose(0, 1), padding, relative)
        return hidden.transpose(0, 1)


def masked_mean(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean over the real positions.

    Not `[CLS]`: this encoder is masked-language-model pretrained with no sentence-level
    objective, so nothing ever trained that position to summarise the sequence.
    """
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class MultiSampleHead(nn.Module):
    """A normalised bottleneck read through several dropout masks at once.

    The masks are the training-time objective and are identity here; inference takes the
    undropped path, which is the one every number on the card was measured through.
    """

    def __init__(self, config: MasinissaSentimentConfig) -> None:
        super().__init__()
        self.label_smoothing = config.label_smoothing
        self.norm_in = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.bottleneck_size)
        self.activation = nn.GELU(approximate="tanh")
        self.norm_mid = nn.LayerNorm(config.bottleneck_size, eps=config.layer_norm_eps)
        self.dropouts = nn.ModuleList(nn.Dropout(rate) for rate in config.dropouts)
        self.classifier = nn.Linear(config.bottleneck_size, len(config.labels))

    def features(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        pooled = masked_mean(hidden, attention_mask)
        bottleneck: Tensor = self.norm_mid(self.activation(self.dense(self.norm_in(pooled))))
        return bottleneck

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        feature = self.features(hidden, attention_mask)
        if not self.training:
            logits = self.classifier(feature)
            loss = None if labels is None else functional.cross_entropy(logits, labels)
            return logits, loss

        sampled = [self.classifier(dropout(feature)) for dropout in self.dropouts]
        logits = torch.stack(sampled, dim=0).mean(dim=0)
        loss = None
        if labels is not None:
            loss = torch.stack(
                [
                    functional.cross_entropy(sample, labels, label_smoothing=self.label_smoothing)
                    for sample in sampled
                ]
            ).mean()
        return logits, loss


class MasinissaSentimentPreTrainedModel(PreTrainedModel):
    config_class = MasinissaSentimentConfig
    base_model_prefix = "masinissa_sentiment"
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


def _real_positions(input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
    """`[B, T]`, True where the position is a real token.

    `~` on an int tensor is a bitwise complement, not a logical one, so the cast is
    load-bearing: `attention_mask` arrives from a tokenizer as 1/0 integers, and their
    complements are -2 and -1, both truthy.
    """
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    return attention_mask.bool()


class MasinissaForSequenceClassification(MasinissaSentimentPreTrainedModel):
    """Kabyle sentiment: negative, neutral, positive, with a calibrated prediction set."""

    _keys_to_ignore_on_load_unexpected = [r"^encoder\.classifier\.", r".*position_indices$"]
    """The pretrained masked-LM head and the derived bucket tables. Both are in the
    training checkpoint and neither is in the release, and a warning naming a dozen
    tensors reads to a downloader as a broken download."""

    def __init__(self, config: MasinissaSentimentConfig) -> None:
        super().__init__(config)
        self.encoder = Encoder(config)
        self.head = MultiSampleHead(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.embedding.word_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.encoder.embedding.word_embedding = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[Tensor, ...]:
        mask = _real_positions(input_ids, attention_mask)
        hidden = self.encoder.contextualise(input_ids, mask)
        logits, loss = self.head(hidden, mask, labels=labels)
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def probabilities(self, logits: Tensor) -> Tensor:
        """Temperature-scaled probabilities, which is what the card's confidences are.

        Monotone, so it moves no argmax: what it moves is the number attached to it. The
        raw softmax of a fine-tune this size is overconfident, and on dialectal input
        that overconfidence is where it is least deserved.
        """
        temperature = max(float(self.config.temperature), 1e-3)
        return torch.softmax(logits / temperature, dim=-1)

    @torch.inference_mode()
    def classify(
        self,
        texts: str | list[str],
        tokenizer: Any,
        batch_size: int = 64,
        max_length: int | None = None,
    ) -> list[dict[str, Any]]:
        """Classify text. One dict per input: `label`, `confidence`, `prediction_set`.

        `prediction_set` is the split-conformal set `{k : p_k >= 1 - q_hat}` when the
        repository carries a calibrated `q_hat`, and the argmax alone when it does not —
        never a singleton dressed up as a guarantee. It widens where the text is genuinely
        ambiguous, which mixed sentiment and irony are.
        """
        wanted = [texts] if isinstance(texts, str) else list(texts)
        if not wanted:
            return []
        limit = max_length or self.config.max_position_embeddings
        device = next(self.parameters()).device
        labels = list(self.config.labels)
        threshold = None if self.config.q_hat is None else 1.0 - float(self.config.q_hat)

        results: list[dict[str, Any]] = []
        for start in range(0, len(wanted), batch_size):
            batch = tokenizer(
                wanted[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=limit,
                return_tensors="pt",
            ).to(device)
            scored = self.probabilities(self(**batch).logits)
            for row in scored.tolist():
                top = max(range(len(row)), key=row.__getitem__)
                members = (
                    tuple(name for name, p in zip(labels, row, strict=True) if p >= threshold)
                    if threshold is not None
                    else (labels[top],)
                )
                results.append(
                    {
                        "label": labels[top],
                        "confidence": row[top],
                        "prediction_set": members or (labels[top],),
                        "probabilities": dict(zip(labels, row, strict=True)),
                    }
                )
        return results


__all__ = [
    "MasinissaForSequenceClassification",
    "MasinissaSentimentConfig",
    "MasinissaSentimentPreTrainedModel",
]
