"""Belaid-31M: punctuation and casing restoration over Kabyle ASR output.

The encoder body is Masinissa's and is written out again rather than imported: a published
repository is standalone, so cross-repository imports are not available to a downloader.
`tests/unit/test_hub_belaid.py` is what keeps the two honest — it asserts this module
reproduces the training `Restorer` tensor for tensor on the same weights.

Module attribute names are the checkpoint's `state_dict` keys, so the tree is
`encoder.{embedding,transformer,classifier}` beside `punctuation` and `case`. The masked-token
classifier is carried because the trained weights contain it and its decoder is tied to the
input embedding; it takes no part in restoration.

`restore` is the whole job — text in, punctuated and capitalised text out. Predicting two
label sequences is not usable on its own: the labels are per *word* and the model sees
subwords, and the alignment is what makes them mean anything.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput

from .configuration_belaid import BelaidConfig


class TokenizerLike(Protocol):
    """What `restore` needs of a tokenizer: the two sentinels and a per-word encode."""

    @property
    def cls_token_id(self) -> int | None: ...

    @property
    def sep_token_id(self) -> int | None: ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


MARKS = {"NONE": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "COLON": ":"}
WORD_CHARS = "-"


def log_bucket_position(relative_position: Tensor, bucket_size: int, max_position: int) -> Tensor:
    """Relative offsets to bucket indices: linear near the diagonal, logarithmic beyond."""
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


def split_words(text: str) -> list[str]:
    """The one definition of what a word is, and the inverse of how the labels were made.

    Composed first, because a combining mark is not alphanumeric: decomposed `ḍ` would lose
    its dot and stop matching its composed twin. Format characters — zero-width space, BOM,
    soft hyphen, directional marks — are removed rather than split on; all are invisible and
    none is a word boundary.
    """
    composed = unicodedata.normalize("NFC", text)
    visible = "".join(char for char in composed if unicodedata.category(char) != "Cf")
    spaced = "".join(char if (char.isalnum() or char in WORD_CHARS) else " " for char in visible)
    return [part for part in spaced.split() if any(char.isalnum() for char in part)]


class GeGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * functional.gelu(gate, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, config: BelaidConfig) -> None:
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
    def __init__(self, config: BelaidConfig) -> None:
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
        from the checkpoint, and this table is derived rather than stored, so as a
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
    def __init__(self, config: BelaidConfig) -> None:
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
    def __init__(self, config: BelaidConfig) -> None:
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
    """Carried because the trained weights contain it. Restoration never calls it."""

    def __init__(self, config: BelaidConfig) -> None:
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


class Encoder(nn.Module):
    """Masinissa, under the attribute name the restorer's checkpoint stored it as."""

    def __init__(self, config: BelaidConfig) -> None:
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Transformer(config)
        self.classifier = MaskClassifier(config)

    def contextualise(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        """`[B, T]` in, `[B, T, D]` out. The mask is True where the position is real."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        padding = ~attention_mask.bool()[:, None, None, :]
        words, relative = self.embedding(input_ids)
        hidden: Tensor = self.transformer(words.transpose(0, 1), padding, relative)
        contextual: Tensor = hidden.transpose(0, 1)
        return contextual


class Head(nn.Module):
    def __init__(self, hidden_size: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, classes)

    def forward(self, hidden: Tensor) -> Tensor:
        projected: Tensor = self.out(self.dropout(self.activation(self.dense(self.norm(hidden)))))
        return projected


@dataclass
class BelaidOutput(ModelOutput):
    """Two independent label sets, so two logit tensors rather than one `logits`.

    Casing is lexical and punctuation is syntactic; a single softmax over their product
    would spend capacity on combinations that never occur and make the two error rates
    impossible to read apart.
    """

    punctuation_logits: Tensor | None = None
    case_logits: Tensor | None = None


class BelaidPreTrainedModel(PreTrainedModel):
    config_class = BelaidConfig
    base_model_prefix = "belaid"
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


class BelaidForTokenClassification(BelaidPreTrainedModel):
    """The encoder plus a punctuation head and a casing head, both over every position."""

    _tied_weights_keys = {
        "encoder.classifier.decoder.weight": "encoder.embedding.word_embedding.weight"
    }

    def __init__(self, config: BelaidConfig) -> None:
        super().__init__(config)
        self.encoder = Encoder(config)
        self.punctuation = Head(
            config.hidden_size, len(config.punctuation_labels), config.classifier_dropout
        )
        self.case = Head(config.hidden_size, len(config.case_labels), config.classifier_dropout)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.embedding.word_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.encoder.embedding.word_embedding = value

    def get_output_embeddings(self) -> nn.Module:
        return self.encoder.classifier.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.encoder.classifier.decoder = new_embeddings

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> BelaidOutput:
        hidden = self.encoder.contextualise(input_ids, attention_mask)
        return BelaidOutput(
            punctuation_logits=self.punctuation(hidden), case_logits=self.case(hidden)
        )

    def _encode(
        self, words: list[str], tokenizer: TokenizerLike, max_length: int
    ) -> tuple[list[int], list[int]]:
        """`[CLS] … [SEP]` and each word's first subword position.

        A word that would overflow is dropped rather than truncated mid-word, so a returned
        position always indexes the start of a whole word.
        """
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            message = "the tokenizer defines no [CLS]/[SEP]; the encoder was trained with both"
            raise ValueError(message)
        ids: list[int] = [cls_id]
        first: list[int] = []
        budget = max_length - 2

        for word in words:
            pieces = tokenizer.encode(word, add_special_tokens=False)
            if not pieces or len(ids) - 1 + len(pieces) > budget:
                break
            first.append(len(ids))
            ids.extend(pieces)

        ids.append(sep_id)
        return ids, first

    @torch.inference_mode()
    def restore(
        self,
        texts: str | list[str],
        tokenizer: TokenizerLike,
        max_length: int = 128,
    ) -> list[str]:
        """Punctuated, capitalised text from unpunctuated ASR output. One result per input.

        Spelling is never edited — the words come back as they went in — so this cannot
        introduce a transcription error into text the ASR model got right. Words beyond
        `max_length` subwords are dropped, so a long input comes back as a covered prefix.
        """
        wanted = [texts] if isinstance(texts, str) else texts
        device = next(self.parameters()).device
        results: list[str] = []

        for text in wanted:
            words = [word.lower() for word in split_words(text)]
            ids, first = self._encode(words, tokenizer, max_length)
            if not first:
                results.append("")
                continue

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            output = self(input_ids, attention)
            index = torch.tensor(first, dtype=torch.long, device=device)
            marks = output.punctuation_logits[0].argmax(-1).index_select(0, index).tolist()
            cases = output.case_logits[0].argmax(-1).index_select(0, index).tolist()

            rebuilt = [
                self._cased(word, self.config.case_labels[case])
                + MARKS[self.config.punctuation_labels[mark]]
                for word, mark, case in zip(words[: len(first)], marks, cases, strict=True)
            ]
            results.append(" ".join(rebuilt))
        return results

    @staticmethod
    def _cased(word: str, label: str) -> str:
        return word[:1].upper() + word[1:] if label == "UPPER_INIT" else word


__all__ = [
    "BelaidConfig",
    "BelaidForTokenClassification",
    "BelaidOutput",
    "BelaidPreTrainedModel",
    "split_words",
]
