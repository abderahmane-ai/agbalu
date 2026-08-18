"""Configuration for the Belaid-31M punctuation and casing restorer."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig

PUNCTUATION_LABELS = ["NONE", "COMMA", "PERIOD", "QUESTION", "COLON"]
CASE_LABELS = ["LOWER", "UPPER_INIT"]


class BelaidConfig(PreTrainedConfig):
    """The encoder's fields, plus the two label sets the heads are defined over.

    `id2label` is deliberately absent. The convention assumes one head, and filling it from
    either of the two would describe half the model while looking like all of it.
    """

    model_type = "belaid"

    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 384,
        intermediate_size: int = 1280,
        num_attention_heads: int = 6,
        num_hidden_layers: int = 12,
        max_position_embeddings: int = 512,
        position_bucket_size: int = 32,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        punctuation_labels: list[str] | None = None,
        case_labels: list[str] | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.layer_norm_eps = layer_norm_eps
        self.punctuation_labels = punctuation_labels or list(PUNCTUATION_LABELS)
        self.case_labels = case_labels or list(CASE_LABELS)

        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("bos_token_id", bos_token_id)
        kwargs.setdefault("eos_token_id", eos_token_id)
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


__all__ = ["CASE_LABELS", "PUNCTUATION_LABELS", "BelaidConfig"]
