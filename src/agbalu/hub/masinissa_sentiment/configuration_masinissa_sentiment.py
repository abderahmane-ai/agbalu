"""Configuration for Masinissa-Sentiment, Kabyle three-class sentiment classification."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig

LABELS = ["negative", "neutral", "positive"]
"""Label order is the trained head's row order. Reordering it renames every prediction."""


class MasinissaSentimentConfig(PreTrainedConfig):
    """The encoder's shapes, the head's, and the two corrections fitted after training.

    `temperature` and `q_hat` live here rather than in a sidecar because they are part of
    what the model predicts: without the temperature the confidence is the fine-tune's
    overconfident one, and without the quantile there is no prediction set. A repository
    shipping the weights alone ships a different model from the one the card measures.

    A distinct `model_type` from `masinissa`: the same trunk, but `MasinissaConfig` has no
    field for a head or a calibration, and a config read into the wrong class loses both
    in silence.
    """

    model_type = "masinissa_sentiment"

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
        layer_norm_eps: float = 1e-5,
        bottleneck_size: int = 192,
        dropouts: list[float] | None = None,
        label_smoothing: float = 0.05,
        labels: list[str] | None = None,
        temperature: float = 1.0,
        q_hat: float | None = None,
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
        self.layer_norm_eps = layer_norm_eps
        self.bottleneck_size = bottleneck_size
        self.dropouts = list(dropouts or [0.10, 0.15, 0.20, 0.25, 0.30])
        self.label_smoothing = label_smoothing
        self.temperature = float(temperature)
        self.q_hat = None if q_hat is None else float(q_hat)

        names = list(labels or LABELS)
        self.labels = names

        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("bos_token_id", bos_token_id)
        kwargs.setdefault("eos_token_id", eos_token_id)
        kwargs.setdefault("id2label", dict(enumerate(names)))
        kwargs.setdefault("label2id", {name: index for index, name in enumerate(names)})
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


__all__ = ["LABELS", "MasinissaSentimentConfig"]
