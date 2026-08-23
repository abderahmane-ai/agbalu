"""Configuration for the Dihya-5M Berber language identifier."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig

CLASSES = [
    "kab_Latn",
    "shi_Latn",
    "rif_Latn",
    "taq_Latn",
    "tzm_Latn",
    "shy_Latn",
    "NOT_AMAZIGH",
]
"""Label order is the trained head's row order. Reordering it renames every prediction."""


class DihyaConfig(PreTrainedConfig):
    """The classifier's shapes, its label set, and the two inference-time corrections.

    `prior_shift` and `q_hat` are stored here rather than in a sidecar because they are
    part of what the model predicts: without the shift the argmax minimises the training
    prior's error rather than balanced error, and without the quantile there is no
    prediction set. A repository that ships the weights alone ships a different model.
    """

    model_type = "dihya"

    def __init__(
        self,
        vocab_size: int = 258,
        hidden_size: int = 256,
        intermediate_size: int = 704,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 6,
        conv_kernels: list[int] | None = None,
        conv_dim: int = 128,
        max_position_embeddings: int = 256,
        rope_theta: float = 10000.0,
        dropout_prob: float = 0.1,
        logit_scale: float = 24.0,
        rms_norm_eps: float = 1e-6,
        classes: list[str] | None = None,
        prior_shift: list[float] | None = None,
        q_hat: float | None = None,
        target_coverage: float = 0.99,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        byte_offset: int = 2,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.conv_kernels = list(conv_kernels or [3, 5, 7])
        self.conv_dim = conv_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.dropout_prob = dropout_prob
        self.logit_scale = logit_scale
        self.rms_norm_eps = rms_norm_eps
        self.unk_token_id = unk_token_id
        self.byte_offset = byte_offset

        labels = list(classes or CLASSES)
        self.classes = labels
        self.prior_shift = list(prior_shift) if prior_shift is not None else [0.0] * len(labels)
        self.q_hat = q_hat
        self.target_coverage = target_coverage

        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("id2label", dict(enumerate(labels)))
        kwargs.setdefault("label2id", {name: i for i, name in enumerate(labels)})
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


__all__ = ["CLASSES", "DihyaConfig"]
