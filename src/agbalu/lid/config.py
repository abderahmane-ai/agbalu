"""Architecture and training configuration for the Berber language identifier.

Published as `agbalu/Dihya-5M`. The release name is not a code identifier, so
the classes here are named for what they configure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.bench.lid_sources import CLASSES

DEFAULT_CHECKPOINT_DIR: Final = Path("artifacts/runs/dihya")
DEFAULT_DATA_DIR: Final = Path("data/processed/bench/lid-dataset")
DEFAULT_EVAL_PATH: Final = Path("data/processed/bench/lid-eval.jsonl")

PAD_ID: Final = 0
UNK_ID: Final = 1
BYTE_OFFSET: Final = 2
VOCAB_SIZE: Final = 256 + BYTE_OFFSET

MIN_LAYERS: Final = 1


class ConfigError(Exception):
    """A configuration that cannot build a model."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shapes of the byte-level classifier. The defaults configure the 5M release.

    Byte-level rather than character-level because the orthographies differ in
    exactly the characters that separate the languages — `ɛ ɣ ḥ ḍ ṣ ṭ ẓ ṛ ṯ` are two
    or three UTF-8 bytes each, and a byte model sees the distinction as a sequence
    rather than as one token it may never have seen in a given language.
    """

    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 256
    conv_kernels: tuple[int, ...] = (3, 5, 7)
    conv_dim: int = 128
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: int = 704
    projection_dim: int = 128
    max_length: int = 256
    dropout: float = 0.1
    scale: float = 24.0
    """Inverse temperature on the cosine logits. Bounded logits are why this model
    reports max-probability and not a free-energy score: `-logsumexp(s * cos)` over a
    normalised head is monotone in the top cosine, so calling it an out-of-distribution
    energy would be maximum softmax probability under another name."""
    margin_scale: float = 0.5
    """`C` in the LDAM margin `m_k = C / n_k^(1/4)`. Zero disables the margins."""
    classes: tuple[str, ...] = field(default=CLASSES)

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads:
            msg = f"embed_dim {self.embed_dim} is not divisible by num_heads {self.num_heads}"
            raise ConfigError(msg)
        if self.num_layers < MIN_LAYERS:
            msg = f"num_layers must be at least {MIN_LAYERS}, got {self.num_layers}"
            raise ConfigError(msg)
        if not self.classes:
            msg = "classes is empty; there is nothing to classify"
            raise ConfigError(msg)
        if any(kernel % 2 == 0 for kernel in self.conv_kernels):
            msg = f"conv kernels must be odd to preserve length, got {self.conv_kernels}"
            raise ConfigError(msg)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training schedule, sampling and loss weights.

    `groups_per_batch` and `samples_per_group` set the batch: the sampler draws whole
    `(language, domain)` cells, not classes. Balancing classes over a corpus where
    each class arrives from one source balances the source and calls it the language.
    """

    epochs: int = 8
    groups_per_batch: int = 12
    samples_per_group: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_batch_size: int = 256
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.05
    memory_bank_size: int = 2048
    group_dro_step: float = 0.01
    """Multiplicative weight step on the worst `(language, domain)` cell. Zero trains
    on the batch mean, which is what let Tamazight-on-the-web sit at 23.2% recall
    while the average looked healthy."""
    deferred_share: float = 0.5
    """Share of the schedule trained on plain cross-entropy before the LDAM margins and
    the group-DRO weights switch on.

    Deferred re-weighting, from the LDAM paper, and measured here: margins from step 0
    ran 120 steps to a loss of 4.01 where the same run without them reached 0.76. The
    margins are worth having once the features exist and are an obstacle before that."""
    label_smoothing: float = 0.05
    logit_adjustment: float = 0.5
    """`tau` in the inference-time shift `logit_k - tau * log(prior_k)`. Zero reports
    the training prior."""
    phrase_augment_prob: float = 0.35
    case_augment_prob: float = 0.10
    seed: int = 20260823

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("groups_per_batch", self.groups_per_batch),
            ("samples_per_group", self.samples_per_group),
            ("eval_batch_size", self.eval_batch_size),
        ):
            if value < 1:
                msg = f"{name} must be at least 1, got {value}"
                raise ConfigError(msg)

    @property
    def batch_size(self) -> int:
        return self.groups_per_batch * self.samples_per_group
