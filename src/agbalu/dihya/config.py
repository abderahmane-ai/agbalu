"""Architecture and training configuration for Dihya-22M Pan-Amazigh LID.

Configures a 21.8M parameter byte-level Conv-Transformer with decoupled multi-task
heads, LDAM margins, and ConLID contrastive objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.dihya.taxonomy import COMPOSITE_CLASSES, SCRIPTS

DEFAULT_RUNS_DIR: Final = Path("artifacts/runs")
DEFAULT_DATA_DIR: Final = Path("data/processed/dihya")

PAD_ID: Final = 0
UNK_ID: Final = 1
BYTE_OFFSET: Final = 2
VOCAB_SIZE: Final = 256 + BYTE_OFFSET


class ConfigError(Exception):
    """Raised when configuration parameters violate mathematical invariants."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Dimensions and hyperparameters for the Dihya-22M neural architecture."""

    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 384
    conv_kernels: tuple[int, ...] = (3, 5, 7, 11)
    conv_dim: int = 160
    num_layers: int = 10
    num_heads: int = 6
    intermediate_dim: int = 1344
    projection_dim: int = 128
    max_length: int = 512
    dropout: float = 0.1
    scale: float = 28.0
    margin_scale: float = 0.5
    classes: tuple[str, ...] = field(default=COMPOSITE_CLASSES)
    scripts: tuple[str, ...] = field(default=SCRIPTS)

    def __post_init__(self) -> None:
        dimensions = (
            self.embed_dim,
            self.conv_dim,
            self.num_heads,
            self.intermediate_dim,
            self.projection_dim,
            self.max_length,
        )
        if min(dimensions) < 1 or self.vocab_size != VOCAB_SIZE:
            msg = f"Invalid dimensions {dimensions!r} or vocab_size {self.vocab_size}"
            raise ConfigError(msg)
        if self.embed_dim % self.num_heads != 0:
            msg = f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}"
            raise ConfigError(msg)
        if any(kernel % 2 == 0 for kernel in self.conv_kernels):
            msg = f"Conv kernels must be odd to preserve sequence length: {self.conv_kernels}"
            raise ConfigError(msg)
        if self.num_layers < 1:
            msg = f"num_layers must be at least 1, got {self.num_layers}"
            raise ConfigError(msg)
        if not self.classes:
            msg = "Target classes list cannot be empty"
            raise ConfigError(msg)
        if self.head_dim % 2 or not self.conv_kernels or min(self.conv_kernels) < 1:
            msg = "RoPE requires even head_dim and convolution kernels must be positive"
            raise ConfigError(msg)
        if len(set(self.classes)) != len(self.classes) or set(self.classes) - set(
            COMPOSITE_CLASSES
        ):
            msg = f"Invalid or duplicate classes {self.classes!r}"
            raise ConfigError(msg)
        if self.scripts != SCRIPTS or not 0 <= self.dropout < 1 or self.scale <= 0:
            msg = "Invalid scripts, dropout or cosine scale"
            raise ConfigError(msg)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_scripts(self) -> int:
        return len(self.scripts)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimization schedule, batch sampling, and multi-objective loss weights."""

    epochs: int = 8
    groups_per_batch: int = 16
    samples_per_group: int = 12
    learning_rate: float = 4e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_batch_size: int = 256
    contrastive_weight: float = 0.4
    script_weight: float = 0.2
    filter_weight: float = 0.2
    contrastive_temperature: float = 0.07
    memory_bank_size: int = 4096
    group_dro_step: float = 0.01
    deferred_share: float = 0.5
    label_smoothing: float = 0.05
    min_group_support: int = 20
    evaluations_per_epoch: int = 4
    warmup_share: float = 0.02
    seed: int = 20261101

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.groups_per_batch,
            self.samples_per_group,
            self.eval_batch_size,
            self.min_group_support,
            self.evaluations_per_epoch,
        )
        if min(positive) < 1:
            msg = f"Training counts must be positive: {positive!r}"
            raise ConfigError(msg)
        if not 0 <= self.deferred_share <= 1 or not 0 <= self.warmup_share < 1:
            msg = "Invalid deferred_share or warmup_share"
            raise ConfigError(msg)
        if not 0 <= self.label_smoothing < 1:
            msg = f"Invalid label_smoothing {self.label_smoothing}"
            raise ConfigError(msg)
        if not 0 < self.min_learning_rate <= self.learning_rate or self.max_grad_norm <= 0:
            msg = "Invalid learning rates or max_grad_norm"
            raise ConfigError(msg)
        weights = (
            self.contrastive_weight,
            self.script_weight,
            self.filter_weight,
            self.group_dro_step,
            self.weight_decay,
        )
        if min(weights) < 0 or self.contrastive_temperature <= 0 or self.memory_bank_size < 0:
            msg = "Loss weights, bank size and temperature are invalid"
            raise ConfigError(msg)

    @property
    def batch_size(self) -> int:
        return self.groups_per_batch * self.samples_per_group
