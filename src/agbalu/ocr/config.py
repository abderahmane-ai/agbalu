"""Architecture and training configuration for the document OCR model.

Published as `agbalu/Feraoun-36M`. The release name is not a code identifier, so
the classes here are named for what they configure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.ocr.vocabulary import VOCAB_SIZE

MIN_VOCAB_SIZE: Final = 10
"""Four specials plus a usable alphabet. Below this the model cannot spell anything."""


class ConfigError(Exception):
    """A configuration that cannot build a model."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shapes and hyperparameters for the Vision-Encoder-Decoder OCR model.

    The defaults configure the 36M parameter release (384-dim hidden states, 6 decoder
    layers, 6 attention heads, 1536-dim intermediate feedforward, and character vocabulary).
    """

    vocab_size: int = VOCAB_SIZE
    hidden_size: int = 384
    intermediate_size: int = 1_536
    num_decoder_layers: int = 6
    num_decoder_heads: int = 6
    max_length: int = 256
    dropout: float = 0.1
    encoder_backbone: str = "microsoft/trocr-small-stage1"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_decoder_heads != 0:
            msg = (
                f"hidden_size {self.hidden_size} is not divisible by "
                f"num_decoder_heads {self.num_decoder_heads}"
            )
            raise ConfigError(msg)
        if self.vocab_size < MIN_VOCAB_SIZE:
            msg = f"vocab_size must be at least {MIN_VOCAB_SIZE}, got {self.vocab_size}"
            raise ConfigError(msg)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_decoder_heads


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training run parameters and learning rate schedule.

    `eval_batches` caps what the in-training evaluation reads, so every CER, WER and exact
    match a run logs is a sample of `eval_batches * batch_size` synthetic lines — a
    checkpoint-selection signal, not a benchmark. A published number comes from
    `agbalu.ocr.cli evaluate` over a held-out draw with the cap lifted.
    """

    output_dir: Path = Path("artifacts/runs/feraoun-36m-v1")
    resume_checkpoint: Path | None = None
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    epochs: int = 5
    max_steps: int | None = None
    gradient_clip_val: float = 1.0
    eval_every_steps: int = 250
    eval_batches: int = 16
    save_every_steps: int = 500
    log_every_steps: int = 20
    tifinagh_ratio: float = 0.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.eval_batches < 1:
            msg = f"eval_batches must be at least 1, got {self.eval_batches}"
            raise ConfigError(msg)
