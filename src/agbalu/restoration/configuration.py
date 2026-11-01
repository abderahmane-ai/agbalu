"""Validate restoration geometry and execution settings."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agbalu.encoder.configuration import Geometry


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    geometry: Geometry = Field(default_factory=Geometry)
    seed: int = Field(default=17, ge=0)
    epochs: int = Field(default=2, ge=1)
    objective: Literal["joint", "balanced"] = "joint"
    validation_checks_per_epoch: int = Field(default=16, ge=1)
    micro_windows: int = Field(default=32, ge=1)
    batch_windows: int = Field(default=32, ge=1)
    workers: int = Field(default=4, ge=0, le=12)
    core_tokens: int = Field(default=640, ge=1)
    left_tokens: int = Field(default=191, ge=1)
    streaming_core_tokens: int = Field(default=256, ge=1)
    lookahead_tokens: int = Field(default=128, ge=1)
    streaming_fraction: float = Field(default=0.2, ge=0, le=1)
    encoder_lr: float = Field(default=1.5e-5, gt=0)
    head_lr: float = Field(default=1e-4, gt=0)
    layer_decay: float = Field(default=1.0, gt=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_fraction: float = Field(default=0.02, gt=0, lt=0.5)
    final_lr_fraction: float = Field(default=0.1, gt=0, le=1)
    gradient_clip: float = Field(default=1.0, gt=0)
    validation_windows: int = Field(default=128, ge=1)
    checkpoint_updates: int = Field(default=100, ge=1)
    checkpoint_seconds: float = Field(default=300.0, gt=0)
    patience: int = Field(default=6, ge=1)
    compile: bool = True
    require_smoke: bool = True
    require_pilot: bool = False
    pilot_updates: int = Field(default=2000, ge=21, le=2500)
    minimum_words: int = Field(default=8, ge=1)
    minimum_template_support: int = Field(default=100, ge=1)
    maximum_templates: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_windows(self) -> Self:
        body = self.geometry.context_tokens - 2
        if self.core_tokens + self.left_tokens >= body:
            msg = "offline core and left context leave no right context"
            raise ValueError(msg)
        if self.streaming_core_tokens + self.lookahead_tokens >= body:
            msg = "streaming core and future context leave no history"
            raise ValueError(msg)
        if self.batch_windows % self.micro_windows:
            msg = "batch_windows must be divisible by micro_windows"
            raise ValueError(msg)
        return self


class Locations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    corpus: Path = Path("data/processed/llm/kab-docs.jsonl")
    tokenizer: Path = Path("artifacts/encoder/measurements/agbalu-tok-base-16k.model")
    data: Path = Path("artifacts/restoration/clean/data")
    source_data: Path = Path("artifacts/restoration/data")
    encoder: Path = Path("artifacts/runs/encoder-pretrain")
    identifier: Path = Path("artifacts/runs/dihya")
    speech: Path = Path("data/processed/speech")
    runs: Path = Path("artifacts/runs")
