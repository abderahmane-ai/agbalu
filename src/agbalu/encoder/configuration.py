"""Validate encoder geometry and training settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPECIAL_TOKENS = 5
VOCABULARY_SIZE = 16000
MATRIX_AXES = 2
SMOKE_UPDATES = 20


class Geometry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    vocab_size: int = Field(default=16_000, ge=6)
    hidden_size: int = Field(default=512, ge=8)
    intermediate_size: int = Field(default=1408, ge=8)
    layers: int = Field(default=20, ge=1)
    heads: int = Field(default=8, ge=1)
    block_layers: int = Field(default=4, ge=1)
    buckets: int = Field(default=32, ge=4)
    context_tokens: int = Field(default=1024, ge=32)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    epsilon: float = Field(default=1e-6, gt=0)
    backend: Literal["reference", "flex"] = "flex"

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if self.hidden_size % self.heads or self.layers % self.block_layers:
            message = f"incompatible hidden/head or layer/block dimensions: {self.model_dump()}"
            raise ValueError(message)
        if self.context_tokens <= self.buckets or self.buckets % 2:
            message = f"invalid context/bucket geometry: {self.model_dump()}"
            raise ValueError(message)
        return self

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.heads

    @property
    def parameters(self) -> int:
        width, depth = self.hidden_size, self.layers
        routers = 2 * depth + 1
        return (
            self.vocab_size * width
            + (2 * self.buckets - 1) * width
            + depth * (4 * width * width + 3 * width * self.intermediate_size)
            + 2 * width * width
            + 2 * routers * width
            + 2 * depth * width
            + 2 * depth * self.head_size
            + width * width
            + 2 * width
            + self.vocab_size
        )


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    geometry: Geometry = Field(default_factory=Geometry)
    seed: int = Field(default=20261030, ge=0)
    total_tokens: int = Field(default=4_260_000_000, ge=1)
    batch_tokens: int = Field(default=131_072, ge=32)
    micro_tokens: int = Field(default=8192, ge=32)
    peak_lr: float = Field(default=6e-4, gt=0)
    warmup_fraction: float = Field(default=0.02, gt=0, lt=0.5)
    final_lr_fraction: float = Field(default=0.1, gt=0, le=1)
    long_context_fraction: float = Field(default=0.8, gt=0, lt=1)
    weight_decay: float = Field(default=0.1, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    workers: int = Field(default=4, ge=0, le=12)
    validation_documents: int = Field(default=512, ge=1)
    checkpoint_updates: int = Field(default=50, ge=1)
    checkpoint_seconds: float = Field(default=300, gt=0)
    patience: int = Field(default=3, ge=1)
    minimum_passes: float = Field(default=4.0, ge=0)
    selection_delta: float = Field(default=0.002, ge=0, le=1)
    regression_tolerance: float = Field(default=0.01, ge=0, le=1)
    compile: bool = True
    require_smoke: bool = True
    task_evaluation: bool = True

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        if self.batch_tokens % self.micro_tokens or self.micro_tokens % 1024:
            message = (
                f"batch_tokens={self.batch_tokens} must divide into micro_tokens="
                f"{self.micro_tokens}, a multiple of 1024"
            )
            raise ValueError(message)
        return self


class Locations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    corpus: Path = Path("data/processed/llm/kab-docs.jsonl")
    tokenizer: Path = Path("artifacts/encoder/measurements/agbalu-tok-base-16k.model")
    data: Path = Path("artifacts/encoder/data")
    treebank: Path = Path("data/raw/git.ud-kabyle-adpt")
    runs: Path = Path("artifacts/runs")
