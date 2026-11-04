"""Validate codec speech preparation and adaptation settings."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RECIPE = Path("resources/codec_tts/recipe.json")
REVISION_LENGTH = 40
SMOKE_PARTS = ("parity", "throughput", "resume", "evaluate", "failure")


def is_run_name(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None


class Geometry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    hidden_size: int = Field(default=2560, ge=1)
    intermediate_size: int = Field(default=9728, ge=1)
    layers: int = Field(default=36, ge=1)
    heads: int = Field(default=32, ge=1)
    kv_heads: int = Field(default=8, ge=1)
    head_dim: int = Field(default=128, ge=2, multiple_of=2)
    text_vocabulary: int = Field(default=151936, ge=1)
    rms_epsilon: float = Field(default=1e-6, gt=0)
    rope_theta: float = Field(default=1_000_000, gt=0)

    @model_validator(mode="after")
    def validate_heads(self) -> Self:
        if self.heads % self.kv_heads:
            message = f"heads={self.heads} must be divisible by kv_heads={self.kv_heads}"
            raise ValueError(message)
        return self

    @property
    def adapter_parameters(self) -> int:
        width, query, key = (
            self.hidden_size,
            self.heads * self.head_dim,
            self.kv_heads * self.head_dim,
        )
        return (
            self.layers
            * 32
            * (2 * (width + query) + 2 * (width + key) + 3 * (width + self.intermediate_size))
        )


class Specification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Recipe(Specification):
    version: Literal[1] = 1
    source: Path = Path("data/raw_v2/taqbaylit.common-voice-scripted-speech-kab-26-huge")
    source_revision: str = "dc3f6c81050794a2653c76c6d2ef6518a7cd87e2"
    source_id: str = "hf.boffire.common-voice-scripted-speech-kab-26-huge"
    license: str = "cc0-1.0"
    output: Path = Path("artifacts/runs/codec-tts-data/prepared")
    base_id: str = "bosonai/higgs-tts-3-4b"
    base_revision: str = "239f63fb7b02b1aa085f98d9efae5e35cc5523e8"
    codec_id: str = "bosonai/higgs-audio-v2-tokenizer"
    codec_revision: str = "815a03f1b57dcb5ae918db415f22cd0c8a6f0f54"
    base_directory: Path = Path("artifacts/runs/codec-tts-base/native")
    codec_configuration: Path = Path("resources/codec_tts/codec.json")
    evaluation_examples: int = Field(default=16, ge=2)
    seed: int = Field(default=20261102, ge=0)
    rate_hz: Literal[24000] = 24000
    min_duration_s: float = Field(default=0.25, gt=0)
    max_duration_s: float = Field(default=120, gt=0)
    max_embedded_bytes: int = Field(default=32_000_000, ge=1)
    validation_fraction: float = Field(default=0.05, gt=0, lt=1)
    test_fraction: float = Field(default=0.05, gt=0, lt=1)
    rank: Literal[32] = 32
    alpha: Literal[64] = 64
    dropout: float = Field(default=0, ge=0, le=0)
    peak_lr: float = Field(default=5e-5, gt=0)
    minimum_lr: float = Field(default=5e-6, gt=0)
    warmup_fraction: float = Field(default=0.02, gt=0, lt=1)
    gradient_clip: float = Field(default=1, gt=0)
    update_examples: int = Field(default=32, ge=1)
    micro_examples: int = Field(default=1, ge=1)
    max_mixed_positions: int = Field(default=8192, ge=1, le=8192)
    workers: int = Field(default=2, ge=0, le=12)
    checkpoint_seconds: float = Field(default=300, gt=0)
    memory_reserve_fraction: float = Field(default=0.1, gt=0, lt=1)
    activation_checkpointing: bool = True

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if self.validation_fraction + self.test_fraction >= 1:
            message = "validation_fraction + test_fraction must be below 1"
            raise ValueError(message)
        if self.min_duration_s >= self.max_duration_s:
            message = "min_duration_s must be below max_duration_s"
            raise ValueError(message)
        if self.minimum_lr > self.peak_lr or self.micro_examples > self.update_examples:
            message = "minimum_lr/peak_lr or micro_examples/update_examples is inconsistent"
            raise ValueError(message)
        source, output = self.source.resolve(), self.output.resolve()
        if output == source or source in output.parents or output in source.parents:
            message = f"source and output overlap: {source}, {output}"
            raise ValueError(message)
        for name, revision in (
            ("source_revision", self.source_revision),
            ("base_revision", self.base_revision),
            ("codec_revision", self.codec_revision),
        ):
            if len(revision) != REVISION_LENGTH or any(
                char not in "0123456789abcdef" for char in revision
            ):
                message = f"{name} must be a full lowercase git SHA, got {revision!r}"
                raise ValueError(message)
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def read_recipe(path: Path = DEFAULT_RECIPE) -> Recipe:
    return Recipe.model_validate_json(path.read_bytes())
