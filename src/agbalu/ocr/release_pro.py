"""The `config.json` a published Feraoun-Pro needs, written from the training constants.

`checkpoint.save` writes `model`, `optimizer`, `state` and `rng` — resumable state, and no
architecture. Every field here comes from the module the trainer builds the recogniser
with, so the published shape cannot drift from the trained one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, TypedDict

from agbalu.ocr.model_pro import (
    DEFAULT_D_MODEL,
    DEFAULT_FFN_DIM,
    DEFAULT_MAX_WIDTH_FRAMES,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
)
from agbalu.ocr.vocabulary_pro import VOCAB_SIZE, VOCABULARY

DROPOUT: Final = 0.1
"""What `HybridCTCRecognizer` passes to every encoder layer."""

RELEASE_CONFIG: Final = Path("artifacts/ocr_pro/config.json")


class PublishedConfig(TypedDict):
    """Every field `FeraounProConfig.__init__` declares, plus the `model_type` it sets."""

    model_type: str
    vocabulary: list[str]
    vocab_size: int
    d_model: int
    num_encoder_layers: int
    num_heads: int
    dim_feedforward: int
    max_width_frames: int
    dropout: float


def config() -> PublishedConfig:
    """The published architecture record. Field names are `FeraounProConfig`'s."""
    return {
        "model_type": "feraoun_pro",
        "vocabulary": list(VOCABULARY),
        "vocab_size": VOCAB_SIZE,
        "d_model": DEFAULT_D_MODEL,
        "num_encoder_layers": DEFAULT_NUM_LAYERS,
        "num_heads": DEFAULT_NUM_HEADS,
        "dim_feedforward": DEFAULT_FFN_DIM,
        "max_width_frames": DEFAULT_MAX_WIDTH_FRAMES,
        "dropout": DROPOUT,
    }


def write_config(out: Path = RELEASE_CONFIG) -> Path:
    """Write `config.json` into `out`, which `tools.export_checkpoint` then stages."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
