"""The `config.json` a published restorer needs, written from the training constants.

`checkpoint.save` writes `model`, `optimizer`, `state` and `rng` — resumable state, and no
architecture. Published without a config the weights would have nothing to construct, which
is the defect `Fadhma-300M` shipped with once. Every field here is read from the preset the
trainer built the encoder with and the label tuples the heads are sized by, so the two
cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from agbalu.model.config import PRESETS, Preset
from agbalu.punctuation.labels import CASE, PUNCTUATION

PRESET: Final[Preset] = "kab"
"""The preset `punctuation.model.build` constructs the encoder with."""

CLASSIFIER_DROPOUT: Final = 0.1
"""`Restorer`'s default, and what both heads were trained under."""


def config(preset: Preset = PRESET) -> dict[str, object]:
    """The published architecture record. Field names are `BelaidConfig`'s."""
    model = PRESETS[preset]
    return {
        "model_type": "belaid",
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "intermediate_size": model.intermediate_size,
        "num_attention_heads": model.num_attention_heads,
        "num_hidden_layers": model.num_hidden_layers,
        "max_position_embeddings": model.max_position_embeddings,
        "position_bucket_size": model.position_bucket_size,
        "hidden_dropout_prob": model.hidden_dropout_prob,
        "attention_probs_dropout_prob": model.attention_probs_dropout_prob,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "layer_norm_eps": model.layer_norm_eps,
        "punctuation_labels": list(PUNCTUATION),
        "case_labels": list(CASE),
        "tie_word_embeddings": True,
    }


def write_config(out: Path, preset: Preset = PRESET) -> Path:
    """Write `config.json` into `out`, which `tools.export_checkpoint` then stages."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config(preset), indent=2) + "\n", encoding="utf-8")
    return out
