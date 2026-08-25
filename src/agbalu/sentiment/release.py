"""The `config.json` a published Masinissa-Sentiment needs, written from what was trained.

`save_checkpoint` writes weights and the evaluation report; neither carries an
architecture. Published without a config the weights would have nothing to construct them,
which is the defect `Fadhma-300M` shipped with once. Every shape here is read from the
preset the encoder was built with and from the head config it was trained under, so the
two cannot drift, and the two calibration constants are read from the report rather than
retyped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from agbalu.bench.sentiment import LABEL_NAMES
from agbalu.model.config import PRESETS, Preset
from agbalu.sentiment.config import SentimentConfig

PRESET: Final[Preset] = "kab"
"""The preset the encoder Masinissa-Sentiment fine-tunes was pretrained with."""

MODEL_TYPE: Final = "masinissa_sentiment"
"""Distinct from `masinissa`: the same trunk, but a config carrying head shapes and a
calibration that `MasinissaConfig` has no field for."""

QUANTILE_KEYS: Final = ("q_hat", "conformal_quantile")
"""What the split-conformal quantile is called in a report, in the order it is looked for.

Two spellings because the run that trained the published checkpoint wrote the second, and
the name is a schema detail while the number is a measurement — re-running an A10 to
rename a JSON key is not a trade this makes. A third spelling belongs here, not in a
caller."""


class ReleaseError(Exception):
    """A report that cannot supply what the published config must carry."""


def config(
    temperature: float,
    q_hat: float,
    head: SentimentConfig | None = None,
    preset: Preset = PRESET,
) -> dict[str, object]:
    """The published architecture record. Field names are `MasinissaSentimentConfig`'s."""
    encoder = PRESETS[preset]
    sizes = head or SentimentConfig()
    return {
        "model_type": MODEL_TYPE,
        "vocab_size": encoder.vocab_size,
        "hidden_size": encoder.hidden_size,
        "intermediate_size": encoder.intermediate_size,
        "num_attention_heads": encoder.num_attention_heads,
        "num_hidden_layers": encoder.num_hidden_layers,
        "max_position_embeddings": encoder.max_position_embeddings,
        "position_bucket_size": encoder.position_bucket_size,
        "hidden_dropout_prob": encoder.hidden_dropout_prob,
        "attention_probs_dropout_prob": encoder.attention_probs_dropout_prob,
        "layer_norm_eps": encoder.layer_norm_eps,
        "bottleneck_size": sizes.bottleneck_size,
        "dropouts": list(sizes.dropouts),
        "label_smoothing": sizes.label_smoothing,
        "labels": list(LABEL_NAMES),
        "temperature": temperature,
        "q_hat": q_hat,
        "tie_word_embeddings": False,
    }


def calibration_of(report: dict[str, object]) -> tuple[float, float]:
    """The two constants the published model applies, taken from the evaluation report.

    Read rather than defaulted: a config carrying temperature 1.0 and no quantile is a
    model that silently does neither correction, and it looks exactly like a calibrated
    one from the outside.
    """
    found = report.get("calibration")
    if not isinstance(found, dict):
        message = "the report carries no calibration block, so the model cannot be published"
        raise ReleaseError(message)
    if "temperature" not in found:
        message = "the calibration block records no temperature"
        raise ReleaseError(message)
    quantile = next((found[key] for key in QUANTILE_KEYS if key in found), None)
    if quantile is None:
        message = f"the calibration block records no quantile under any of {QUANTILE_KEYS}"
        raise ReleaseError(message)
    return float(found["temperature"]), float(quantile)


def write_config(directory: Path, head: SentimentConfig | None = None) -> Path:
    """Write `config.json` beside a pulled checkpoint, from its own `evaluation.json`."""
    report_path = directory / "evaluation.json"
    if not report_path.is_file():
        message = f"no evaluation report at {report_path}; pull the checkpoint first"
        raise ReleaseError(message)
    temperature, q_hat = calibration_of(json.loads(report_path.read_text(encoding="utf-8")))
    path = directory / "config.json"
    path.write_text(json.dumps(config(temperature, q_hat, head), indent=2) + "\n", encoding="utf-8")
    return path
