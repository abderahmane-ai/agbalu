"""Record calibrated Dihya observations without treating ambiguity as rejection."""

from __future__ import annotations

import json

import torch
from pydantic import Field
from torch import nn
from transformers import AutoModelForSequenceClassification

from agbalu.codec_tts.artifacts import (
    atomic_binary,
    digest_file,
    provenance,
    read_recordings,
    run_lock,
    verify_conservation,
    write_json,
)
from agbalu.codec_tts.configuration import Recipe, Specification
from agbalu.llm.progress import Tracker

IDENTIFIER_ID = "agbalu/Dihya-22M"
IDENTIFIER_REVISION = "84b41e03aad716e6016b35fbf9e6c9903cf78f07"
MAX_IDENTIFIER_BYTES = 512


class Identification(Specification):
    language: str
    confidence: float = Field(ge=0, le=1)
    prediction_set: tuple[str, ...] = Field(min_length=1)
    probabilities: dict[str, float]


def validate_identifications(value: object, expected: int) -> tuple[Identification, ...]:
    if not isinstance(value, list) or len(value) != expected:
        message = f"identifier returned {type(value).__name__} with unexpected support"
        message += f"; expected={expected}"
        raise ValueError(message)
    return tuple(Identification.model_validate(row) for row in value)


@torch.inference_mode()
def observe_languages(recipe: Recipe, device: torch.device) -> dict[str, object]:
    model: nn.Module = AutoModelForSequenceClassification.from_pretrained(
        IDENTIFIER_ID, revision=IDENTIFIER_REVISION, trust_remote_code=True
    )
    model.to(device).train(False)
    identify = getattr(model, "identify", None)
    config = getattr(model, "config", None)
    quantile = getattr(config, "q_hat", None)
    if not callable(identify) or not isinstance(quantile, float) or not 0 < quantile < 1:
        message = f"pinned identifier lacks calibrated identify contract: q_hat={quantile!r}"
        raise ValueError(message)
    controls = validate_identifications(
        identify(
            [
                "Azul fell-awen, amek tettilim ass-a?",
                "Bonjour tout le monde, comment allez-vous?",
            ]
        ),
        2,
    )
    if "kab_Latn" not in controls[0].prediction_set or controls[1].language != "fra_Latn":
        message = f"identifier controls failed: {[item.model_dump() for item in controls]}"
        raise ValueError(message)
    path = recipe.output / "recordings.jsonl"
    records = tuple(read_recordings(path))
    uncertain = excluding_kabyle = observed = 0
    tracker = Tracker("Dihya transcripts", len(records), interval_s=5)
    with run_lock(recipe.output), atomic_binary(recipe.output / "language.jsonl") as stream:
        for start in range(0, len(records), 128):
            batch = records[start : start + 128]
            results = validate_identifications(
                identify([record.text for record in batch]), len(batch)
            )
            for record, result in zip(batch, results, strict=True):
                observed += 1
                uncertain += len(result.prediction_set) != 1
                excluding_kabyle += "kab_Latn" not in result.prediction_set
                observation = {
                    "record_id": record.record_id,
                    **result.model_dump(),
                    "is_truncated": len(record.text.encode()) > MAX_IDENTIFIER_BYTES,
                    "decision": "retain_for_review"
                    if "kab_Latn" not in result.prediction_set
                    else "retain",
                    "model_revision": IDENTIFIER_REVISION,
                }
                stream.write((json.dumps(observation, allow_nan=False) + "\n").encode())
            tracker.advance(len(batch))
    verify_conservation(len(records), observed, 0)
    report = {
        **provenance(recipe.digest),
        "recordings_sha256": digest_file(path),
        "identifier_revision": IDENTIFIER_REVISION,
        "q_hat": quantile,
        "observed_rows": observed,
        "ambiguous_rows": uncertain,
        "excluding_kabyle_rows": excluding_kabyle,
        "dropped_rows": 0,
        "observations_sha256": digest_file(recipe.output / "language.jsonl"),
        "controls": [item.model_dump() for item in controls],
    }
    write_json(recipe.output / "language.stats.json", report)
    return report
