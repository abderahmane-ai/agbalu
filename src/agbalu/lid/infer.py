"""Inference for the Berber language identifier.

`Identifier` satisfies the `agbalu.bench.lid.Identifier` protocol, so the trained model
is scored by the same harness as GlotLID and `lid218e` rather than by a second one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from agbalu.acquire.storage import sha256_file
from agbalu.lid.calibrate import Calibration, adjusted_probabilities
from agbalu.lid.calibrate import load as load_calibration
from agbalu.lid.config import PAD_ID, ModelConfig
from agbalu.lid.dataset import encode
from agbalu.lid.model import CharClassifier
from agbalu.model.checkpoint import BEST

DEFAULT_BATCH: Final = 128

TRUNK_PREFIX: Final = "model."
"""The checkpoint bundles the group-DRO weights and the contrastive memory bank beside
the model so a preempted run resumes with them. Neither belongs in an exported artifact,
and both share a key *prefix* rather than a suffix, so the prefix is what the filter
reads — the suffix filter that exports the other models here shipped Feraoun-Pro's
auxiliary decoder to every downloader."""


class InferenceError(Exception):
    """A checkpoint that cannot be loaded, or an input that cannot be classified."""


def exported_weights(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """The classifier's own tensors, without the training-only submodules beside them."""
    return {
        key.removeprefix(TRUNK_PREFIX): value
        for key, value in state.items()
        if key.startswith(TRUNK_PREFIX)
    }


def read_model_config(path: Path) -> ModelConfig:
    """Read a written config into the fields declared here, never with `**`.

    A stored config carries whatever the writing version wrote, and JSON has no tuples,
    so unpacking it hands `list` to a field typed `tuple` and any unknown key straight
    into the constructor. Named fields make an added key a default rather than a crash,
    and a removed one an error naming the field.
    """
    if not path.is_file():
        msg = f"config not found at {path}; the checkpoint cannot be shaped"
        raise InferenceError(msg)
    stored = json.loads(path.read_text(encoding="utf-8"))["model"]
    default = ModelConfig()
    return ModelConfig(
        vocab_size=int(stored.get("vocab_size", default.vocab_size)),
        embed_dim=int(stored.get("embed_dim", default.embed_dim)),
        conv_kernels=tuple(int(k) for k in stored.get("conv_kernels", default.conv_kernels)),
        conv_dim=int(stored.get("conv_dim", default.conv_dim)),
        num_layers=int(stored.get("num_layers", default.num_layers)),
        num_heads=int(stored.get("num_heads", default.num_heads)),
        intermediate_dim=int(stored.get("intermediate_dim", default.intermediate_dim)),
        projection_dim=int(stored.get("projection_dim", default.projection_dim)),
        max_length=int(stored.get("max_length", default.max_length)),
        dropout=float(stored.get("dropout", default.dropout)),
        scale=float(stored.get("scale", default.scale)),
        margin_scale=float(stored.get("margin_scale", default.margin_scale)),
        classes=tuple(str(name) for name in stored.get("classes", default.classes)),
    )


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classified line."""

    language: str
    confidence: float
    prediction_set: tuple[str, ...]
    probabilities: dict[str, float]

    @property
    def is_amazigh(self) -> bool:
        return self.language != "NOT_AMAZIGH"


class Identifier:
    """A loaded classifier with its prior shift and conformal quantile applied."""

    name: Final = "dihya"

    def __init__(
        self,
        model: CharClassifier,
        calibration: Calibration | None = None,
        device: str = "cpu",
        revision: str = "local",
    ) -> None:
        self.model = model.eval().to(device)
        self.calibration = calibration
        self.device = device
        self._revision = revision
        self.classes = tuple(model.config.classes)
        shift = (
            torch.tensor(calibration.prior_shift, dtype=torch.float)
            if calibration is not None
            else torch.zeros(len(self.classes))
        )
        if shift.numel() != len(self.classes):
            msg = f"calibration covers {shift.numel()} classes, model has {len(self.classes)}"
            raise InferenceError(msg)
        self.prior_shift = shift.to(device)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(self.classes)

    @classmethod
    def load(
        cls,
        checkpoint_dir: Path,
        *,
        device: str = "cpu",
        name: str = BEST,
    ) -> Identifier:
        """Load a training checkpoint and, if it is beside it, the calibration."""
        target = checkpoint_dir / name
        if not target.is_file():
            msg = f"checkpoint not found at {target}"
            raise InferenceError(msg)
        payload = torch.load(target, map_location=device, weights_only=True)
        config = read_model_config(checkpoint_dir / "config.json")
        model = CharClassifier(config)
        model.load_state_dict(exported_weights(payload["model"]))

        calibration_path = checkpoint_dir / "calibration.json"
        calibration = load_calibration(calibration_path) if calibration_path.is_file() else None
        digest, _ = sha256_file(target)
        return cls(model, calibration=calibration, device=device, revision=digest[:12])

    def _batch(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        rows = [encode(text, self.model.config.max_length) or [PAD_ID] for text in texts]
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        pad_mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for position, row in enumerate(rows):
            input_ids[position, : len(row)] = torch.tensor(row, dtype=torch.long)
            pad_mask[position, : len(row)] = True
        return input_ids.to(self.device), pad_mask.to(self.device)

    @torch.inference_mode()
    def probabilities(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH) -> Tensor:
        """Prior-adjusted probabilities, one row per input, in input order."""
        if not texts:
            return torch.zeros((0, len(self.classes)))
        chunks: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            input_ids, pad_mask = self._batch(texts[start : start + batch_size])
            logits = self.model(input_ids, pad_mask=pad_mask).logits
            chunks.append(adjusted_probabilities(logits, self.prior_shift).cpu())
        return torch.cat(chunks)

    def predict(self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH) -> list[Prediction]:
        """Classify a batch, with a conformal prediction set when calibrated."""
        rows = self.probabilities(texts, batch_size=batch_size)
        threshold = 1.0 - self.calibration.q_hat if self.calibration is not None else None
        predictions: list[Prediction] = []
        for row in rows.tolist():
            top = max(range(len(row)), key=lambda index: row[index])
            members = (
                tuple(self.classes[i] for i, p in enumerate(row) if p >= threshold)
                if threshold is not None
                else (self.classes[top],)
            )
            predictions.append(
                Prediction(
                    language=self.classes[top],
                    confidence=row[top],
                    prediction_set=members or (self.classes[top],),
                    probabilities=dict(zip(self.classes, row, strict=True)),
                )
            )
        return predictions

    def identify(self, texts: Sequence[str]) -> list[str]:
        """The `agbalu.bench.lid.Identifier` protocol: one label per input."""
        return [prediction.language for prediction in self.predict(texts)]
