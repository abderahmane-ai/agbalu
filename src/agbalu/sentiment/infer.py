"""Sentiment over Kabyle text, with the two calibrations the model was published with.

The plain forward pass returns raw logits and applies neither, so a caller reading
`logits.argmax()` is running a different classifier from the one the card describes. This
is the path that runs the published one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor

from agbalu.bench.sentiment import LABEL_NAMES, MAX_PIECES, Tokenizer
from agbalu.model.config import PRESETS, Preset
from agbalu.model.modeling import Encoder
from agbalu.normalise import Normaliser
from agbalu.sentiment.calibrate import ConformalPredictor, TemperatureScaler
from agbalu.sentiment.config import SentimentConfig
from agbalu.sentiment.model import SequenceClassifier
from agbalu.tokenizer.evaluate import load as load_tokenizer
from agbalu.tokenizer.spec import CLS_ID, PAD_ID, SEP_ID

if TYPE_CHECKING:
    from agbalu.model.config import ModelConfig

DEFAULT_BATCH: Final = 64


class InferenceError(Exception):
    """A directory that does not hold a usable classifier."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """One sentence's calibrated verdict.

    `prediction_set` widens where the text is genuinely ambiguous. A forced singleton
    there is a guess with a confidence attached to it.
    """

    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float]
    prediction_set: tuple[str, ...]
    normalised_text: str

    @property
    def is_ambiguous(self) -> bool:
        return len(self.prediction_set) > 1

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 6),
            "probabilities": {name: round(value, 6) for name, value in self.probabilities.items()},
            "prediction_set": list(self.prediction_set),
            "is_ambiguous": self.is_ambiguous,
            "normalised_text": self.normalised_text,
        }


def _head_config(published: dict[str, object]) -> SentimentConfig:
    """The head's shapes as the published `config.json` records them.

    Read field by field rather than with `**`: a published config carries `auto_map`,
    `architectures`, `model_type` and `transformers_version`, and a frozen dataclass
    rejects every one of them by name.
    """
    missing = [
        key
        for key in ("hidden_size", "bottleneck_size", "dropouts", "layer_norm_eps", "labels")
        if key not in published
    ]
    if missing:
        message = f"the config is missing {missing}, so the head cannot be rebuilt"
        raise InferenceError(message)
    labels = published["labels"]
    dropouts = published["dropouts"]
    if not isinstance(labels, list) or not isinstance(dropouts, list):
        message = "`labels` and `dropouts` must both be lists in the published config"
        raise InferenceError(message)
    return SentimentConfig(
        hidden_size=int(str(published["hidden_size"])),
        bottleneck_size=int(str(published["bottleneck_size"])),
        num_classes=len(labels),
        dropouts=tuple(float(rate) for rate in dropouts),
        layer_norm_eps=float(str(published["layer_norm_eps"])),
        label_smoothing=float(str(published.get("label_smoothing", 0.05))),
    )


def _encoder_config(published: dict[str, object], preset: Preset) -> ModelConfig:
    """The encoder the weights were trained under, checked against the config on disk.

    A preset that disagrees with the published shapes loads with a shape error naming a
    tensor rather than the mismatch, so the mismatch is named here.
    """
    model = PRESETS[preset]
    disagreements = {
        key: (published[key], getattr(model, key))
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads")
        if key in published and published[key] != getattr(model, key)
    }
    if disagreements:
        message = f"the {preset!r} preset disagrees with the published config: {disagreements}"
        raise InferenceError(message)
    return model


class SentimentClassifier:
    """The encoder, its head, and the temperature and quantile fitted after training."""

    def __init__(
        self,
        model: SequenceClassifier,
        tokenizer: Tokenizer,
        *,
        scaler: TemperatureScaler,
        conformal: ConformalPredictor,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.scaler = scaler.to(self.device)
        self.conformal = conformal
        self.normaliser = Normaliser()
        self.labels = tuple(LABEL_NAMES[: model.config.num_classes])

    @classmethod
    def load(
        cls,
        directory: str | Path,
        tokenizer_path: str | Path,
        *,
        preset: Preset = "kab",
        device: str | torch.device = "cpu",
    ) -> SentimentClassifier:
        """Rebuild from a staged release directory, or refuse to.

        Every file is required. A missing weight file loaded as a warning would leave a
        randomly initialised classifier answering every question with a confidence.
        """
        root = Path(directory)
        target = torch.device(device) if isinstance(device, str) else device

        config_path = root / "config.json"
        if not config_path.is_file():
            message = f"no config.json in {root}; run `make release REPO=masinissa-sentiment`"
            raise InferenceError(message)
        published = json.loads(config_path.read_text(encoding="utf-8"))

        weights = root / "model.pt"
        if not weights.is_file():
            message = f"no model.pt in {root}; pull the checkpoint off the volume first"
            raise InferenceError(message)
        pieces = Path(tokenizer_path)
        if not pieces.is_file():
            message = f"no SentencePiece model at {pieces}"
            raise InferenceError(message)

        model = SequenceClassifier(
            Encoder(_encoder_config(published, preset)), _head_config(published)
        )
        checkpoint = torch.load(weights, map_location=target, weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)

        return cls(
            model,
            load_tokenizer(pieces),
            scaler=TemperatureScaler(float(published.get("temperature", 1.0))),
            conformal=ConformalPredictor(published.get("q_hat")),
            device=target,
        )

    def _batch(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        rows = [
            [CLS_ID, *self.tokenizer.encode(text, out_type=int)[:MAX_PIECES], SEP_ID]
            for text in texts
        ]
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = True
        return input_ids.to(self.device), mask.to(self.device)

    @torch.inference_mode()
    def predict(
        self, texts: str | Sequence[str], *, batch_size: int = DEFAULT_BATCH
    ) -> list[Prediction]:
        """One prediction per input, in the input's order.

        Text is normalised first, because the encoder's vocabulary was built over
        normalised text: the Greek `ε` that corrupts 2.6% of Kabyle sources is a different
        piece from the Latin `ɛ` it stands in for.
        """
        wanted = [texts] if isinstance(texts, str) else list(texts)
        if not wanted:
            return []

        predictions: list[Prediction] = []
        for start in range(0, len(wanted), batch_size):
            chunk = wanted[start : start + batch_size]
            window = [self.normaliser.normalise(text) for text in chunk]
            input_ids, mask = self._batch(window)
            logits, _ = self.model(input_ids, mask)
            probabilities = self.scaler.probabilities(logits)
            sets = self.conformal.predict_set(probabilities)
            for text, row, members in zip(window, probabilities.tolist(), sets, strict=True):
                best = max(range(len(row)), key=row.__getitem__)
                predictions.append(
                    Prediction(
                        label=self.labels[best],
                        label_id=best,
                        confidence=row[best],
                        probabilities=dict(zip(self.labels, row, strict=True)),
                        prediction_set=tuple(self.labels[index] for index in members),
                        normalised_text=text,
                    )
                )
        return predictions
