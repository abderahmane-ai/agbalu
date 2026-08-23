"""Conformal prediction sets and the logit adjustment, fitted on held-out rows.

`q_hat` is fitted on one half of the calibration split and its coverage measured on
the other. Measuring coverage on the rows that set the quantile reports the fit, not
the guarantee, and the two differ by exactly the amount that matters.

The score is `1 - p(true class)` on the adjusted probabilities, so a prediction set is
`{k : p_k >= 1 - q_hat}`. There is no energy threshold: the head is a scaled cosine,
`-logsumexp(s * cos)` is monotone in the top cosine, and calling that an
out-of-distribution energy would be maximum softmax probability under another name.
Out-of-distribution input is what the `NOT_AMAZIGH` class is for.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from agbalu.lid.config import TrainConfig
from agbalu.lid.dataset import Corpus, sequential_batches
from agbalu.lid.losses import logit_adjustment
from agbalu.lid.model import CharClassifier

MIN_CALIBRATION_ROWS: Final = 40


class CalibrationError(Exception):
    """Calibration data that cannot support a quantile."""


@dataclass(frozen=True, slots=True)
class Calibration:
    """A coverage target, the quantile that meets it, and what it measured."""

    target_coverage: float
    q_hat: float
    fitted_on: int
    measured_on: int
    empirical_coverage: float
    mean_set_size: float
    classes: tuple[str, ...]
    prior_shift: tuple[float, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> Calibration:
        data = json.loads(payload)
        data["classes"] = tuple(data["classes"])
        data["prior_shift"] = tuple(data["prior_shift"])
        return cls(**data)


def adjusted_probabilities(logits: Tensor, prior_shift: Tensor) -> Tensor:
    """Softmax after subtracting `tau * log(prior)` from every logit."""
    return torch.softmax(logits - prior_shift.to(logits), dim=-1)


@torch.inference_mode()
def collect_probabilities(
    model: CharClassifier, corpus: Corpus, prior_shift: Tensor, *, batch_size: int, device: str
) -> tuple[Tensor, Tensor]:
    """Adjusted probabilities and gold labels for a whole split, in split order."""
    model.eval()
    chunks: list[Tensor] = []
    labels: list[Tensor] = []
    for indices in sequential_batches(corpus, batch_size):
        batch = corpus.collate(indices).to(torch.device(device))
        logits = model(batch.input_ids, pad_mask=batch.pad_mask).logits
        chunks.append(adjusted_probabilities(logits, prior_shift).cpu())
        labels.append(batch.labels.cpu())
    return torch.cat(chunks), torch.cat(labels)


def conformal_quantile(scores: Tensor, target_coverage: float) -> float:
    """The finite-sample quantile `ceil((n + 1) * coverage) / n` of the scores."""
    count = scores.numel()
    if count == 0:
        msg = "no calibration scores"
        raise CalibrationError(msg)
    rank = min(int(torch.ceil(torch.tensor((count + 1) * target_coverage)).item()), count)
    return float(scores.sort().values[rank - 1])


def calibrate(
    model: CharClassifier,
    corpus: Corpus,
    *,
    settings: TrainConfig,
    target_coverage: float = 0.99,
    device: str = "cpu",
) -> Calibration:
    """Fit `q_hat` on the first half of `corpus` and measure coverage on the second."""
    if len(corpus) < MIN_CALIBRATION_ROWS:
        msg = f"calibration needs at least {MIN_CALIBRATION_ROWS} rows, got {len(corpus)}"
        raise CalibrationError(msg)
    if not 0.0 < target_coverage < 1.0:
        msg = f"target_coverage must be in (0, 1), got {target_coverage}"
        raise CalibrationError(msg)

    shift = logit_adjustment(corpus.class_counts, settings.logit_adjustment)
    probabilities, labels = collect_probabilities(
        model, corpus, shift, batch_size=settings.eval_batch_size, device=device
    )
    permutation = torch.randperm(probabilities.size(0), generator=_generator(settings.seed))
    fit, measure = permutation.chunk(2)

    true_probability = probabilities[fit, labels[fit]]
    q_hat = conformal_quantile(1.0 - true_probability, target_coverage)

    held = probabilities[measure]
    sets = held >= (1.0 - q_hat)
    covered = sets[torch.arange(measure.numel()), labels[measure]]
    return Calibration(
        target_coverage=target_coverage,
        q_hat=q_hat,
        fitted_on=int(fit.numel()),
        measured_on=int(measure.numel()),
        empirical_coverage=float(covered.float().mean()),
        mean_set_size=float(sets.sum(dim=1).float().mean()),
        classes=tuple(corpus.classes),
        prior_shift=tuple(shift.tolist()),
    )


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def save(calibration: Calibration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibration.to_json(), encoding="utf-8")


def load(path: Path) -> Calibration:
    if not path.is_file():
        msg = f"calibration not found at {path}"
        raise CalibrationError(msg)
    return Calibration.from_json(path.read_text(encoding="utf-8"))


__all__: Final[Sequence[str]] = [
    "Calibration",
    "CalibrationError",
    "adjusted_probabilities",
    "calibrate",
    "conformal_quantile",
    "load",
    "save",
]
