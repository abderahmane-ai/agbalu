"""Temperature scaling and split-conformal prediction sets, fitted after training.

Both are post-hoc: they read logits the fine-tune has already produced and change no
weight. They travel in the published `config.json` rather than in a sidecar because they
are part of what the model predicts — a repository shipping the weights alone ships an
overconfident classifier and no prediction set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.optim import LBFGS

MIN_TEMPERATURE: Final = 0.05
MAX_TEMPERATURE: Final = 10.0
"""L-BFGS on a near-separable validation set drives the temperature towards zero, where
the softmax saturates and the gradient vanishes. The bound is what `scale` applies, so it
is also what `value` reports and what is written down."""

DEFAULT_ALPHA: Final = 0.05


class CalibrationError(Exception):
    """A calibration that cannot be fitted from the data it was given."""


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """What the two corrections are worth, measured on a split neither was fitted on."""

    temperature: float
    q_hat: float
    raw_ece: float
    calibrated_ece: float
    coverage: float
    mean_set_size: float
    ambiguity_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "temperature": round(self.temperature, 6),
            "q_hat": round(self.q_hat, 6),
            "raw_ece": round(self.raw_ece, 6),
            "calibrated_ece": round(self.calibrated_ece, 6),
            "coverage": round(self.coverage, 6),
            "mean_set_size": round(self.mean_set_size, 6),
            "ambiguity_rate": round(self.ambiguity_rate, 6),
        }


class TemperatureScaler(nn.Module):
    """One scalar divided into the logits, fitted to minimise validation NLL.

    Monotone, so it moves no argmax and costs no accuracy. What it moves is the
    confidence attached to the argmax.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([float(temperature)]))

    @property
    def value(self) -> float:
        """The temperature `scale` actually applies, which is the clamped one."""
        clamped = self.temperature.detach().clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
        return float(clamped.item())

    def scale(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)

    def probabilities(self, logits: Tensor) -> Tensor:
        return functional.softmax(self.scale(logits), dim=-1)

    def fit(
        self, logits: Tensor, labels: Tensor, *, max_iter: int = 100, lr: float = 0.01
    ) -> float:
        """Fit on held-out logits and return the temperature that was fitted."""
        if logits.numel() == 0:
            message = "temperature cannot be fitted on an empty split"
            raise CalibrationError(message)
        self.to(logits.device)
        targets = logits.detach().float()
        gold = labels.detach().long()
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def objective() -> Tensor:
            optimizer.zero_grad(set_to_none=False)
            loss: Tensor = functional.cross_entropy(self.scale(targets), gold)
            loss.backward()
            return loss

        optimizer.step(objective)
        return self.value


class ConformalPredictor:
    """Split conformal over the softmax score, giving a set rather than a label.

    The non-conformity score is `1 - p_gold`, so the set is `{k : p_k >= 1 - q_hat}`. The
    marginal coverage guarantee holds only if the rows `q_hat` was fitted on are
    exchangeable with the rows it is applied to, which is why it is fitted on a split the
    model was not selected on and reported on a third.
    """

    def __init__(self, q_hat: float | None = None) -> None:
        self.q_hat = q_hat

    @property
    def threshold(self) -> float | None:
        """Minimum probability for a class to enter the set, or `None` when unfitted."""
        return None if self.q_hat is None else 1.0 - self.q_hat

    def fit(self, probabilities: Tensor, labels: Tensor, alpha: float = DEFAULT_ALPHA) -> float:
        """The finite-sample quantile at coverage `1 - alpha`.

        `ceil((n + 1) * (1 - alpha))` and not `n * (1 - alpha)`: the correction is what
        makes the guarantee hold at finite `n`, and at these sizes it is a whole rank.
        """
        if not 0.0 < alpha < 1.0:
            message = f"alpha must be in (0, 1), got {alpha}"
            raise CalibrationError(message)
        count = probabilities.shape[0]
        if count == 0:
            message = "the conformal quantile cannot be fitted on an empty split"
            raise CalibrationError(message)

        gold = probabilities[torch.arange(count, device=probabilities.device), labels]
        scores = (1.0 - gold).detach().flatten().sort().values
        rank = math.ceil((count + 1) * (1.0 - alpha))
        self.q_hat = 1.0 if rank > count else float(scores[rank - 1].item())
        return self.q_hat

    def predict_set(self, probabilities: Tensor) -> list[tuple[int, ...]]:
        """One tuple of class indices per row, never empty.

        Unfitted, the set is the argmax alone — a singleton with no guarantee behind it,
        which is reported as such rather than dressed up as one.
        """
        best = probabilities.argmax(dim=-1)
        threshold = self.threshold
        if threshold is None:
            return [(int(index),) for index in best.tolist()]
        members = probabilities >= threshold
        members[torch.arange(probabilities.shape[0], device=probabilities.device), best] = True
        return [tuple(row.nonzero(as_tuple=True)[0].tolist()) for row in members]


def expected_calibration_error(probabilities: Tensor, labels: Tensor, bins: int = 10) -> float:
    """Confidence minus accuracy, averaged over equal-width confidence bins by occupancy."""
    if probabilities.numel() == 0:
        message = "expected calibration error is undefined on an empty split"
        raise CalibrationError(message)
    confidence, predicted = probabilities.max(dim=-1)
    correct = predicted.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)

    error = torch.zeros((), device=probabilities.device)
    for index in range(bins):
        occupied = confidence.gt(edges[index]) & confidence.le(edges[index + 1])
        share = occupied.float().mean()
        if float(share) > 0.0:
            error = error + share * (confidence[occupied].mean() - correct[occupied].mean()).abs()
    return float(error.item())


def evaluate_calibration(
    logits: Tensor,
    labels: Tensor,
    scaler: TemperatureScaler,
    conformal: ConformalPredictor,
) -> CalibrationMetrics:
    """Score both corrections on a split neither was fitted on."""
    calibrated = scaler.probabilities(logits)
    sets = conformal.predict_set(calibrated)
    sizes = [len(members) for members in sets]
    covered = sum(1 for gold, members in zip(labels.tolist(), sets, strict=True) if gold in members)
    return CalibrationMetrics(
        temperature=scaler.value,
        q_hat=1.0 if conformal.q_hat is None else conformal.q_hat,
        raw_ece=expected_calibration_error(functional.softmax(logits, dim=-1), labels),
        calibrated_ece=expected_calibration_error(calibrated, labels),
        coverage=covered / len(sizes),
        mean_set_size=sum(sizes) / len(sizes),
        ambiguity_rate=sum(1 for size in sizes if size > 1) / len(sizes),
    )
