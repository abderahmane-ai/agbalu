"""The classification head's shapes, and the label set that sizes it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from agbalu.bench.sentiment import NUM_CLASSES

MIN_CLASSES: Final = 2
"""A one-class head has no decision to make and `cross_entropy` accepts it in silence."""


class SentimentConfigError(Exception):
    """A head configuration that cannot be built, or that the encoder cannot feed."""


@dataclass(frozen=True, slots=True)
class SentimentConfig:
    """Hyperparameters of the multi-sample dropout head.

    `hidden_size` is the encoder's, not a free choice: `SequenceClassifier` refuses a pair
    that disagrees, because a mismatch otherwise surfaces as a shape error inside a Linear
    with neither side named.
    """

    hidden_size: int = 384
    bottleneck_size: int = 192
    num_classes: int = NUM_CLASSES
    dropouts: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
    layer_norm_eps: float = 1e-5
    label_smoothing: float = 0.05

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            message = f"hidden_size must be positive, got {self.hidden_size}"
            raise SentimentConfigError(message)
        if self.bottleneck_size <= 0:
            message = f"bottleneck_size must be positive, got {self.bottleneck_size}"
            raise SentimentConfigError(message)
        if self.num_classes < MIN_CLASSES:
            message = f"num_classes must be >= {MIN_CLASSES}, got {self.num_classes}"
            raise SentimentConfigError(message)
        if not self.dropouts:
            message = "dropouts cannot be empty"
            raise SentimentConfigError(message)
        for rate in self.dropouts:
            if not 0.0 <= rate < 1.0:
                message = f"dropout rate must be in [0, 1), got {rate}"
                raise SentimentConfigError(message)
        if not 0.0 <= self.label_smoothing < 1.0:
            message = f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            raise SentimentConfigError(message)
