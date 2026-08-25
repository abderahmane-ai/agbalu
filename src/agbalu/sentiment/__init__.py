"""Three-class sentiment for Kabyle — negative, neutral, positive.

Published as `agbalu/Masinissa-Sentiment`. The encoder fine-tuned over several seeds and
weight-averaged, then temperature-scaled and given a split-conformal quantile, so the
model can return a set of labels where the text carries no polarity of its own.

The labels it was fitted and scored against are projected from the English side of a
Tatoeba pair, not annotated, and `docs/cards/masinissa-sentiment.md` leads with what that
makes every number here a number about.
"""

from __future__ import annotations

from agbalu.bench.sentiment import LABEL_NAMES, NUM_CLASSES
from agbalu.sentiment.calibrate import CalibrationMetrics, ConformalPredictor, TemperatureScaler
from agbalu.sentiment.config import SentimentConfig, SentimentConfigError
from agbalu.sentiment.infer import Prediction, SentimentClassifier
from agbalu.sentiment.model import SequenceClassifier
from agbalu.sentiment.release import write_config
from agbalu.sentiment.train import (
    EvaluationReport,
    SweepSummary,
    TrainConfig,
    save_checkpoint,
    train_classifier,
)

__all__ = [
    "LABEL_NAMES",
    "NUM_CLASSES",
    "CalibrationMetrics",
    "ConformalPredictor",
    "EvaluationReport",
    "Prediction",
    "SentimentClassifier",
    "SentimentConfig",
    "SentimentConfigError",
    "SequenceClassifier",
    "SweepSummary",
    "TemperatureScaler",
    "TrainConfig",
    "save_checkpoint",
    "train_classifier",
    "write_config",
]
