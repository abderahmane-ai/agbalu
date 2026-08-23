"""Berber language identification — Kabyle against its Latin-script siblings.

Published as `agbalu/Dihya-5M`. Six Berber varieties plus an explicit
`NOT_AMAZIGH` class, trained on `(language, domain)` groups rather than on classes.
"""

from __future__ import annotations

from agbalu.lid.calibrate import Calibration, calibrate
from agbalu.lid.config import ModelConfig, TrainConfig
from agbalu.lid.dataset import Augmentation, Batch, Corpus, Record, encode
from agbalu.lid.evaluate import Report
from agbalu.lid.infer import Identifier, Prediction
from agbalu.lid.model import CharClassifier

__all__ = [
    "Augmentation",
    "Batch",
    "Calibration",
    "CharClassifier",
    "Corpus",
    "Identifier",
    "ModelConfig",
    "Prediction",
    "Record",
    "Report",
    "TrainConfig",
    "calibrate",
    "encode",
]
