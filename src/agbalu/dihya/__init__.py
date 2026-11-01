"""Dihya-22M Pan-Amazigh language and script identification package.

Exports ModelConfig, TrainConfig, Classifier, and Taxonomy mappings.
"""

from __future__ import annotations

from agbalu.dihya.config import ModelConfig, TrainConfig
from agbalu.dihya.model import Classifier, Output
from agbalu.dihya.taxonomy import (
    AMAZIGH_VARIETIES,
    COMPOSITE_CLASSES,
    SCRIPTS,
    detect_dominant_script,
    is_amazigh_label,
)

__all__ = [
    "AMAZIGH_VARIETIES",
    "COMPOSITE_CLASSES",
    "SCRIPTS",
    "Classifier",
    "ModelConfig",
    "Output",
    "TrainConfig",
    "detect_dominant_script",
    "is_amazigh_label",
]
