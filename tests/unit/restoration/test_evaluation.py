"""Positive-control restoration metrics with exact and shifted labels."""

import pytest
import torch

from agbalu.restoration.dataset import Example, collate
from agbalu.restoration.evaluation import Metrics, confusion, macro, report
from agbalu.restoration.model import Restorer


def test_perfect_and_shifted_predictions_have_different_scores() -> None:
    gold = torch.tensor([0, 1, 2, 1, 0, -100])
    perfect = confusion(gold, gold.clamp(min=0), 3)
    shifted = confusion(gold, torch.tensor([1, 2, 0, 0, 1, 0]), 3)
    assert macro(perfect, start=1) == 1.0
    assert macro(shifted, start=1) == 0.0
    rows = report(perfect, ("none", "comma", "period"))
    assert rows["scored"] == 5


def test_unsupported_class_is_unmeasured() -> None:
    matrix = torch.zeros((3, 3), dtype=torch.long)
    assert macro(matrix) is None
    assert report(matrix, ("a", "b", "c"))["scored"] == 0


def test_macro_excludes_none_when_requested() -> None:
    matrix = torch.tensor([[90, 0], [10, 0]])
    assert macro(matrix, start=1) == 0
    assert macro(matrix) == pytest.approx(90 / 190)


def test_word_denominator_includes_masked_casing(model: Restorer, example: Example) -> None:
    batch = collate([example], 64)
    batch.casing.fill_(-100)
    metrics = Metrics.create(torch.device("cpu"), len(model.templates))
    with torch.inference_mode():
        metrics.add(model.forward(batch), batch)
    assert metrics.finish(model.templates)["words"] == 3
