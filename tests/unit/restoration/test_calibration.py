"""Verify output calibration against known class frequencies."""

import pytest
import torch

from agbalu.restoration.calibration import Calibration, fit_intercepts
from agbalu.restoration.dataset import Windows
from agbalu.restoration.evaluation import fold_indices
from agbalu.restoration.model import Scores
from agbalu.restoration.windows import Window


def test_intercepts_recover_shifted_presence_and_case_priors() -> None:
    marks = torch.tensor([1] * 20 + [0] * 80)
    labels = torch.tensor([0] * 70 + [1] * 20 + [2] * 10)
    fitted = fit_intercepts(torch.full((100,), 2.0), marks, torch.zeros(100, 3), labels)
    probability = torch.tensor(2.0 + fitted.presence_bias).sigmoid()
    assert float(probability) == pytest.approx(0.2, abs=0.005)
    torch.testing.assert_close(
        torch.tensor(fitted.casing_bias).softmax(-1),
        torch.tensor([0.7, 0.2, 0.1]),
        atol=0.005,
        rtol=0,
    )


def test_empty_calibration_is_refused() -> None:
    with pytest.raises(ValueError, match="support"):
        fit_intercepts(torch.empty(0), torch.empty(0), torch.empty(0, 3), torch.empty(0))


def test_identity_calibration_keeps_probabilities() -> None:
    scores = Scores(torch.ones(2), torch.ones(2, 8), torch.ones(2, 3), torch.ones(2, 1))
    produced = Calibration().apply(scores)
    torch.testing.assert_close(scores.presence, produced.presence)
    torch.testing.assert_close(scores.casing, produced.casing)
    torch.testing.assert_close(scores.punctuation, produced.punctuation)


def test_type_bias_recovers_rare_mark_base_rates() -> None:
    counts = (700, 150, 50, 30, 25, 20, 15, 10)
    marks = torch.tensor([label + 1 for label, count in enumerate(counts) for _ in range(count)])
    fitted = fit_intercepts(
        torch.zeros(1000),
        marks,
        torch.zeros(1000, 3),
        torch.zeros(1000, dtype=torch.long),
        torch.zeros(1000, 8),
    )
    torch.testing.assert_close(
        torch.tensor(fitted.type_bias).softmax(-1),
        torch.tensor(counts) / 1000,
        atol=0.01,
        rtol=0,
    )


def test_calibration_and_selection_never_share_a_document() -> None:
    dataset = Windows.__new__(Windows)
    dataset.windows = [(document, Window(0, 3, 0, 3)) for document in range(20) for _ in range(5)]
    calibration = fold_indices(dataset, "calibration", None)
    selection = fold_indices(dataset, "selection", None)
    left = {dataset.windows[int(index)][0] for index in calibration}
    right = {dataset.windows[int(index)][0] for index in selection}
    assert not left & right
    assert len(left | right) == 20
    assert len(calibration) + len(selection) == len(dataset)
