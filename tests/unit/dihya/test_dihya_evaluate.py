"""Verify support-aware metrics and finite-sample conformal ranks."""

from __future__ import annotations

import pytest

from agbalu.dihya.dataset import Example, TextDataset
from agbalu.dihya.evaluate import classification_metrics, conformal_quantile, partition_development


def test_calibration_rows_are_reserved_before_model_selection() -> None:
    dataset = TextDataset(
        [Example(f"Azul fell awen {index}", "kab_Latn", "web", "test") for index in range(40)]
    )
    selection, fit, coverage = partition_development(dataset, 42)
    assert len(selection) == 20
    assert len(fit) == len(coverage) == 10
    first = {example.text for example in selection.examples}
    second = {example.text for example in fit.examples}
    third = {example.text for example in coverage.examples}
    assert not first & second
    assert not first & third
    assert not second & third
    assert len(first | second | third) == 40


def test_metrics_match_known_confusion_and_report_tiny_cell() -> None:
    report = classification_metrics(
        [0, 1, 1, 0], [0, 0, 1, 1], ["a/web", "a/web", "b/web", "b/tiny"], ["a", "b"], 2
    )
    assert report["accuracy"] == 0.5
    assert report["macro_f1"] == 0.5
    assert report["worst_group_recall"] == 0.5
    assert report["raw_worst_group_recall"] == 0.0
    assert report["selection_score"] == 0.5
    assert report["excluded_groups"] == ["b/tiny", "b/web"]
    assert report["group_support"]["b/tiny"] == 1
    assert report["confusion"] == [[1, 1], [1, 1]]


def test_unmeasured_groups_and_classes_are_none() -> None:
    report = classification_metrics([0], [0], ["a/tiny"], ["a", "b"])
    assert report["worst_group_recall"] is None
    assert report["class_recalls"]["b"] is None
    assert report["class_support"]["b"] == 0
    assert report["selection_score"] == 1


def test_macro_f1_improves_selection_while_worst_recall_remains_zero() -> None:
    first = classification_metrics([1, 0], [0, 1], ["a/web", "b/web"], ["a", "b"], 1)
    second = classification_metrics([0, 0], [0, 1], ["a/web", "b/web"], ["a", "b"], 1)
    assert first["worst_group_recall"] == second["worst_group_recall"] == 0
    assert second["selection_score"] > first["selection_score"]


def test_conformal_small_sample_uses_maximum_possible_score() -> None:
    assert conformal_quantile([0.1, 0.2], 0.99) == 1.0
    assert conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.5) == 0.3


@pytest.mark.parametrize("scores", [[], [float("nan")], [-0.1], [1.1]])
def test_calibration_rejects_invalid_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError, match=r"Calibration needs|Invalid nonconformity"):
        conformal_quantile(scores, 0.9)
