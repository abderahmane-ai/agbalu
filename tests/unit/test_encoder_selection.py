"""Verify deployment promotion and regression controls."""

from agbalu.encoder.checkpoint import State
from agbalu.encoder.configuration import Recipe
from agbalu.encoder.selection import choose


def test_improvement_never_hides_a_corruption_regression() -> None:
    state = State(reference_clean=0.9, reference_corrupted=0.87, best_score=0.91)
    assert not choose(0.95, 0.8, state, Recipe()).eligible
    assert choose(0.92, 0.88, state, Recipe()).improved
    assert not choose(0.911, 0.88, state, Recipe()).improved


def test_nonfinite_or_out_of_range_metrics_are_rejected() -> None:
    for clean in [float("nan"), float("inf"), -0.1, 1.1]:
        assert not choose(clean, 0.9, State(), Recipe()).eligible
