"""Verify token-based schedules and context transitions."""

import pytest

from agbalu.encoder.configuration import Recipe
from agbalu.encoder.schedule import context_length, learning_rate, mask_probability


def test_rates_and_transition_boundaries() -> None:
    recipe = Recipe(total_tokens=10000)
    assert learning_rate(0, recipe) > 0
    assert learning_rate(200, recipe) == pytest.approx(recipe.peak_lr)
    assert learning_rate(10000, recipe) == pytest.approx(recipe.peak_lr * 0.1)
    assert mask_probability(0, 10000) == pytest.approx(0.3)
    assert mask_probability(7000, 10000) == pytest.approx(0.2)
    assert context_length(7999, recipe) == 512
    assert context_length(8000, recipe) == 1024
