"""Reject incompatible codec speech recipes."""

import pytest
from pydantic import ValidationError

from agbalu.codec_tts.configuration import Recipe


def test_selected_adapter_and_stable_identity() -> None:
    recipe = Recipe()
    assert (recipe.rank, recipe.alpha, recipe.rate_hz) == (32, 64, 24000)
    assert recipe.digest == Recipe.model_validate_json(recipe.model_dump_json()).digest


@pytest.mark.parametrize(
    "value",
    [
        {"rank": 16},
        {"unknown": True},
        {"peak_lr": float("nan")},
        {"validation_fraction": 0.6, "test_fraction": 0.5},
        {"source_revision": "main"},
        {"source": "data/raw", "output": "data/raw/out"},
        {"source": "data/raw", "output": "data"},
        {"min_duration_s": 4, "max_duration_s": 2},
        {"micro_examples": 40},
        {"workers": 100},
    ],
)
def test_invalid_recipe(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate(value)
