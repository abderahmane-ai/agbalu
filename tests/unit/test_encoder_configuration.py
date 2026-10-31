"""Verify encoder geometry and strict configuration."""

import pytest
from pydantic import ValidationError

from agbalu.encoder.configuration import Geometry, Recipe
from agbalu.encoder.model import Encoder


def test_exact_requested_parameter_count() -> None:
    config = Geometry(backend="reference")
    assert config.parameters == 73_318_016
    assert sum(p.numel() for p in Encoder(config).parameters()) == config.parameters


@pytest.mark.parametrize("fields", [{"hidden_size": 511}, {"layers": 19}, {"surprise": 1}])
def test_invalid_geometry_is_rejected(fields: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Geometry.model_validate(fields)


def test_invalid_accumulation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Recipe(batch_tokens=1000)
