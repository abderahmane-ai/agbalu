"""Reject incompatible restoration settings before a job starts."""

import pytest
from pydantic import ValidationError

from agbalu.restoration.configuration import Settings


def test_defaults_reserve_special_tokens_and_full_context() -> None:
    settings = Settings()
    assert settings.core_tokens + settings.left_tokens == 831
    assert settings.geometry.context_tokens == 1024
    assert settings.batch_windows % settings.micro_windows == 0


@pytest.mark.parametrize(
    "payload",
    [{"unknown": 1}, {"micro_windows": 7}, {"core_tokens": 1024}, {"encoder_lr": float("nan")}],
)
def test_invalid_configuration_fails_by_field(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)
