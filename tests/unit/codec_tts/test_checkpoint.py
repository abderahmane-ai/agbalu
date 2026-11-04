"""Reject invalid recovery identities before tensor deserialization."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agbalu.codec_tts.checkpoint import NumpyState, RunState, snapshot_path


@pytest.mark.parametrize("name", ["../x", "/outside/a", "step-../x", "step-", "step-a"])
def test_checkpoint_paths_cannot_escape_run(name: str) -> None:
    with pytest.raises(ValueError, match="invalid checkpoint name"):
        snapshot_path(Path("run"), name)


def test_negative_recovery_counts_fail() -> None:
    with pytest.raises(ValidationError):
        RunState(step=-1)
    with pytest.raises(ValidationError):
        NumpyState(keys=(1,), position=0, has_gauss=0, gauss=0)
