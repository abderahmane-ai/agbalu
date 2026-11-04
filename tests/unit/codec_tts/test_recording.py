"""Reject malformed prepared recording schemas."""

import pytest
from pydantic import ValidationError

from agbalu.codec_tts.artifacts import Pair, Recording


def test_missing_recording_fields_are_not_defaulted() -> None:
    with pytest.raises(ValidationError, match="audio_sha256"):
        Recording.model_validate({"record_id": "broken"})


def test_unknown_pair_fields_and_split_fail() -> None:
    with pytest.raises(ValidationError, match="split"):
        Pair.model_validate({"target": "x", "reference": "y", "speaker": "s", "split": "dev"})
