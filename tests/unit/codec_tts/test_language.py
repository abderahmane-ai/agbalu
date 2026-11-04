"""Preserve ambiguous project-LID prediction sets."""

import pytest

from agbalu.codec_tts.language import validate_identifications


def test_short_kabyle_is_not_rejected_by_argmax() -> None:
    result = validate_identifications(
        [
            {
                "language": "shy_Latn",
                "confidence": 0.5,
                "prediction_set": ("kab_Latn", "shy_Latn"),
                "probabilities": {"kab_Latn": 0.4, "shy_Latn": 0.5},
            }
        ],
        1,
    )
    assert "kab_Latn" in result[0].prediction_set
    assert result[0].language == "shy_Latn"


def test_identifier_cannot_silently_drop_rows() -> None:
    with pytest.raises(ValueError, match="expected=1"):
        validate_identifications([], 1)
