"""Prove record accounting rejects deliberate data loss."""

import pytest

from agbalu.codec_tts.artifacts import verify_conservation


def test_complete_accounting() -> None:
    verify_conservation(10, 7, 3)
    verify_conservation(0, 0, 0)


@pytest.mark.parametrize("counts", [(10, 7, 2), (10, 8, 3), (-1, -1, 0)])
def test_detect_missing_duplicate_and_negative_rows(counts: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match=r"input=.*output=.*rejected="):
        verify_conservation(*counts)
