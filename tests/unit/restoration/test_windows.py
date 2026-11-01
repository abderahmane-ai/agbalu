"""Prove whole-word supervision covers every eligible position once."""

import pytest

from agbalu.restoration.windows import tile


@pytest.mark.parametrize("lengths", [[1], [1] * 400, [2, 9, 5, 15, 7, 4], [35, 1]])
def test_every_word_has_exactly_one_core_owner(lengths: list[int]) -> None:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    windows = tile(offsets, 62, 32, 15)
    assert [
        index for window in windows for index in range(window.core_start, window.core_stop)
    ] == list(range(len(lengths)))
    for window in windows:
        assert offsets[window.stop] - offsets[window.start] <= 62
        assert window.start <= window.core_start < window.core_stop <= window.stop


@pytest.mark.parametrize("offsets", [[], [1], [0, 0], [0, 100]])
def test_invalid_and_oversized_words_fail_explicitly(offsets: list[int]) -> None:
    with pytest.raises(ValueError, match=r"invalid|exceeds"):
        tile(offsets, 62, 32, 15)


def test_streaming_lookahead_does_not_expand_when_history_is_short() -> None:
    windows = tile(list(range(2001)), 1022, 256, 638, right_tokens=128)
    assert windows[0].stop == 384
    assert all(window.stop - window.core_stop <= 128 for window in windows)
    assert [
        word for window in windows for word in range(window.core_start, window.core_stop)
    ] == list(range(2000))
