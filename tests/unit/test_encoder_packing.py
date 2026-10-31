"""Verify word grouping and lossless segmentation."""

import pytest

from agbalu.encoder.packing import segments, word_groups


def test_byte_offsets_preserve_unicode_and_hyphen_chains() -> None:
    assert word_groups("ɣer yefka-yas", [(0, 4), (4, 10), (10, 11), (11, 14)]) == [0, 1, 1, 1]


def test_segments_preserve_every_token_and_do_not_split_fitting_groups() -> None:
    groups = [0, 0, 2, 2, 4, 4, 6, 6, 8]
    ranges = segments(0, len(groups), groups, 5)
    assert ranges == [(0, 4), (4, 9)]
    assert [i for start, end in ranges for i in range(start, end)] == list(range(9))


def test_oversized_groups_still_conserve_tokens() -> None:
    assert segments(0, 8, [0] * 8, 3) == [(0, 3), (3, 6), (6, 8)]


def test_invalid_offsets_report_the_value() -> None:
    with pytest.raises(ValueError, match=r"piece 0.*\(0, 9\)"):
        word_groups("abc", [(0, 9)])
