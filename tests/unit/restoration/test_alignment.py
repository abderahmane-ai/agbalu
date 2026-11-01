"""Positive-control byte alignment without depending on a tokenizer artifact."""

import pytest

from agbalu.restoration.alignment import EncodedWord, align_cases


def test_component_spans_use_bytes_and_keep_distinct_targets() -> None:
    encoded = EncodedWord((7, 8, 9), ((0, 4), (4, 5), (5, 9)))
    assert align_cases("ɣer-Ɣer", encoded) == ((0, 1, 0, 3, 7, 0), (2, 3, 4, 7, 7, 1))


def test_shared_pieces_preserve_component_position_features() -> None:
    encoded = EncodedWord((7,), ((0, 11),))
    assert align_cases("d-Taqbaylit", encoded) == ((0, 1, 0, 1, 11, 0), (0, 1, 2, 11, 11, 1))


def test_missing_piece_alignment_fails_with_the_exact_word() -> None:
    with pytest.raises(ValueError, match="Taqbaylit"):
        align_cases("d-Taqbaylit", EncodedWord((7,), ((0, 1),)))
