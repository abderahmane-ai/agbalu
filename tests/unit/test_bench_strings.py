"""Exact match and CER over paired strings."""

from __future__ import annotations

import pytest

from agbalu.bench.strings import ScoreError, score_strings


def test_identical_sequences_score_perfectly() -> None:
    score = score_strings(["azul", "tanemmirt"], ["azul", "tanemmirt"])
    assert score.exact_match == 1.0
    assert score.character_error_rate == 0.0
    assert score.characters == len("azul") + len("tanemmirt")


def test_the_rates_are_corpus_totals_not_per_sentence_means() -> None:
    """One error in a long string must not weigh the same as one in a short string.

    A per-sentence mean would give 0.5 * (1/2 + 1/20); the corpus rate is 2/22.
    """
    score = score_strings(["ad", "a" * 20], ["ax", "a" * 19 + "x"])
    assert score.errors == 2
    assert score.characters == 22
    assert score.character_error_rate == pytest.approx(2 / 22)


def test_a_hypothesis_may_be_longer_or_shorter_than_its_reference() -> None:
    assert score_strings(["azul"], ["azulen"]).errors == 2
    assert score_strings(["azulen"], ["azul"]).errors == 2


def test_an_empty_hypothesis_costs_the_whole_reference() -> None:
    score = score_strings(["azul"], [""])
    assert score.errors == 4
    assert score.character_error_rate == 1.0


def test_scoring_nothing_raises_rather_than_returning_zero() -> None:
    """A rate of 0.0 from an empty set reads as a perfect score."""
    with pytest.raises(ScoreError):
        score_strings([], [])


def test_mismatched_lengths_raise() -> None:
    """Silently truncating to the shorter side would score a subset and report a total."""
    with pytest.raises(ValueError, match="argument"):
        score_strings(["a", "b"], ["a"])


def test_zero_length_references_do_not_divide_by_zero() -> None:
    score = score_strings(["", ""], ["", ""])
    assert score.characters == 0
    assert score.character_error_rate == 0.0
    assert score.exact_match == 1.0


def test_unicode_is_counted_by_codepoint_not_by_byte() -> None:
    """`ɣ` is two bytes and one character; a byte-wise distance would report 2."""
    assert score_strings(["aɣ"], ["ax"]).errors == 1
    assert score_strings(["aɣ"], ["aɣ"]).characters == 2
