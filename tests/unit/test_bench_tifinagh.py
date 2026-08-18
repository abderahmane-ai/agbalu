"""Script-conversion scoring, and the character table it is read against."""

from __future__ import annotations

import pytest

from agbalu.bench.tifinagh import (
    LATN_TO_TFNG,
    TFNG_TO_LATN,
    score_conversion,
    score_schwa,
    skeleton,
    to_latin,
    to_tifinagh,
)


def test_the_two_tables_are_inverses_except_where_the_script_loses_information() -> None:
    for latin, tifinagh in LATN_TO_TFNG.items():
        assert TFNG_TO_LATN[tifinagh] == latin


def test_the_table_round_trips_a_sentence_that_contains_no_schwa() -> None:
    assert to_latin(to_tifinagh("azul")) == "azul"


def test_the_table_cannot_restore_a_schwa_the_corpus_does_not_write() -> None:
    """The whole reason a model exists: Kabyle Tifinagh omits `e`, so the table cannot."""
    assert to_latin("ⵜⵛⴼⵉⴹ") == "tcfiḍ"


def test_the_table_folds_case_and_passes_punctuation_through() -> None:
    assert to_tifinagh("Azul !") == "ⴰⵣⵓⵍ !"


def test_skeleton_removes_every_schwa_and_records_where_it_was() -> None:
    """The index is how many non-`e` characters precede it — `tud` is three, so `tudert`
    is `tudrt` with a schwa at 3."""
    assert skeleton("tudert") == ("tudrt", (3,))
    assert skeleton("azul") == ("azul", ())


def test_skeleton_folds_case_so_two_spellings_of_one_word_agree() -> None:
    assert skeleton("Tudert") == skeleton("tudert")


def test_schwa_scoring_is_positional_not_a_count() -> None:
    """The defect this metric exists to replace.

    `tuedrt` has exactly as many `e` as `tudert` and puts it one consonant early. A count
    of them scores that as a perfect match; the position does not.
    """
    counted = score_schwa(["tudert"], ["tuedrt"])
    assert counted.true_positives == 0
    assert counted.false_positives == 1
    assert counted.false_negatives == 1
    assert counted.f1 == 0.0


def test_a_correctly_placed_schwa_scores() -> None:
    placed = score_schwa(["tudert"], ["tudert"])
    assert (placed.true_positives, placed.false_positives, placed.false_negatives) == (1, 0, 0)
    assert placed.f1 == 1.0
    assert placed.undefined == 0


def test_a_wrong_consonant_makes_every_schwa_in_the_sentence_undefined() -> None:
    """A hypothesis that changed the skeleton has no position to be judged against.

    Both columns are charged rather than the row dropped: dropping it would score a model
    that failed the sentence outright as if it had abstained.
    """
    broken = score_schwa(["tudert"], ["tudirt"])
    assert broken.undefined == 1
    assert broken.false_negatives == 1
    assert broken.false_positives == 0


def test_two_schwas_at_one_skeleton_position_are_matched_once_each() -> None:
    """A set intersection would score the second as missing however it was predicted."""
    doubled = score_schwa(["tee"], ["tee"])
    assert doubled.true_positives == 2
    assert doubled.false_negatives == 0


def test_a_hypothesis_with_no_schwa_at_all_has_no_precision_to_report() -> None:
    none = score_schwa(["tudert"], ["tudrt"])
    assert none.recall == 0.0
    assert none.precision == 0.0
    assert none.f1 == 0.0


def test_an_empty_set_reports_zero_rather_than_dividing_by_zero() -> None:
    empty = score_schwa([], [])
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


def test_the_report_dictionaries_carry_their_counts() -> None:
    """A rate without its denominator cannot be pooled across splits."""
    assert set(score_schwa(["tudert"], ["tudert"]).as_dict()) == {
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "undefined_sentences",
    }
    assert score_conversion(["azul"], ["azul"]).as_dict()["sentences"] == 1


@pytest.mark.parametrize("text", ["", "ⴰ", "azul d ameqqran", "Ɣef 3 n wussan!"])
def test_the_table_never_raises_on_input_it_has_no_mapping_for(text: str) -> None:
    assert isinstance(to_latin(text), str)
    assert isinstance(to_tifinagh(text), str)
