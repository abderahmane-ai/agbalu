"""Verify reference conservation and restoration labels on Unicode edge cases."""

import unicodedata

import pytest

from agbalu.restoration.annotations import IGNORE, Document, annotate, case_label, parse_gap


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "«Azul, a mmi!»",
        "Axxam-a yewweḍ-d.",
        "Yenna: («D acu?!»)",
        "a\u0301b",
        "A\u200bb",
        "A\xa0b",
        "\ufeffAzul.",
        "Azul\u200f fell-awen.",
    ],
)
def test_every_reference_character_is_accounted_for(text: str) -> None:
    document = annotate(text)
    rebuilt = document.gaps[0].raw + "".join(
        document.reference[start:end] + document.gaps[i + 1].raw
        for i, (start, end) in enumerate(document.spans)
    )
    assert rebuilt == unicodedata.normalize("NFC", text)


def test_hyphens_and_combining_marks_stay_with_their_word() -> None:
    assert annotate("Ass-a, yewweḍ-d umdakkel-is.").words == ("ass-a", "yewweḍ-d", "umdakkel-is")
    assert annotate("a\u0301b").words == ("áb",)


def test_labels_include_leading_structures_and_terminal_gap() -> None:
    document = annotate("«Azul, a mmi!»")
    assert [gap.mark for gap in document.gaps] == [0, 1, 0, 4]
    assert document.gaps[0].template == "@«"
    assert document.gaps[-1].template == "@»"
    assert document.casing == (1, 0, 0)


@pytest.mark.parametrize(
    ("word", "label"),
    [("a", 0), ("A", 1), ("Ɣer", 1), ("AƔBALU", 2), ("taQbaylit", IGNORE), ("2026", IGNORE)],
)
def test_casing_has_explicit_unrepresentable_cases(word: str, label: int) -> None:
    assert case_label(word) == label


def test_compound_marks_have_explicit_normalization() -> None:
    assert parse_gap(" !? ").mark == 8
    assert parse_gap("... »").mark == 7
    assert not parse_gap("!!").valid


def test_numeric_punctuation_stays_inside_lexical_spans() -> None:
    document = annotate("Azal-is 3,14; 12:30, 01/11/2026.")
    assert document.words == ("azal-is", "3,14", "12:30", "01/11/2026")
    assert [gap.mark for gap in document.gaps] == [0, 0, 6, 1, 2]


def test_conservation_positive_control_rejects_a_missing_character() -> None:
    document = annotate("Azul.")
    with pytest.raises(ValueError, match="conservation"):
        Document("Azul!", document.words, document.spans, document.casing, document.gaps)
