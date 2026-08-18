"""The label scheme, and the invariant everything downstream is built on.

`annotate` and `restore` must be inverses, and `annotate(text).text` must equal
`collation_key(text)` — the corpus builder joins the audio splits against the text corpus on
that key while the dataset trains on those words, so if the two ever disagree the model is
scored against a decontamination that did not decontaminate.
"""

from __future__ import annotations

import unicodedata

import pytest

from agbalu.punctuation.labels import (
    CASE_INDEX,
    PUNCTUATION_INDEX,
    Annotation,
    annotate,
    collation_key,
    restore,
    split_words,
    strip_text,
)

PUNCTUATED = "Err-d Tunhinan, ɛawed-as ḥwaǧeɣ-tt!"
CTC_TARGET = "err-d tunhinan ɛawed-as ḥwaǧeɣ-tt"

ROUND_TRIP = [
    "Azul fell-awen, amek tellid?",
    "Tegzem yiwen n useklu.",
    "Yenna-d: Ad ruḥeɣ.",
    "Ur tt-id-iṣaḥ ara, ɣas akken.",
    "D acu i txedmed assa?",
]


def test_join_matches_the_case_it_must_match() -> None:
    """The positive control: a reference and its CTC target collate to one record."""
    assert collation_key(PUNCTUATED) == collation_key(CTC_TARGET) == CTC_TARGET


def test_join_survives_decomposed_input() -> None:
    decomposed = unicodedata.normalize("NFD", PUNCTUATED)
    assert decomposed != PUNCTUATED
    assert collation_key(decomposed) == collation_key(PUNCTUATED)


def test_join_does_not_merge_a_homoglyph_with_its_repair() -> None:
    """Greek ε U+03B5 is a different letter from Latin ɛ U+025B, and 2.6-3.2% of the seed
    corpora carry it. Contamination is under-counted on unrepaired rows, never over-counted."""
    assert collation_key(PUNCTUATED.replace("ɛ", "ε")) != collation_key(PUNCTUATED)


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_restore_inverts_annotate(text: str) -> None:
    annotation = annotate(text)
    assert restore(annotation.words, annotation.punctuation, annotation.case) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Medlet idlisen-nwen!", "Medlet idlisen-nwen."),
        ("Wagi; wayeḍ.", "Wagi. wayeḍ."),
        ("Ṯamurt-nneɣ AƔBALU d tin.", "Ṯamurt-nneɣ Aɣbalu d tin."),
    ],
)
def test_the_folded_classes_are_not_restored(text: str, expected: str) -> None:
    annotation = annotate(text)
    assert restore(annotation.words, annotation.punctuation, annotation.case) == expected


@pytest.mark.parametrize("text", [*ROUND_TRIP, PUNCTUATED, "«Azul!»", "D'awal-nni yelha.", ""])
def test_annotation_text_equals_the_collation_key(text: str) -> None:
    """The invariant the corpus builder depends on, checked on every shape that reaches it."""
    assert annotate(text).text == collation_key(text) == strip_text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("...", ""),
        ("!?!", ""),
        ("A\u200bB", "ab"),
        ("a\xa0b", "a b"),
        ("a\r\nb", "a b"),
        ("fell-awen", "fell-awen"),
        ("«Azul!»", "azul"),
        ("Azul   fell-awen", "azul fell-awen"),
        ("D'awal", "d awal"),
    ],
)
def test_collation_key_edges(text: str, expected: str) -> None:
    assert collation_key(text) == expected


def test_split_words_drops_parts_with_no_letter_or_digit() -> None:
    assert split_words("a -- b") == ["a", "b"]
    assert split_words("---") == []


@pytest.mark.parametrize(
    ("text", "marks"),
    [
        ("azul, d acu?", ["COMMA", "NONE", "QUESTION"]),
        ("azul , d acu ?", ["COMMA", "NONE", "QUESTION"]),
        ('yenna." ihi', ["PERIOD", "NONE"]),
        ("tellid?» ihi", ["QUESTION", "NONE"]),
        ("atan: wagi", ["COLON", "NONE"]),
    ],
)
def test_marks_attach_to_the_word_they_follow(text: str, marks: list[str]) -> None:
    expected = [PUNCTUATION_INDEX[mark] for mark in marks]  # type: ignore[index]
    assert list(annotate(text).punctuation) == expected


@pytest.mark.parametrize(
    ("text", "cases"),
    [
        ("azul fell-awen", ["LOWER", "LOWER"]),
        ("Azul Ɣef", ["UPPER_INIT", "UPPER_INIT"]),
        ("AƔBALU d tin", ["UPPER_INIT", "LOWER", "LOWER"]),
        ("A d tin", ["UPPER_INIT", "LOWER", "LOWER"]),
    ],
)
def test_case_labels(text: str, cases: list[str]) -> None:
    expected = [CASE_INDEX[case] for case in cases]  # type: ignore[index]
    assert list(annotate(text).case) == expected


def test_empty_text_annotates_to_nothing() -> None:
    annotation = annotate("")
    assert annotation.words == ()
    assert annotation.punctuation == ()
    assert annotation.case == ()
    assert restore((), (), ()) == ""


def test_ragged_annotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="ragged annotation"):
        Annotation(("a", "b"), (0,), (0, 0))
