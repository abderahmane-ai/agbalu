from __future__ import annotations

import pytest

from agbalu.parallel.langid import identify, rates
from agbalu.parallel.quality import (
    HARD_DEFECTS,
    MAX_CHARS,
    SOFT_DEFECTS,
    inspect,
    length_ratio,
    numbers,
)

KAB = "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a."
ENG = "I am at home and I will not go to the city today."
FRA = "Je suis a la maison et je n irai pas en ville aujourd hui."


def test_a_clean_pair_has_no_defects() -> None:
    assert inspect(KAB, ENG, "eng").kinds == ()


def test_identical_sides_are_untranslated() -> None:
    assert "untranslated-copy" in inspect(KAB, KAB, "eng").kinds


def test_case_only_difference_still_counts_as_a_copy() -> None:
    assert "untranslated-copy" in inspect(KAB, KAB.upper(), "eng").kinds


@pytest.mark.parametrize(("kab", "foreign"), [("", ENG), (KAB, ""), ("", ""), ("  ", ENG)])
def test_empty_sides_short_circuit(kab: str, foreign: str) -> None:
    assert inspect(kab, foreign, "eng").kinds == ("empty",)


def test_a_french_side_labelled_english_is_flagged() -> None:
    assert "foreign-language-mismatch" in inspect(KAB, FRA, "eng").kinds
    assert "foreign-language-mismatch" not in inspect(KAB, FRA, "fra").kinds


def test_tifinagh_on_the_kabyle_side_is_flagged() -> None:
    assert "kab-wrong-script" in inspect("ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ⴰⵔ ⵜⵉⵎⵍⵉⵍⵉⵜ", ENG, "eng").kinds


def test_a_wild_length_ratio_is_flagged() -> None:
    assert "length-ratio" in inspect(KAB, KAB * 6, "eng").kinds


def test_short_pairs_are_exempt_from_the_length_ratio() -> None:
    """'Yes.' / 'Ih.' is a legitimate 1:1 pair whatever its character ratio."""
    assert "length-ratio" not in inspect("Ih.", "Yes indeed.", "eng").kinds


def test_over_long_sides_are_flagged() -> None:
    assert "too-long" in inspect("a" * (MAX_CHARS + 1), ENG, "eng").kinds


class TestNumbers:
    def test_placeholders_are_not_quantities(self) -> None:
        """`%1$S` digits are argument positions, and translation reorders them."""
        assert numbers("%1$S n %2$S") == numbers("%2$S of %1$S")
        assert "number-mismatch" not in inspect("%1$S n %2$S", "%2$S of %1$S", "eng").kinds

    def test_digit_grouping_is_normalised(self) -> None:
        assert numbers("1,000") == numbers("1 000") == numbers("1.000")

    def test_order_does_not_matter(self) -> None:
        assert numbers("3 seg 5") == numbers("5 of 3")

    def test_a_genuine_difference_is_caught(self) -> None:
        assert "number-mismatch" in inspect("Yella 12 n wussan", "There were 21 days", "eng").kinds

    def test_repeats_are_counted(self) -> None:
        assert numbers("5 5") != numbers("5")

    def test_no_numbers_is_not_a_mismatch(self) -> None:
        assert numbers("ulac") == numbers("none")


def test_url_mismatch() -> None:
    assert "url-mismatch" in inspect(f"{KAB} https://a.example", ENG, "eng").kinds


@pytest.mark.parametrize(
    ("kab", "foreign", "expected"), [("ab", "abcd", 2.0), ("", "abc", 1.0), ("abc", "abc", 1.0)]
)
def test_length_ratio(kab: str, foreign: str, expected: float) -> None:
    assert length_ratio(kab, foreign) == pytest.approx(expected)


def test_hard_and_soft_are_disjoint_and_cover_every_kind() -> None:
    """A defect that is in neither set would vanish from both reported rates."""
    assert not HARD_DEFECTS & SOFT_DEFECTS
    seen = {
        kind
        for case in (
            ("", ENG),
            (KAB, KAB),
            ("ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ⴰⵔ ⵜⵉⵎⵍⵉⵍⵉⵜ", ENG),
            (KAB, KAB * 6),
            ("a" * (MAX_CHARS + 1), ENG),
            ("Yella 12", "There were 21 days here"),
            (f"{KAB} https://a.example", ENG),
            ("a", "b"),
            (KAB, FRA),
        )
        for kind in inspect(case[0], case[1], "eng").kinds
    }
    assert seen <= HARD_DEFECTS | SOFT_DEFECTS
    assert seen


class TestLangid:
    def test_english_and_french_separate(self) -> None:
        assert identify([ENG] * 5) == "eng"
        assert identify([FRA] * 5) == "fra"

    def test_kabyle_is_neither(self) -> None:
        assert identify([KAB] * 5) == "other"

    def test_empty_input_is_other(self) -> None:
        assert identify([]) == "other"
        assert identify(["", "  "]) == "other"

    def test_rates_of_a_blank_string(self) -> None:
        assert rates("") == (0.0, 0.0)
