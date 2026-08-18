"""Folding a shouted heading into the case NLLB was fitted on.

Every example here is a line from the staged corpus or its actual translation. 985 lines of
that corpus are shouted, against 4 genuine off-target decodes in 10,551 sentences, so this is
the larger of the two quality defects by two orders of magnitude.
"""

from __future__ import annotations

import pytest

from agbalu.mt.casing import restore, shouted, soften


class TestShouted:
    @pytest.mark.parametrize(
        "line",
        [
            "CHAPTER III. ATTACK BY STRATAGEM",
            "CHAPTER I. LAYING PLANS",
            "THE MILLENNIUM FULCRUM EDITION 3.0",
            "THE COUNTRY LIFE PRESS, GARDEN CITY, N.Y.",
        ],
    )
    def test_real_shouted_lines_are_detected(self, line: str) -> None:
        assert shouted(line)

    @pytest.mark.parametrize(
        "line",
        [
            "CHAPTER I. Jonathan Harker's Journal",
            "The grey of the morning has passed, and the sun is high.",
            "Chapter III. Attack by stratagem",
            "",
            "   ",
            "*      *      *      *      *",
            "3.0",
        ],
    )
    def test_ordinary_lines_are_not(self, line: str) -> None:
        assert not shouted(line)

    def test_a_single_capitalised_word_is_not_shouting(self) -> None:
        """`CHAPTER` alone carries no case information worth folding, and a name at the head
        of a paragraph must not be rewritten."""
        assert not shouted("CHAPTER")
        assert not shouted("LONDON.")

    def test_an_initial_does_not_make_a_line_shouted(self) -> None:
        """A single letter is upper case in every convention, so it cannot be evidence."""
        assert not shouted("J. Amrouche wrote this")


class TestSoften:
    def test_the_art_of_war_heading(self) -> None:
        """The line that came back `AƔAS wis III. ATTAY S STRATAGEM` — half-untranslated."""
        assert soften("CHAPTER III. ATTACK BY STRATAGEM") == "Chapter III. Attack By Stratagem"

    def test_roman_numerals_keep_their_capitals(self) -> None:
        assert soften("CHAPTER XXVII") == "Chapter XXVII"
        assert soften("BOOK IV. THE END") == "Book IV. The End"

    def test_dotted_initialisms_are_left_alone(self) -> None:
        """`N.Y.` and `M.D.` are not words whose case says anything. A length rule was tried
        first and kept `CITY` and `BY` shouted, which is why the rule is the dots."""
        assert soften("THE COUNTRY LIFE PRESS, GARDEN CITY, N.Y.") == (
            "The Country Life Press, Garden City, N.Y."
        )

    def test_apostrophes_do_not_capitalise_the_wrong_letter(self) -> None:
        """`str.title()` produces `Harker'S`, which is why this is not `str.title()`."""
        assert soften("JONATHAN HARKER'S JOURNAL") == "Jonathan Harker's Journal"

    def test_leading_punctuation_is_preserved(self) -> None:
        assert soften('"THE END OF IT ALL"') == '"The End Of It All"'

    def test_digits_and_symbols_survive(self) -> None:
        assert soften("THE MILLENNIUM FULCRUM EDITION 3.0") == "The Millennium Fulcrum Edition 3.0"

    def test_already_soft_text_is_unchanged_where_it_matters(self) -> None:
        """Applied to every segment, not only shouted ones, so it must not damage prose."""
        assert soften("Chapter III. Attack by stratagem") == "Chapter III. Attack by stratagem"

    @pytest.mark.parametrize("text", ["", "   ", "*  *  *", "3.0", "-- --"])
    def test_textless_input_is_returned_unchanged(self, text: str) -> None:
        assert soften(text) == text

    def test_it_is_idempotent(self) -> None:
        once = soften("CHAPTER III. ATTACK BY STRATAGEM")
        assert soften(once) == once


class TestRestore:
    def test_a_shouted_source_gets_a_shouted_translation(self) -> None:
        assert restore("CHAPTER III. ATTACK BY STRATAGEM", "Tagurt III. Awway s tḥila") == (
            "TAGURT III. AWWAY S TḤILA"
        )

    def test_an_ordinary_source_is_left_alone(self) -> None:
        source = "The grey of the morning has passed."
        assert restore(source, "Yekfa lḥal n ṣṣbeḥ.") == "Yekfa lḥal n ṣṣbeḥ."

    def test_kabyle_specific_letters_uppercase_correctly(self) -> None:
        """`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ` all have upper-case forms, and the release depends on them."""
        assert restore("TITLE HERE", "aɣbalu ḥemmleɣ ṣṣeḥḥa") == "AƔBALU ḤEMMLEƔ ṢṢEḤḤA"

    def test_an_empty_translation_is_not_an_error(self) -> None:
        assert restore("CHAPTER ONE", "") == ""
