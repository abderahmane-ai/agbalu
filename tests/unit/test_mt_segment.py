"""Splitting a document for a sentence-level model, and putting it back together.

Two invariants carry the module. `assemble` fed each segment's own text returns the input
byte for byte whenever no hard wrap was joined, and preserves every non-whitespace character
when one was — a splitter that drops a newline silently reformats the document it was asked
to translate, and nothing downstream would notice. And a line break survives as a boundary
unless the line it ends was full, which is what keeps a heading or a line of verse off the
front of the paragraph below it.
"""

from __future__ import annotations

import textwrap

import pytest

from agbalu.mt.segment import (
    MAX_SOURCE_TOKENS,
    WRAP_MARGIN,
    Segment,
    SegmentError,
    assemble,
    ends_sentence,
    plan,
    unwrap,
    wrap_width,
)


def wrapped(paragraphs: list[str], width: int = 71) -> str:
    """A document hard-wrapped the way a plain-text edition is."""
    return "\n\n".join(textwrap.fill(p, width=width) for p in paragraphs)


PROSE = [
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
    "having nothing to do: once or twice she had peeped into the book her sister was "
    "reading, but it had no pictures or conversations in it.",
    "So she was considering in her own mind whether the pleasure of making a daisy-chain "
    "would be worth the trouble of getting up and picking the daisies, when suddenly a "
    "White Rabbit with pink eyes ran close by her.",
]


def words(text: str) -> int:
    """A stand-in for a tokenizer: the budget logic is what is under test, not the counts."""
    return len(text.split())


def identity(text: str, budget: int = MAX_SOURCE_TOKENS) -> str:
    segments = plan(text, words, budget)
    return assemble(segments, [segment.text for segment in segments])


class TestLossless:
    @pytest.mark.parametrize(
        "text",
        [
            "One sentence.",
            "Two sentences. Here is the second.",
            "Paragraph one.\n\nParagraph two, after a blank line.",
            "   leading and trailing whitespace   ",
            "Windows line endings.\r\nSecond line.",
            "\n\nonly newlines around\n\n",
            "Tabs\tinside\tand a sentence. Another.",
            "Azul fell-awen. Amek i tettiliḍ? Ilaq ad nemlil.",
            "No final punctuation",
            "Ellipsis… then more. And a question? Yes!",
            "Quoted «phrase». Then more.",
            "M. Amrouche est arrivé. Il a parlé.",
        ],
    )
    def test_reassembly_returns_the_input(self, text: str) -> None:
        assert identity(text) == text

    def test_reassembly_survives_a_tight_budget(self) -> None:
        text = "A long line with no sentence end that must be split into several pieces here"
        assert identity(text, budget=4) == text

    def test_an_empty_document_has_no_segments(self) -> None:
        assert plan("", words) == []
        assert plan("   \n  ", words) == []

    def test_a_wrapped_document_keeps_every_non_whitespace_character(self) -> None:
        text = wrapped(PROSE)
        segments = plan(text, words)
        rebuilt = assemble(segments, [segment.text for segment in segments])
        assert "".join(rebuilt.split()) == "".join(text.split())

    def test_a_wrapped_document_round_trips_through_unwrap(self) -> None:
        text = wrapped(PROSE)
        segments = plan(text, words)
        assert assemble(segments, [segment.text for segment in segments]) == unwrap(text)


class TestWrapWidth:
    def test_a_wrapped_document_reports_its_width(self) -> None:
        width = wrap_width(wrapped(PROSE * 3, width=71))
        assert width is not None
        assert 71 - WRAP_MARGIN <= width <= 71

    def test_a_document_of_whole_paragraphs_per_line_is_not_wrapped(self) -> None:
        assert wrap_width("\n".join(paragraph for paragraph in PROSE * 6)) is None

    def test_a_short_fragment_is_not_wrapped(self) -> None:
        assert wrap_width("One line.\nAnother line.") is None

    def test_an_empty_document_is_not_wrapped(self) -> None:
        assert wrap_width("") is None
        assert wrap_width("\n\n\n") is None

    def test_a_list_of_equal_length_short_items_is_not_wrapped(self) -> None:
        items = "\n".join(f"line number {index}" for index in range(20))
        assert wrap_width(items) is None


class TestUnwrap:
    def test_every_wrapped_paragraph_becomes_one_line(self) -> None:
        blocks = unwrap(wrapped(PROSE * 3)).split("\n\n")
        assert len(blocks) == 6
        assert all("\n" not in block for block in blocks)

    def test_a_paragraph_break_survives(self) -> None:
        assert "\n\n" in unwrap(wrapped(PROSE))

    def test_an_unwrapped_document_is_returned_unchanged(self) -> None:
        text = "One paragraph.\n\nAnother paragraph."
        assert unwrap(text) == text

    def test_no_non_whitespace_character_is_lost(self) -> None:
        text = wrapped(PROSE)
        assert "".join(unwrap(text).split()) == "".join(text.split())

    def test_unwrapping_is_idempotent(self) -> None:
        text = wrapped(PROSE)
        assert unwrap(unwrap(text)) == unwrap(text)


class TestStructuralLines:
    def test_a_heading_does_not_join_the_paragraph_below_it(self) -> None:
        text = wrapped(PROSE).replace("Alice was", "CHAPTER I.\nDown the Rabbit-Hole\n\nAlice was")
        texts = [segment.text for segment in plan(text, words)]
        assert "Down the Rabbit-Hole" in texts

    def test_a_line_of_verse_is_its_own_segment(self) -> None:
        verse = '"You are old, Father William," the young man said,\n    "And your hair is white;'
        text = f"{wrapped(PROSE)}\n\n{verse}\n\n{wrapped(PROSE)}"
        texts = [segment.text for segment in plan(text, words)]
        assert '"You are old, Father William," the young man said,' in texts
        assert '"And your hair is white;' in texts

    def test_a_scene_break_does_not_absorb_the_line_after_it(self) -> None:
        text = f"{wrapped(PROSE)}\n\n*      *      *\n\nWhat a curious feeling!"
        texts = [segment.text for segment in plan(text, words)]
        assert "*      *      *" in texts
        assert "What a curious feeling!" in texts

    def test_a_table_of_contents_row_stays_one_line(self) -> None:
        rows = "\n".join(f"CHAPTER {n}   The title of chapter {n}" for n in range(2, 12))
        segments = plan(f"Contents\n\n{rows}\n\n{wrapped(PROSE)}", words)
        assert "CHAPTER 5   The title of chapter 5" in [s.text for s in segments]

    def test_no_segment_holds_a_line_break(self) -> None:
        text = f"CHAPTER I.\nDown the Rabbit-Hole\n\n{wrapped(PROSE)}\n\n*  *  *\n"
        assert all("\n" not in segment.text for segment in plan(text, words))


class TestSentences:
    def test_sentences_are_separated(self) -> None:
        segments = plan("First one. Second one.", words)
        assert [s.text for s in segments] == ["First one.", "Second one."]

    def test_the_separator_is_kept_on_the_left_piece(self) -> None:
        segments = plan("First.\n\nSecond.", words)
        assert segments[0].suffix == "\n\n"

    def test_a_question_and_an_exclamation_end_sentences(self) -> None:
        segments = plan("Amek? Azul! Ihi.", words)
        assert len(segments) == 3

    def test_a_closing_quote_stays_with_its_sentence(self) -> None:
        segments = plan('He said "go home." Then he left.', words)
        assert segments[0].text == 'He said "go home."'

    @pytest.mark.parametrize("abbreviation", ["M.", "Mme.", "etc.", "cf.", "Dr."])
    def test_an_abbreviation_does_not_end_a_sentence(self, abbreviation: str) -> None:
        segments = plan(f"Il a vu {abbreviation} le matin. Puis il est parti.", words)
        assert len(segments) == 2

    def test_an_initial_does_not_end_a_sentence(self) -> None:
        segments = plan("J. Amrouche wrote it. She sang it.", words)
        assert len(segments) == 2

    def test_a_decimal_point_is_not_a_sentence_end(self) -> None:
        """No whitespace follows the point, so the boundary never matches."""
        assert len(plan("It cost 3.50 in total. Then more.", words)) == 2


class TestBudget:
    def test_a_sentence_within_budget_is_one_unit(self) -> None:
        assert len(plan("Three words here.", words, 10)) == 1

    def test_an_over_long_sentence_splits_at_clauses(self) -> None:
        text = "one two three, four five six, seven eight nine"
        pieces = [s.text for s in plan(text, words, 4)]
        assert len(pieces) == 3
        assert pieces[0] == "one two three,"

    def test_a_sentence_with_no_punctuation_splits_at_words(self) -> None:
        pieces = plan("alpha beta gamma delta epsilon", words, 2)
        assert len(pieces) == 3
        assert all(words(p.text) <= 2 for p in pieces)

    def test_every_unit_respects_the_budget_where_it_can(self) -> None:
        text = "one two three, four five six seven eight, nine ten"
        assert all(words(s.text) <= 3 for s in plan(text, words, 3))

    def test_a_sentence_past_the_target_splits_at_clauses_although_it_fits(self) -> None:
        """The King James genealogies chain verses with semicolons and no sentence-final
        punctuation, so one reached the encoder at 641 tokens — inside the hard limit and
        far outside anything NLLB was trained on."""
        chain = " ".join(f"and number {n} begat number {n + 1};" for n in range(60))
        units = [segment.text for segment in plan(chain, words, budget=1022, target=40)]
        assert len(units) > 1
        assert all(words(unit) <= 40 for unit in units)

    def test_clauses_are_packed_rather_than_emitted_one_by_one(self) -> None:
        """Trading one over-long unit for a dozen fragments is off the distribution in the
        other direction."""
        chain = "; ".join(f"clause number {n}" for n in range(30))
        units = plan(chain, words, budget=1022, target=20)
        assert all(words(unit.text) > 10 for unit in units[:-1])

    def test_a_long_unbroken_clause_is_kept_whole_below_the_budget(self) -> None:
        """Only a real clause boundary is used to come under the target: cutting a phrase
        mid-way is worse than handing over a long sentence."""
        run_on = " ".join(f"word{n}" for n in range(80)) + "."
        assert len(plan(run_on, words, budget=1022, target=20)) == 1

    def test_the_hard_budget_still_wins_where_there_is_no_clause(self) -> None:
        run_on = " ".join(f"word{n}" for n in range(80)) + "."
        units = plan(run_on, words, budget=25, target=20)
        assert all(words(unit.text) <= 25 for unit in units)

    def test_commas_are_only_reached_when_the_stronger_marks_are_not_enough(self) -> None:
        """A comma fences an apposition or a list whose halves belong together, so a sentence
        that comes under the target on its semicolons keeps its commas."""
        text = "alpha, beta and gamma were there; delta, epsilon and zeta were not there."
        units = [segment.text for segment in plan(text, words, budget=1022, target=8)]
        assert "alpha, beta and gamma were there;" in units

    def test_commas_are_used_when_there_is_nothing_stronger(self) -> None:
        listing = ", ".join(f"item number {n}" for n in range(20)) + "."
        units = [segment.text for segment in plan(listing, words, budget=1022, target=12)]
        assert len(units) > 1
        assert all(words(unit) <= 12 for unit in units)

    def test_a_numbered_heading_ends_a_sentence(self) -> None:
        """`\\w` matches digits, so `Chapter 5.` was read as an initial and never split."""
        units = [segment.text for segment in plan("Chapter 5. Then it began.", words)]
        assert units == ["Chapter 5.", "Then it began."]

    def test_an_initial_still_does_not_end_a_sentence(self) -> None:
        units = [segment.text for segment in plan("Written by J. Amrouche in Paris.", words)]
        assert units == ["Written by J. Amrouche in Paris."]

    def test_a_target_above_the_budget_is_the_budget(self) -> None:
        chain = "; ".join(f"clause number {n}" for n in range(30))
        assert plan(chain, words, budget=10, target=500) == plan(chain, words, budget=10, target=10)

    def test_a_single_word_longer_than_the_budget_is_kept_whole(self) -> None:
        """Truncating it would lose text; an over-long unit is the lesser defect."""
        segments = plan("supercalifragilistic", words, 1)
        assert [s.text for s in segments] == ["supercalifragilistic"]

    @pytest.mark.parametrize("budget", [0, -1])
    def test_a_non_positive_budget_is_refused(self, budget: int) -> None:
        with pytest.raises(SegmentError, match="must be positive"):
            plan("text", words, budget)


class TestAssemble:
    def test_translations_land_where_their_sources_were(self) -> None:
        segments = plan("First.\n\nSecond.", words)
        assert assemble(segments, ["Tamezwarut.", "Tis snat."]) == "Tamezwarut.\n\nTis snat."

    def test_a_count_mismatch_is_refused(self) -> None:
        segments = plan("First. Second.", words)
        with pytest.raises(SegmentError, match="2 segments"):
            assemble(segments, ["only one"])

    def test_leading_whitespace_is_restored(self) -> None:
        segments = plan("\n  Azul.", words)
        assert assemble(segments, ["Hello."]) == "\n  Hello."

    def test_an_empty_plan_assembles_to_nothing(self) -> None:
        assert assemble([], []) == ""


class TestEndsSentence:
    @pytest.mark.parametrize("text", ["It ended.", "Really!", "Who?", "Said 'yes.'"])
    def test_real_endings(self, text: str) -> None:
        assert ends_sentence(text)

    @pytest.mark.parametrize("text", ["Voir M.", "and etc.", "by J."])
    def test_non_endings(self, text: str) -> None:
        assert not ends_sentence(text)

    def test_a_segment_carries_no_surrounding_space_by_default(self) -> None:
        assert Segment(text="x").restore("y") == "y"
