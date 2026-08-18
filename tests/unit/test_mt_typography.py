"""Folding a document onto what NLLB's vocabulary can represent.

The table is asserted against the live tokenizer in
`tests/integration/test_mt_typography_vocabulary.py`; here the concern is that folding
changes only the marks it names and nothing else.
"""

from __future__ import annotations

import pytest

from agbalu.mt.typography import (
    FOLD,
    fold,
    prepare,
    strip_emphasis,
    translatable,
)

INVISIBLE = [chr(code) for code in (0xFEFF, 0x200B, 0x00A0, 0x2009)]
"""Byte-order mark, zero-width space, no-break space, thin space, written as code points
because a literal one is invisible in a diff. The normaliser owns these; the fold must leave
them exactly as it found them or two modules are repairing one defect."""


class TestFold:
    @pytest.mark.parametrize(("mark", "ascii_form"), sorted(FOLD.items()))
    def test_every_listed_mark_is_replaced(self, mark: str, ascii_form: str) -> None:
        assert fold(f"a{mark}b") == f"a{ascii_form}b"

    def test_a_line_of_dialogue_loses_no_words(self) -> None:
        assert fold("“Tut, tut, child!”") == '"Tut, tut, child!"'

    def test_kabyle_letters_are_untouched(self) -> None:
        text = "Aɣbalu n tmaziɣt: ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ"
        assert fold(text) == text

    def test_ascii_punctuation_is_untouched(self) -> None:
        text = "\"one\" 'two' three-four, five; six: seven. eight! nine? ten…"
        assert fold(text) == text

    def test_folding_is_idempotent(self) -> None:
        text = "“He said ‘no’—twice.”"
        assert fold(fold(text)) == fold(text)

    def test_the_table_maps_onto_ascii(self) -> None:
        assert all(value.isascii() for value in FOLD.values())

    def test_no_key_is_its_own_value(self) -> None:
        assert all(key != value for key, value in FOLD.items())

    @pytest.mark.parametrize("text", ["", " ", "\n\n", "\t"])
    def test_whitespace_only_input_survives(self, text: str) -> None:
        assert fold(text) == text

    @pytest.mark.parametrize("text", INVISIBLE)
    def test_invisible_characters_are_left_to_the_normaliser(self, text: str) -> None:
        assert fold(text) == text


class TestStripEmphasis:
    def test_gutenberg_emphasis_is_removed(self) -> None:
        assert strip_emphasis("when _I_ find a thing") == "when I find a thing"

    def test_several_spans_on_one_line(self) -> None:
        assert strip_emphasis("_one_ and _two_") == "one and two"

    def test_an_unpaired_marker_is_left_alone(self) -> None:
        assert strip_emphasis("a _lonely marker") == "a _lonely marker"

    def test_a_marker_inside_a_word_is_left_alone(self) -> None:
        assert strip_emphasis("snake_case_name") == "snake_case_name"

    def test_a_span_does_not_cross_a_line(self) -> None:
        text = "_start of one line\nend of another_"
        assert strip_emphasis(text) == text

    def test_an_empty_span_is_left_alone(self) -> None:
        assert strip_emphasis("__") == "__"


class TestTranslatable:
    @pytest.mark.parametrize("text", ["Azul", "1 azul", "ɣ", "a"])
    def test_text_with_letters_is_translatable(self, text: str) -> None:
        assert translatable(text)

    @pytest.mark.parametrize("text", ["", "   ", "*      *      *", "-----", "123", "…"])
    def test_text_without_letters_is_not(self, text: str) -> None:
        assert not translatable(text)

    def test_kabyle_letters_count_as_letters(self) -> None:
        assert translatable("ɣ ɛ ḥ")


class TestPrepare:
    def test_both_passes_are_applied(self) -> None:
        assert prepare("“when _I_ find a thing”") == '"when I find a thing"'

    def test_no_non_whitespace_character_is_lost_except_the_markers(self) -> None:
        source = "“A—B” said _her_ sister."
        assert prepare(source) == '"A-B" said her sister.'
