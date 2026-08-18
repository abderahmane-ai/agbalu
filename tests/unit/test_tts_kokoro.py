"""Folding a Kabyle reading onto the base's own symbols.

The fold is what takes the inventory shortfall from four symbols to three, so it has to
remove the tie bar from every reading that carries one and change nothing else. It runs on
every utterance the corpus keeps, which makes it the one place a silent rewrite of the
phoneme string could enter.
"""

from __future__ import annotations

from agbalu.tts import g2p
from agbalu.tts.kokoro import FOLD, fold


class TestFolding:
    def test_the_tie_bar_sequences_become_single_symbols(self) -> None:
        assert fold("t͡ʃ") == "ʧ"
        assert fold("d͡ʒ") == "ʤ"

    def test_folding_leaves_everything_else_alone(self) -> None:
        assert fold("æzulː") == "æzulː"

    def test_folding_is_idempotent(self) -> None:
        once = fold(g2p.phonemize_word("ččuṛ"))
        assert fold(once) == once

    def test_folding_an_empty_string_is_empty(self) -> None:
        assert fold("") == ""

    def test_every_fold_target_is_a_single_symbol(self) -> None:
        """The shortfall arithmetic depends on it: a two-character replacement would need
        its own rows rather than reusing ones the base already has."""
        assert all(len(symbol) == 1 for symbol in FOLD.values())

    def test_a_folded_reading_carries_no_tie_bar(self) -> None:
        reading = g2p.phonemize_word("aǧǧaǧ")
        assert "͡" in reading
        assert "͡" not in fold(reading)

    def test_folding_a_geminate_affricate_keeps_its_length_mark(self) -> None:
        assert fold(g2p.phonemize_word("aǧǧaǧ")).count(g2p.LENGTH) == 1

    def test_folding_introduces_nothing_outside_the_fold_table(self) -> None:
        for word in ("azul", "tameṭṭut", "aɣbalu", "ččuṛ", "ḥemmleɣ", "ɛemmi", "aǧǧaǧ"):
            reading = g2p.phonemize_word(word)
            assert set(fold(reading)) - set(reading) <= set(FOLD.values())

    def test_a_word_with_no_affricate_is_untouched(self) -> None:
        for word in ("azul", "tameṭṭut", "aɣbalu", "ḥemmleɣ", "ɛemmi"):
            reading = g2p.phonemize_word(word)
            assert fold(reading) == reading

    def test_folding_never_lengthens_a_reading(self) -> None:
        for word in ("ččuṛ", "aǧǧaǧ", "azul"):
            reading = g2p.phonemize_word(word)
            assert len(fold(reading)) <= len(reading)
