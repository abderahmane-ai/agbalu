"""The evaluation-overlap index: what it matches, what it ignores, and its positive control."""

from __future__ import annotations

import pytest

from agbalu.llm.decontaminate import (
    MIN_WORDS,
    WINDOW,
    DecontaminationError,
    Decontaminator,
    first_indexable,
    words,
)

LONG = " ".join(f"w{i}" for i in range(20))
MEDIUM = "the cat sat on the mat by the door"
SHORT = "azul fellawen"


class TestBuild:
    def test_short_sentences_are_counted_and_left_out(self) -> None:
        index = Decontaminator.build([LONG, SHORT])
        assert (index.indexed, index.too_short) == (1, 1)

    def test_an_index_of_nothing_long_enough_is_refused(self) -> None:
        with pytest.raises(DecontaminationError):
            Decontaminator.build([SHORT])


class TestMatches:
    def test_a_shared_window_matches(self) -> None:
        index = Decontaminator.build([LONG])
        window = " ".join(LONG.split()[3 : 3 + WINDOW])
        assert index.matches(f"before {window} after")

    def test_one_word_short_of_a_window_does_not(self) -> None:
        index = Decontaminator.build([LONG])
        assert not index.matches(" ".join(LONG.split()[: WINDOW - 1]))

    def test_a_whole_medium_sentence_matches(self) -> None:
        assert len(words(MEDIUM)) >= MIN_WORDS
        assert Decontaminator.build([MEDIUM]).matches(f"Yesterday {MEDIUM.upper()}!")

    def test_greek_epsilon_and_latin_open_e_are_one_word(self) -> None:
        sentence = "yeḍra-d deg useggas-a mi d-yuɣal ɣer tmurt-is s lɛeqqel d ameqqran"
        greek = sentence.replace("ɛ", "ε")
        assert greek != sentence
        assert Decontaminator.build([sentence]).matches(greek)


class TestWhole:
    def test_a_short_held_out_sentence_is_found_inside_a_document(self) -> None:
        held = "ad kksaɣ laḥzen yellan deg wacḥal ayya"
        assert len(words(held)) < MIN_WORDS
        plain = Decontaminator.build([LONG])
        index = Decontaminator.build([LONG], whole=[held])
        document = f"Tella yiwet n tikkelt. {held.capitalize()}, i as-yenna."
        assert not plain.matches(document)
        assert index.matches(document)

    def test_whole_sentences_under_four_words_are_not_indexed(self) -> None:
        index = Decontaminator.build([LONG], whole=["azul fell-awen"])
        assert not index.matches("azul fell-awen a yimdukal")


class TestControl:
    def test_the_index_fires_on_its_own_sentence(self) -> None:
        index = Decontaminator.build([LONG])
        index.require_positive(LONG)

    def test_a_broken_index_is_refused(self) -> None:
        index = Decontaminator.build([LONG])
        index.grams.clear()
        with pytest.raises(DecontaminationError):
            index.require_positive(LONG)

    def test_the_control_is_long_enough_to_index_whole(self) -> None:
        assert first_indexable([SHORT, MEDIUM, LONG]) == LONG
        with pytest.raises(DecontaminationError):
            first_indexable([SHORT])
