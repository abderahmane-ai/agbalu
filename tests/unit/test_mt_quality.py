"""Naming the segments whose translation failed, so only those are decoded again.

The detector's cost is asymmetric. A missed failure is a ruined paragraph in the output; a
false positive costs one extra decode and risks replacing a good translation with one made
under penalties. So the thresholds are set where legitimate text does not reach them, and
the cases that matter here are the ones real runs produced.
"""

from __future__ import annotations

import pytest

from agbalu.mt.quality import copied, empty, failed, loops, malformed_length


class TestLoops:
    def test_the_alice_failure(self) -> None:
        assert loops("Ay " * 253)

    def test_the_duchess_failure(self) -> None:
        assert loops("Tamecṭuḥt " * 25)

    def test_a_two_word_phrase_repeating(self) -> None:
        assert loops("d acu d acu d acu d acu d acu")

    def test_normal_kabyle_prose_does_not_loop(self) -> None:
        text = "Yenna-yas: d acu i txedmeḍ ass-a? Nekk ur ẓriɣ ara, maca ad ruḥeɣ ɣer temdint."
        assert not loops(text)

    def test_a_litany_of_three_is_not_a_loop(self) -> None:
        """`Holy, holy, holy` is scripture, not degeneration, and the New Testament has it."""
        assert not loops("Holy holy holy is the Lord God Almighty")

    def test_a_genealogy_is_not_a_loop(self) -> None:
        """The King James genealogies repeat by design and must survive untouched."""
        chain = " ".join(f"and {name} begat {name}son" for name in ("Abraham", "Isaac", "Jacob"))
        assert not loops(chain)

    @pytest.mark.parametrize("text", ["", " ", "one", "one two three"])
    def test_short_input_cannot_loop(self, text: str) -> None:
        assert not loops(text)

    def test_a_repeat_longer_than_the_window_is_left_alone(self) -> None:
        phrase = "one two three four five six "
        assert not loops(phrase * 4)


class TestEmpty:
    def test_a_source_with_words_and_no_output(self) -> None:
        assert empty("Azul fell-awen", "")

    def test_whitespace_output_counts_as_empty(self) -> None:
        assert empty("Azul fell-awen", "     ")

    def test_a_source_without_letters_is_not_a_failure(self) -> None:
        """A scene break is copied through, so a blank result for one is correct."""
        assert not empty("*   *   *", "")

    def test_a_normal_pair_is_not_empty(self) -> None:
        assert not empty("Hello", "Azul")


class TestMalformedLength:
    def test_a_hallucinated_expansion(self) -> None:
        source = "Oh dear, what nonsense I am talking, said Alice to herself very quietly"
        assert malformed_length(source, "ay " * 200)

    def test_a_source_too_short_for_a_ratio_is_left_to_the_loop_check(self) -> None:
        """Three words is under `RATIO_MIN_WORDS`, so the ratio abstains — a short source
        can legitimately double or halve — and `loops` is what catches it. The division of
        labour, not a gap."""
        short = "Oh dear me"
        assert not malformed_length(short, "ay " * 200)
        assert failed(short, "ay " * 200)

    def test_a_dropped_sentence(self) -> None:
        source = "I know what it means well enough, when I find a thing, said the Duck"
        assert malformed_length(source, "Asteqsi")

    def test_kabyle_running_shorter_than_english_is_fine(self) -> None:
        """Kabyle is 23,070 words against Alice's 26,525 — 0.87, well inside the band."""
        source = " ".join(["word"] * 100)
        assert not malformed_length(source, " ".join(["awal"] * 87))

    def test_a_short_source_is_not_judged_on_ratio(self) -> None:
        assert not malformed_length("Yes.", "Ih, akka i gella wawal-nni.")

    def test_the_boundaries(self) -> None:
        source = " ".join(["word"] * 20)
        assert not malformed_length(source, " ".join(["awal"] * 10))
        assert malformed_length(source, " ".join(["awal"] * 4))
        assert malformed_length(source, " ".join(["awal"] * 61))


class TestFailed:
    def test_a_good_translation_passes(self) -> None:
        source = "The old man of the village went out to the great market this morning."
        assert not failed(source, "Amɣar n taddart yeffeɣ ɣer ssuq ameqqran ass-a taṣebḥit.")

    @pytest.mark.parametrize(
        ("source", "hypothesis"),
        [
            ("Oh dear, what nonsense I am talking!", "Ay " * 253),
            ("I know what it means well enough when I find a thing", "   "),
            ("Tut, tut, child! Everything has got a moral if only you can find it", "Aṭu " * 90),
        ],
    )
    def test_every_failure_a_real_run_produced_is_caught(
        self, source: str, hypothesis: str
    ) -> None:
        assert failed(source, hypothesis)


class TestCopied:
    """A segment that comes back fluent, well-formed, the right length and still in the
    source language passes all three of the other checks. Both examples are verbatim from
    the translated documents."""

    def test_the_dracula_dialect_line(self) -> None:
        source = "'E's been a-gettin' over some bloomin' wall or other."
        assert copied(source, source)
        assert failed(source, source)

    def test_the_alice_hatter_line(self) -> None:
        source = (
            '"I am a poor man, Your Majesty", the Hatter began, in a trembling voice, '
            '"and I hadn\'t started my tea-not over a week or so"'
        )
        assert copied(source, source)

    def test_the_typography_fold_does_not_hide_a_copy(self) -> None:
        """The source reaches the model with `’` folded to `'`, so the copy that comes back
        differs from the file by exactly those characters and must still be caught."""
        original = "’E’s been a-gettin’ over some bloomin’ wall or other."
        folded = "'E's been a-gettin' over some bloomin' wall or other."
        assert copied(original, folded)

    def test_a_real_translation_is_not_a_copy(self) -> None:
        source = "The old man of the village went out to the great market this morning."
        assert not copied(source, "Amɣar n taddart yeffeɣ ɣer ssuq ameqqran ass-a taṣebḥit.")

    def test_the_densest_proper_noun_run_in_the_corpus_is_not_flagged(self) -> None:
        """The tightest false-positive case there is: a correct translation whose words are
        almost all names. Measured at 0.69 against a 0.9 threshold."""
        source = "Letter, Abraham Van Helsing, M.D., D.Ph., D.Lit., etc., etc., to Dr. Seward."
        hypothesis = "Tabrat, Abraham Van Helsing, M. D., D. Ph., D. Lit., atg, atg, i Dr. Seward."
        assert not copied(source, hypothesis)
        assert not failed(source, hypothesis)

    def test_case_is_ignored_so_a_softened_copy_is_caught(self) -> None:
        """A shouted line reaches the model title-cased, so a copy comes back in that case
        and differs from the source in nothing else."""
        source = "THE COUNTRY LIFE PRESS OF GARDEN CITY IN THE STATE OF NEW YORK"
        assert copied(source, "The Country Life Press Of Garden City In The State Of New York")

    def test_a_short_segment_is_not_judged(self) -> None:
        """A heading legitimately keeps its numerals and names, so below the word floor an
        overlap says nothing. `Chapter III` translating to `Chapter III` is not a defect
        this check can distinguish from a correct one."""
        assert not copied("Chapter III.", "Chapter III.")

    def test_an_empty_hypothesis_is_left_to_the_empty_check(self) -> None:
        assert not copied("The old man of the village went out to the market", "")

    def test_partial_overlap_below_the_threshold_passes(self) -> None:
        """Kabyle borrows French and English nouns, so some overlap is normal and only a
        near-total copy is a failure."""
        source = "The telephone in the hotel office rang for the doctor at midnight."
        hypothesis = "Telefun n hotel office yesṛeɣ i doctor deg midnight n yiḍ ass-nni."
        assert not copied(source, hypothesis)
