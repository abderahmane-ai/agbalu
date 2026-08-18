"""The synthesis entrypoint's constants and its agreement scorer.

The generation loop needs a GPU, but the two things that decide whether the output is
usable do not: the target codes must be real NLLB tokens, and the scorer must be a real
similarity rather than something that always passes.
"""

from __future__ import annotations

import pytest
from modal_app.synth import BASE, NUM_BEAMS, TARGETS, THRESHOLD, by_length, chrf


class TestTargets:
    def test_arabic_comes_first(self) -> None:
        """Algeria's official language is the direction Kabyle speakers actually need,
        and it is the smallest real resource on disk — 3,360 pairs."""
        assert TARGETS[0] == "arb_Arab"

    def test_every_target_is_an_nllb_language_code(self) -> None:
        """A code the tokenizer does not hold becomes `unk` and the model then translates
        fluently into the wrong language."""
        assert all(len(code.split("_")) == 2 for code in TARGETS)
        assert len(set(TARGETS)) == len(TARGETS)

    def test_generation_uses_the_untrimmed_base(self) -> None:
        """The fine-tune's 52,209 rows hold no Arabic script, so it cannot be the teacher."""
        assert BASE == "facebook/nllb-200-distilled-1.3B"

    def test_bulk_generation_is_greedy(self) -> None:
        assert NUM_BEAMS == 1


class TestByLength:
    """Order restoration is the correctness risk: a batch is reordered for speed, so the
    index has to ride with the text or every pair is silently mismatched."""

    def test_every_item_survives_exactly_once(self) -> None:
        items = [(i, "x" * (10 - i)) for i in range(10)]
        seen = [pair for chunk in by_length(items, 3) for pair in chunk]
        assert sorted(seen) == sorted(items)

    def test_batches_are_sorted_shortest_first(self) -> None:
        items = [(0, "long text here"), (1, "hi"), (2, "medium one")]
        assert [i for chunk in by_length(items, 3) for i, _ in chunk] == [1, 2, 0]

    def test_a_batch_holds_near_equal_lengths(self) -> None:
        items = [(i, "x" * length) for i, length in enumerate([1, 100, 2, 99, 3])]
        first = next(iter(by_length(items, 3)))
        assert [len(text) for _, text in first] == [1, 2, 3]

    def test_the_last_batch_may_be_short(self) -> None:
        items = [(i, "x") for i in range(7)]
        assert [len(chunk) for chunk in by_length(items, 3)] == [3, 3, 1]

    def test_an_empty_input_yields_nothing(self) -> None:
        assert list(by_length([], 4)) == []

    def test_the_caller_s_list_is_not_reordered(self) -> None:
        items = [(0, "long text"), (1, "hi")]
        list(by_length(items, 2))
        assert items == [(0, "long text"), (1, "hi")]


class TestChrf:
    def test_identical_text_scores_one_hundred(self) -> None:
        assert chrf("مرحبا بالعالم", "مرحبا بالعالم") == pytest.approx(100.0)

    def test_unrelated_text_scores_far_below_the_threshold(self) -> None:
        assert chrf("the cat sat on the mat", "zzz qqq vvv") < THRESHOLD

    def test_a_near_paraphrase_scores_above_the_threshold(self) -> None:
        assert chrf("le chat est sur le tapis", "le chat est sur un tapis") > THRESHOLD

    def test_it_is_symmetric_enough_to_use_as_agreement(self) -> None:
        first, second = "el gato está en la alfombra", "el gato esta en la alfombra"
        assert chrf(first, second) == pytest.approx(chrf(second, first), abs=1.0)
