"""The OCR metrics, on inputs built to separate the cases they can confuse.

A diacritic score that counts glyphs rather than placing them reads as near-perfect on a
hypothesis that has moved every one of them, and a rate over an empty population reads as a
measurement. Both are asserted against here.
"""

from __future__ import annotations

import pytest

from agbalu.ocr.evaluate import (
    EvaluationError,
    align,
    compute_cer,
    compute_diacritic_f1,
    compute_exact_match,
    compute_wer,
)


class TestAlignment:
    def test_identical_strings_align_position_for_position(self) -> None:
        assert align("aɣbalu", "aɣbalu") == [(c, c) for c in "aɣbalu"]

    def test_a_deletion_carries_none_on_the_hypothesis_side(self) -> None:
        assert (("ɣ", None)) in align("aɣu", "au")

    def test_an_insertion_carries_none_on_the_reference_side(self) -> None:
        assert ((None, "ɣ")) in align("au", "aɣu")

    def test_alignment_length_is_at_least_the_longer_string(self) -> None:
        pairs = align("aɣbalu", "abl")
        assert len(pairs) >= len("aɣbalu")


class TestDiacriticF1:
    def test_the_right_glyphs_in_the_wrong_places_do_not_score_as_correct(self) -> None:
        """Both strings hold one `ḍ` and one `ḥ`, so a count-based score returns 1.0. They
        are transposed, so a positional one must not."""
        reference = ["aḍris n uḥric"]
        swapped = ["aḥris n uḍric"]

        scored = compute_diacritic_f1(swapped, reference)
        assert scored["total_gold_special_glyphs"] == 2
        assert scored["f1"] == pytest.approx(0.0)

    def test_an_exact_transcription_scores_one(self) -> None:
        reference = ["aḍris n uḥric ɣef tɛeṛṛamt"]
        assert compute_diacritic_f1(reference, reference)["f1"] == pytest.approx(1.0)

    def test_a_dropped_diacritic_is_recall_and_a_spurious_one_is_precision(self) -> None:
        dropped = compute_diacritic_f1(["adris"], ["aḍris"])
        assert dropped["recall"] == pytest.approx(0.0)
        assert dropped["total_gold_special_glyphs"] == 1

        spurious = compute_diacritic_f1(["aḍris"], ["adris"])
        assert spurious["precision"] == pytest.approx(0.0)
        assert spurious["total_gold_special_glyphs"] == 0

    def test_a_hypothesis_with_no_marked_glyphs_at_all_scores_zero_not_one(self) -> None:
        scored = compute_diacritic_f1(["azul"], ["aẓul"])
        assert scored["precision"] == pytest.approx(0.0)
        assert scored["f1"] == pytest.approx(0.0)

    def test_text_carrying_no_kabyle_glyphs_reports_no_support(self) -> None:
        scored = compute_diacritic_f1(["salut"], ["salut"])
        assert scored["total_gold_special_glyphs"] == 0
        assert scored["f1"] == pytest.approx(0.0)


class TestRates:
    def test_cer_divides_by_reference_characters(self) -> None:
        assert compute_cer(["abc"], ["abd"]) == pytest.approx(1 / 3)

    def test_wer_divides_by_reference_words(self) -> None:
        assert compute_wer(["azul a yemma"], ["azul a baba"]) == pytest.approx(1 / 3)

    def test_an_empty_reference_population_raises_rather_than_reporting_zero(self) -> None:
        """A denominator clamped to at least one lets an empty reference contribute a
        character, and a set of blank references then returns a plausible rate."""
        with pytest.raises(EvaluationError, match="reference characters"):
            compute_cer([""], [""])
        with pytest.raises(EvaluationError, match="reference words"):
            compute_wer([""], [""])

    def test_exact_match_compares_stripped_lines(self) -> None:
        assert compute_exact_match([" azul "], ["azul"]) == pytest.approx(1.0)
        assert compute_exact_match(["azul"], ["azul a yemma"]) == pytest.approx(0.0)

    def test_a_perfect_transcription_has_no_error(self) -> None:
        lines = ["aḍris n uḥric", "ɣef tɛeṛṛamt"]
        assert compute_cer(lines, lines) == pytest.approx(0.0)
        assert compute_wer(lines, lines) == pytest.approx(0.0)
