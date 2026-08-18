"""WER and CER under the fixed normalisation policy (task 7.3).

The class that matters is the last one: both sides go through `ctc_target`, so a
system that emits case and punctuation is not charged for information the audio does
not carry — and two systems are only comparable because the same reduction is applied
to both. A corpus rate is also not a mean of per-utterance rates, and the difference
is large enough on short utterances to change a ranking.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from agbalu.speech.metrics import MetricError, Score, cer, edit_distance, wer


def _full_matrix(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """The textbook O(nm)-space form, kept only as an oracle for the two-row one."""
    grid = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        grid[i][0] = i
    for j in range(len(hypothesis) + 1):
        grid[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            grid[i][j] = min(grid[i - 1][j] + 1, grid[i][j - 1] + 1, grid[i - 1][j - 1] + cost)
    return grid[-1][-1]


class TestEditDistance:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ([], [], 0),
            (["a"], [], 1),
            ([], ["a"], 1),
            (["a", "b"], ["a", "b"], 0),
            (["a", "b"], ["a", "c"], 1),
            (["a", "b", "c"], ["a", "c"], 1),
            (["a", "c"], ["a", "b", "c"], 1),
            (["a", "b", "c"], ["c", "b", "a"], 2),
        ],
    )
    def test_distances(self, left: list[str], right: list[str], expected: int) -> None:
        assert edit_distance(left, right) == expected

    def test_symmetric(self) -> None:
        left, right = ["azul", "fell", "awen"], ["azul", "awen"]
        assert edit_distance(left, right) == edit_distance(right, left)

    def test_over_characters(self) -> None:
        assert edit_distance("azul", "azuk") == 1

    def test_agrees_with_a_full_matrix_over_random_pairs(self) -> None:
        """The two-row form got the match case wrong once; this is what would catch it."""
        rng = random.Random(20260811)
        alphabet = "azuɣɛ-"
        for _ in range(300):
            left = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            right = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            assert edit_distance(left, right) == _full_matrix(left, right)


class TestWer:
    def test_perfect_transcription(self) -> None:
        assert wer(["azul fell-awen"], ["azul fell-awen"]).rate == 0.0

    def test_one_word_wrong_in_two(self) -> None:
        assert wer(["azul fell-awen"], ["azul tmurt"]).rate == 0.5

    def test_everything_wrong(self) -> None:
        assert wer(["azul"], ["tmurt"]).rate == 1.0

    def test_empty_hypothesis_is_a_full_deletion(self) -> None:
        assert wer(["azul fell-awen"], [""]).rate == 1.0

    def test_hallucinated_words_exceed_one(self) -> None:
        """A rate above 1.0 is correct, not a bug: insertions have no upper bound."""
        assert wer(["azul"], ["azul azul azul"]).rate == 2.0

    def test_counts_are_reported(self) -> None:
        score = wer(["azul fell-awen", "tmurt"], ["azul tmurt", "tmurt"])
        assert score == Score(errors=1, total=3, utterances=2)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="argument"):
            wer(["azul"], ["azul", "tmurt"])

    def test_nothing_to_score(self) -> None:
        with pytest.raises(MetricError, match="no utterances"):
            wer([], [])

    def test_a_reference_of_only_punctuation_has_no_rate(self) -> None:
        scored = wer(["..."], ["azul"])
        assert scored.total == 0
        with pytest.raises(MetricError, match="no reference units"):
            _ = scored.rate


class TestCer:
    def test_perfect_transcription(self) -> None:
        assert cer(["azul"], ["azul"]).rate == 0.0

    def test_one_character_in_four(self) -> None:
        assert cer(["azul"], ["azuk"]).rate == 0.25

    def test_counts_the_delimiter(self) -> None:
        assert cer(["a b"], ["a b"]).total == 3

    def test_is_lower_than_wer_for_a_near_miss(self) -> None:
        reference, hypothesis = ["aɣbalu n tmurt"], ["aɣbalu n tmurk"]
        assert cer(reference, hypothesis).rate < wer(reference, hypothesis).rate


class TestTheNormalisationPolicy:
    def test_case_is_not_charged(self) -> None:
        assert wer(["Azul Fell-Awen"], ["azul fell-awen"]).rate == 0.0

    def test_punctuation_is_not_charged(self) -> None:
        assert wer(["Azul, ay amdan!"], ["azul ay amdan"]).rate == 0.0

    def test_the_hyphen_is_charged(self) -> None:
        """It is a letter of the writing system, so dropping it is a real error."""
        assert wer(["yenna-yas"], ["yenna yas"]).rate > 0.0

    def test_a_homoglyph_in_the_hypothesis_is_charged(self) -> None:
        """`ctc_target` cannot represent Greek epsilon, so it splits the word in two."""
        assert wer(["aɛdawen"], ["aεdawen"]).rate == 2.0

    def test_a_kabyle_specific_letter_is_charged(self) -> None:
        assert cer(["aɣbalu"], ["aghbalu"]).rate > 0.0


class TestPooling:
    def test_the_corpus_rate_is_not_the_mean_of_utterance_rates(self) -> None:
        references = ["azul", "azul fell-awen ay amdan n tmurt"]
        hypotheses = ["tmurt", "azul fell-awen ay amdan n tmurt"]
        pooled = wer(references, hypotheses).rate
        per_utterance = (wer([references[0]], [hypotheses[0]]).rate + 0.0) / 2
        assert pooled == pytest.approx(1 / 7)
        assert per_utterance == pytest.approx(0.5)
        assert pooled != per_utterance
