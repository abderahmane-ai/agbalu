"""Word and character error rate, under one fixed normalisation policy.

Both sides are reduced by `ctc_target` before they are compared. That is not a
convenience: a system that emits punctuation and case is otherwise penalised for
producing information the reference does not test and the audio does not carry, and
two systems reduced differently are not comparable at all. Task 7.3's requirement is
that the policy be fixed and stated, so it lives in one function and both the
hypothesis and the reference go through it.

Written here rather than taken from `jiwer` because the policy is the measurement:
the dependency would supply the edit distance and leave the decision implicit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from agbalu.speech.vocabulary import ctc_target


class MetricError(Exception):
    """A scoring request that cannot produce a number."""


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance, two rows rather than a full matrix.

    The full matrix over a 15,003-utterance test set is the difference between a
    scoring pass that fits in memory and one that does not.
    """
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for j, right in enumerate(hypothesis, start=1):
        current = [j]
        for i, left in enumerate(reference, start=1):
            current.append(
                previous[i - 1]
                if left == right
                else 1 + min(previous[i], previous[i - 1], current[i - 1])
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class Score:
    """An error rate and the counts it was computed from.

    The counts are kept because a rate without its denominator cannot be pooled, and
    a per-utterance mean is not the corpus rate the field reports.
    """

    errors: int
    total: int
    utterances: int

    @property
    def rate(self) -> float:
        if not self.total:
            message = "no reference units, so no error rate exists"
            raise MetricError(message)
        return self.errors / self.total

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rate": round(self.rate, 6),
            "percent": round(100 * self.rate, 4),
            "errors": self.errors,
            "total": self.total,
            "utterances": self.utterances,
        }


def _accumulate(references: Iterable[str], hypotheses: Iterable[str], *, by_word: bool) -> Score:
    errors = total = utterances = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        left = ctc_target(reference)
        right = ctc_target(hypothesis)
        units_left: Sequence[str] = left.split() if by_word else left
        units_right: Sequence[str] = right.split() if by_word else right
        errors += edit_distance(units_left, units_right)
        total += len(units_left)
        utterances += 1
    if not utterances:
        message = "no utterances to score"
        raise MetricError(message)
    return Score(errors=errors, total=total, utterances=utterances)


def wer(references: Iterable[str], hypotheses: Iterable[str]) -> Score:
    """Word error rate over the whole set: total edits over total reference words."""
    return _accumulate(references, hypotheses, by_word=True)


def cer(references: Iterable[str], hypotheses: Iterable[str]) -> Score:
    """Character error rate, counting the word delimiter as a character."""
    return _accumulate(references, hypotheses, by_word=False)
