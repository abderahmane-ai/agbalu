"""Exact match and character error rate, for every task whose output is a string.

Script conversion and morphological inflection are scored identically — did the model
produce the reference string, and how far off was it when it did not. Written once here
because two copies of a metric drift, and a metric that drifts between two tables is
worse than no metric: the tables still line up.

Both rates are corpus totals, not means over sentences. A per-sentence mean CER weights
a three-character form the same as a forty-character one, which is not the quantity
anybody reports.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agbalu.speech.metrics import edit_distance


class ScoreError(Exception):
    """A scoring request that cannot produce a number."""


@dataclass(frozen=True, slots=True)
class StringScore:
    """Counts, and the two rates derived from them.

    The counts are kept because a rate cannot be pooled across splits without them.
    """

    sentences: int
    exact: int
    errors: int
    characters: int

    @property
    def exact_match(self) -> float:
        return self.exact / self.sentences

    @property
    def character_error_rate(self) -> float:
        return self.errors / self.characters if self.characters else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "sentences": self.sentences,
            "exact_match": round(self.exact_match, 6),
            "character_error_rate": round(self.character_error_rate, 6),
            "errors": self.errors,
            "characters": self.characters,
        }


def score_strings(references: Iterable[str], hypotheses: Iterable[str]) -> StringScore:
    """Exact match and CER over paired sequences of equal length."""
    sentences = exact = errors = characters = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        sentences += 1
        exact += reference == hypothesis
        errors += edit_distance(reference, hypothesis)
        characters += len(reference)
    if not sentences:
        message = "no strings to score"
        raise ScoreError(message)
    return StringScore(sentences=sentences, exact=exact, errors=errors, characters=characters)
