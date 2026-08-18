"""Score the lexicon against `UD_Kabyle-ADPT`, 1,930 hand-annotated sentences.

`pos.py`'s mapping is read off exemplars, which is evidence and not a measurement.
Agreement on the forms both resources know is the measurement.

The treebank is never merged into the lexicon.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.lexicon.analyser import Analyser
from agbalu.treebank import DEFAULT_ROOT, read_all

IGNORED_UPOS: Final[frozenset[str]] = frozenset({"PUNCT", "SYM", "X", "_"})
"""Scoring punctuation would inflate agreement with tokens no lexicon holds."""


@dataclass
class PosReport:
    """Agreement between the lexicon's part of speech and the treebank's."""

    tokens: int = 0
    scored: int = 0
    known: int = 0
    unambiguous: int = 0
    correct: int = 0
    lemma_scored: int = 0
    lemma_correct: int = 0
    by_gold: Counter[str] = field(default_factory=Counter)
    correct_by_gold: Counter[str] = field(default_factory=Counter)
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def coverage(self) -> float:
        """Share of scored tokens the lexicon recognises at all."""
        return self.known / self.scored if self.scored else 0.0

    @property
    def accuracy(self) -> float:
        """Agreement over tokens the lexicon tags unambiguously."""
        return self.correct / self.unambiguous if self.unambiguous else 0.0

    @property
    def lemma_agreement(self) -> float:
        """Share of recognised tokens whose gold lemma is among the readings."""
        return self.lemma_correct / self.lemma_scored if self.lemma_scored else 0.0


def treebank_tokens(root: Path = DEFAULT_ROOT) -> list[tuple[str, str, str]]:
    """`(form, lemma, upos)` over every split."""
    return [
        (word.form, word.lemma, word.upos) for sentence in read_all(root) for word in sentence.words
    ]


def score(analyser: Analyser, tokens: list[tuple[str, str, str]]) -> PosReport:
    report = PosReport()
    for form, gold_lemma, gold_upos in tokens:
        report.tokens += 1
        if gold_upos in IGNORED_UPOS:
            continue
        report.scored += 1
        report.by_gold[gold_upos] += 1

        analyses = analyser.analyse(form)
        if not analyses:
            continue
        report.known += 1

        predicted = analyser.upos(form)
        if predicted is not None:
            report.unambiguous += 1
            if predicted == gold_upos:
                report.correct += 1
                report.correct_by_gold[gold_upos] += 1
            else:
                report.confusions[(gold_upos, predicted)] += 1

        lemmas = analyser.lemmas(form)
        if lemmas:
            report.lemma_scored += 1
            if gold_lemma in lemmas:
                report.lemma_correct += 1
    return report
