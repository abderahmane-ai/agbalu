"""Morphological analysis of a Kabyle word form.

Four lookup routes, tried in order: exact, clitic-stripped, case-folded, annexed state.

An unknown form yields no analysis rather than a most-likely tag. `coverage.py` measures
what the lexicon genuinely does not know, and a back-off guess would report it complete.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from agbalu.lexicon.models import Entry
from agbalu.lexicon.pipeline import read_lexicon
from agbalu.lexicon.pos import pos_confidence
from agbalu.lexicon.state import free_candidates
from agbalu.normalise.rules import HYPHEN

Route = Literal["exact", "clitic", "casefold", "state", "none"]

CLITIC_HEAD_MIN: Final = 2
"""`d-yusa` opens with the predicative particle, not a lexeme."""


def _clitic_head(form: str) -> str | None:
    """The part before the first hyphen, when it is long enough to be a word."""
    head, sep, _ = form.partition(HYPHEN)
    if sep and len(head) >= CLITIC_HEAD_MIN:
        return head
    return None


@dataclass(frozen=True, slots=True)
class Analysis:
    """One reading of a form, and how the analyser reached it."""

    form: str
    lemma: str | None
    upos: str | None
    features: tuple[tuple[str, str], ...]
    route: Route
    source: str

    def feats(self) -> str:
        if not self.features:
            return "_"
        return "|".join(f"{k}={v}" for k, v in self.features)


@dataclass
class Analyser:
    """An in-memory index over a built lexicon."""

    index: dict[str, list[Entry]]

    @classmethod
    def from_lexicon(cls, path: Path) -> Analyser:
        index: dict[str, list[Entry]] = defaultdict(list)
        for entry in read_lexicon(path):
            index[entry.form].append(entry)
            folded = entry.form.casefold()
            if folded != entry.form:
                index[folded].append(entry)
        return cls(index=dict(index))

    @classmethod
    def from_entries(cls, entries: Iterable[Entry]) -> Analyser:
        index: dict[str, list[Entry]] = defaultdict(list)
        for entry in entries:
            index[entry.form].append(entry)
            folded = entry.form.casefold()
            if folded != entry.form:
                index[folded].append(entry)
        return cls(index=dict(index))

    def __len__(self) -> int:
        return len(self.index)

    def _readings(self, form: str, route: Route) -> list[Analysis]:
        return [
            Analysis(
                form=form,
                lemma=entry.lemma,
                upos=entry.upos,
                features=entry.features,
                route=route,
                source=entry.source,
            )
            for entry in self.index.get(form, ())
        ]

    def _probes(self, form: str) -> Iterator[tuple[str, Route]]:
        """Forms to look up, most direct first."""
        yield form, "exact"

        head = _clitic_head(form)
        if head:
            yield head, "clitic"

        folded = form.casefold()
        if folded != form:
            yield folded, "casefold"
            head = _clitic_head(folded)
            if head:
                yield head, "casefold"

        for candidate in free_candidates(folded):
            yield candidate, "state"

    def analyse(self, form: str) -> tuple[Analysis, ...]:
        """Every reading of `form`, from the first route that finds one.

        Routes are not merged: an exact match is a fact, a state or clitic match a
        hypothesis, and merging would let the hypothesis outvote the fact in `upos`.
        """
        if not form:
            return ()
        for probe, route in self._probes(form):
            readings = self._readings(probe, route)
            if readings:
                return tuple(readings)
        return ()

    def knows(self, form: str) -> bool:
        return bool(self.analyse(form))

    def lemmas(self, form: str) -> tuple[str, ...]:
        """Distinct lemmas across every reading, in order."""
        seen: dict[str, None] = {}
        for analysis in self.analyse(form):
            if analysis.lemma:
                seen.setdefault(analysis.lemma, None)
        return tuple(seen)

    def upos(self, form: str) -> str | None:
        """The part of speech, from the best-qualified source that offers one.

        Only the best `pos_confidence` tier votes. Ties within it return nothing rather
        than an arbitrary winner.
        """
        readings = [a for a in self.analyse(form) if a.upos]
        if not readings:
            return None
        best = min(pos_confidence(a.source) for a in readings)

        counts: dict[str, int] = defaultdict(int)
        for analysis in readings:
            if pos_confidence(analysis.source) == best:
                counts[analysis.upos or ""] += 1

        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]
