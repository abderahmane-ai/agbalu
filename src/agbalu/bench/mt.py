"""MT scoring under the two-condition protocol of `docs/benchmark.md` §2.

Every result is scored twice — as published, and with both sides normalised — and the
difference reported as the orthography gap. FLORES+ `kab_Latn` is 16.2% corrupt
(task 7.0), so a system spelling `ɛ` U+025B correctly is scored against a reference
writing Greek `ε` on a sixth of the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final, Literal

from sacrebleu.metrics.base import Metric as SacreMetric
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from agbalu.bench.flores import ARABIC, ENGLISH, FRENCH, GERMAN, KABYLE, SPANISH, Sentence, Split
from agbalu.normalise import Normaliser

Direction = Literal["kab-eng", "eng-kab", "kab-fra", "fra-kab", "arb-kab", "spa-kab", "deu-kab"]
DIRECTIONS: Final[tuple[Direction, ...]] = (
    "kab-eng",
    "eng-kab",
    "kab-fra",
    "fra-kab",
    "arb-kab",
    "spa-kab",
    "deu-kab",
)
"""Arabic, Spanish and German are into Kabyle only. The reverse would have to be trained on
NLLB's own Kabyle output, which caps it at the teacher; it is served by pivoting through
`kab-eng` at inference instead."""

LANGUAGE_CODE: Final[dict[str, str]] = {
    "kab": KABYLE,
    "eng": ENGLISH,
    "fra": FRENCH,
    "arb": ARABIC,
    "spa": SPANISH,
    "deu": GERMAN,
}

CHRF_WORD_ORDER: Final = 2
"""chrF++ is chrF with word n-grams to order 2 (Popović 2017); `0` is a different
metric under a different name."""


class ScoringError(Exception):
    """Hypotheses and references could not be scored against each other."""


def as_direction(name: str) -> Direction:
    """Narrow a name off a command line to the `Direction` the harness is typed on.

    Membership in `DIRECTIONS` is what makes the narrowing sound, so the check and the type
    cannot disagree — which an assignment with an ignore comment would permit.
    """
    for known in DIRECTIONS:
        if known == name:
            return known
    message = f"unknown direction {name!r}; have {list(DIRECTIONS)}"
    raise ScoringError(message)


def target_language(direction: Direction) -> str:
    """The side being generated."""
    return direction.split("-")[1]


def source_language(direction: Direction) -> str:
    return direction.split("-")[0]


@dataclass(frozen=True, slots=True)
class Metric:
    """A metric value and the signature that makes it comparable."""

    name: str
    score: float
    signature: str


@dataclass(frozen=True, slots=True)
class Condition:
    """Every metric under one scoring condition."""

    name: Literal["raw", "normalised"]
    metrics: tuple[Metric, ...]

    def get(self, metric: str) -> Metric:
        for item in self.metrics:
            if item.name == metric:
                return item
        msg = f"no metric named {metric!r} in condition {self.name!r}"
        raise ScoringError(msg)


@dataclass(frozen=True, slots=True)
class Result:
    """An MT result carrying the mechanically checkable half of `docs/benchmark.md` §4."""

    direction: Direction
    split: Split
    sentences: int
    normaliser_version: str
    raw: Condition
    normalised: Condition

    def gap(self, metric: str) -> float:
        """normalised - raw. Positive means the reference penalises correct spelling."""
        return self.normalised.get(metric).score - self.raw.get(metric).score


@cache
def _suite() -> tuple[tuple[str, SacreMetric], ...]:
    """Built once per process: repeatedly constructing the `flores200` sentencepiece
    tokenizer segfaults the interpreter. The objects are stateless across scores."""
    return (
        ("chrf++", CHRF(word_order=CHRF_WORD_ORDER)),
        ("bleu", BLEU()),
        # spBLEU is BLEU over the FLORES-200 sentencepiece tokenizer; it is the number
        # the FLORES-200/NLLB literature reports, and is not comparable to `tok:13a`.
        ("spbleu", BLEU(tokenize="flores200")),
    )


def _metrics(hypotheses: list[str], references: list[str]) -> tuple[Metric, ...]:
    refs = [references]
    return tuple(
        Metric(
            name=name,
            score=round(metric.corpus_score(hypotheses, refs).score, 4),
            signature=str(metric.get_signature()),
        )
        for name, metric in _suite()
    )


def score(
    hypotheses: list[str],
    references: list[str],
    direction: Direction,
    split: Split,
    normaliser: Normaliser | None = None,
) -> Result:
    """Score one system output under both conditions.

    A length mismatch raises: silently scoring the overlap yields a plausible number
    from misaligned data.
    """
    if len(hypotheses) != len(references):
        msg = f"{len(hypotheses)} hypotheses against {len(references)} references"
        raise ScoringError(msg)
    if not hypotheses:
        msg = "nothing to score"
        raise ScoringError(msg)

    engine = normaliser if normaliser is not None else Normaliser()
    target = target_language(direction)

    # Kabyle rules over an English or French target is the `Chişinău` -> `Chiṣinău`
    # defect. Into-foreign directions therefore have an empty gap by construction.
    if target == "kab":
        normalised_hyps = [engine.normalise(h) for h in hypotheses]
        normalised_refs = [engine.normalise(r) for r in references]
    else:
        normalised_hyps, normalised_refs = hypotheses, references

    return Result(
        direction=direction,
        split=split,
        sentences=len(hypotheses),
        normaliser_version=engine.version,
        raw=Condition(name="raw", metrics=_metrics(hypotheses, references)),
        normalised=Condition(name="normalised", metrics=_metrics(normalised_hyps, normalised_refs)),
    )


def references_for(sentences: list[Sentence]) -> list[str]:
    """Reference strings in `(split, id)` order.

    FLORES+ ids restart at 0 per split, so `id` alone interleaves dev and devtest.
    """
    return [s.text for s in sorted(sentences, key=lambda s: (s.split, s.id))]
