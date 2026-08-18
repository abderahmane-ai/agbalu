"""Whether a document translates the same source word the same way twice.

chrF++ on FLORES+ cannot see this. Every FLORES+ row is one sentence scored against one
reference, so a term rendered three ways across a novel costs nothing there and ruins the
novel. Reading *Fables of Bidpai* is what found it: "frogs" comes back as `iberdan` — roads
— then as `amqerqur` two sentences later and as `agru` after that. The model has the right
word and does not keep it.

The measurement is reference-free and document-internal, which is what makes it usable on
the documents already translated: it asks only whether a source term's target rendering is
stable *within one document*, never whether it is correct. A model that is consistently
wrong scores 1.0 here, and that is the intended division of labour — chrF++ says whether a
translation is right, this says whether it is the same one twice.

Alignment is by line and then by sentence, the same two splits `segment.plan` makes, so a
line whose translation kept its sentence count aligns and one that did not is skipped and
counted. No word alignment model is involved: a term's candidate renderings are scored by
Dice coefficient over segment co-occurrence, which is what early phrase tables were built
from and is enough to separate a rare noun's translation from the function words that
surround it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Container, Iterable

from agbalu.mt.segment import SENTENCE_END, split_keeping_space, unwrap

MIN_OCCURRENCES: Final = 3
"""Occurrences of a source term before its renderings are worth counting.

Two occurrences give a consistency of either 1.0 or 0.5 and nothing between, so the
statistic is dominated by noise below this."""

MIN_TERM_LENGTH: Final = 4
"""Shorter source tokens are overwhelmingly function words, whose renderings are clitics and
particles that no co-occurrence signal separates."""

MAX_DOCUMENT_FRACTION: Final = 0.15
"""A source term in more than this share of segments is treated as a function word.

Not sufficient on its own: `which` occurs in 4.2% of *The Art of War*'s segments and passed
this filter while having no single rendering to be consistent about. `common_words` is what
excludes those, from the training corpus's own frequencies rather than a hand-written list."""

MIN_DOMINANT: Final = 0.34
"""A term whose best candidate covers less of it than this has no identified rendering.

The distinction the first version missed: a low dominant share can mean the model was
inconsistent, or it can mean the co-occurrence signal found nothing. Only the first is a
finding, and scoring both as failures reported every document at 0.2."""

COMMON_WORDS: Final = 2000
"""How many of the source language's most frequent words to exclude."""

WORD: Final = re.compile(r"[^\W\d_]+", re.UNICODE)

VOWELS: Final = frozenset("aeiou")

SHORTEST_STATE_BEARING: Final = 4
"""Below this a word is a particle, and stripping a prefix from it invents a stem."""

MASCULINE_ANNEXED: Final = ("wa", "we", "wu", "u")
"""`amcic` → `umcic`, `axxam` → `wexxam`. Folded back onto `a-`."""

FEMININE_ANNEXED: Final = ("te",)
"""`tamɣart` → `temɣart`, and separately `t` before a consonant → `tmɣart`.

`ti-` and `tu-` are **not** here. `ti-` is the feminine *plural* — `timenɣiwin`, battles —
and folding it onto `ta-` produced `taimenɣiwin`, a word that exists in no language, which
this module then reported as the model's output."""


def tokens(text: str) -> list[str]:
    """Case-folded, accent-preserving word tokens. Kabyle's diacritics are letters."""
    return [unicodedata.normalize("NFC", match.group()).lower() for match in WORD.finditer(text)]


def free_state(word: str) -> str:
    """A Kabyle noun's free-state form, so two occurrences of one noun compare equal.

    `amcic`/`umcic` and `azrem`/`uzrem` are one noun in two states, and counting the states
    apart reports a document that translated *cat* perfectly as 60% consistent. Surface
    level and deliberately so — a counting aid, not the morphology the lexicon owns.
    """
    if word.startswith("a") or len(word) < SHORTEST_STATE_BEARING:
        return word
    for prefix in MASCULINE_ANNEXED:
        if word.startswith(prefix) and len(word) > len(prefix) + 2:
            return "a" + word[len(prefix) :]
    if word[0] != "t" or word[1] in VOWELS:
        return word if word[0] != "t" else _feminine(word)
    return "ta" + word[1:]


def _feminine(word: str) -> str:
    """`te-` is the annexed state; `ta-`, `ti-` and `tu-` are not."""
    return "ta" + word[2:] if word.startswith(FEMININE_ANNEXED) else word


def sentences(line: str) -> list[str]:
    """The sentence units `segment.plan` would make of one line."""
    return [piece.strip() for piece in split_keeping_space(line, SENTENCE_END) if piece.strip()]


@dataclass(frozen=True, slots=True)
class Alignment:
    """Segment pairs, and how much of the document could not be paired."""

    pairs: tuple[tuple[str, str], ...]
    skipped_lines: int
    total_lines: int

    @property
    def skip_rate(self) -> float:
        return self.skipped_lines / self.total_lines if self.total_lines else 0.0


def align(source: str, target: str) -> Alignment:
    """Pair source and target segments by line, then by sentence within the line.

    A line whose translation did not keep its sentence count is skipped rather than guessed
    at: a wrong pairing does not fail, it silently attributes one sentence's vocabulary to
    another, which is the defect this module exists to detect.
    """
    source_lines = [line for line in unwrap(source).splitlines() if line.strip()]
    target_lines = [line for line in target.splitlines() if line.strip()]
    pairs: list[tuple[str, str]] = []
    skipped = 0
    for left, right in zip(source_lines, target_lines, strict=False):
        source_units, target_units = sentences(left), sentences(right)
        if len(source_units) != len(target_units):
            skipped += 1
            continue
        pairs.extend(zip(source_units, target_units, strict=True))
    return Alignment(
        pairs=tuple(pairs),
        skipped_lines=skipped + abs(len(source_lines) - len(target_lines)),
        total_lines=max(len(source_lines), len(target_lines)),
    )


@dataclass(frozen=True, slots=True)
class Term:
    """One source term, and how stably it was rendered."""

    term: str
    occurrences: int
    renderings: tuple[tuple[str, int], ...]

    @property
    def dominant(self) -> int:
        return self.renderings[0][1] if self.renderings else 0

    @property
    def consistency(self) -> float:
        """Share of this term's segments carrying its most frequent rendering."""
        return self.dominant / self.occurrences if self.occurrences else 0.0


def _dice(shared: int, left: int, right: int) -> float:
    total = left + right
    return 2 * shared / total if total else 0.0


def renderings(
    alignment: Alignment,
    *,
    common: Container[str] = frozenset(),
    min_occurrences: int = MIN_OCCURRENCES,
    max_fraction: float = MAX_DOCUMENT_FRACTION,
    min_dominant: float = MIN_DOMINANT,
) -> list[Term]:
    """Each frequent source term's candidate renderings, most co-occurrent first.

    A candidate is scored by Dice over segments rather than by raw count, which is what
    keeps a target function word appearing in most segments from outranking the rare noun
    that is the actual translation.

    Two filters decide what is scored at all, and both were added because the first version
    reported *`which`*, *`when`* and *`make`* as the least stable terms in every document.
    `common` excludes the source language's own high-frequency words, which have no single
    lexical translation to be consistent about; `min_dominant` excludes terms whose best
    candidate is too weak to be a rendering, where a low score means the alignment found
    nothing rather than that the model was inconsistent.
    """
    pairs = alignment.pairs
    if not pairs:
        return []

    source_sets: defaultdict[str, set[int]] = defaultdict(set)
    target_sets: defaultdict[str, set[int]] = defaultdict(set)
    for index, (left, right) in enumerate(pairs):
        for token in set(tokens(left)):
            source_sets[token].add(index)
        for token in {free_state(token) for token in tokens(right)}:
            target_sets[token].add(index)

    ceiling = max(1, int(max_fraction * len(pairs)))
    found: list[Term] = []
    for term, where in source_sets.items():
        if len(term) < MIN_TERM_LENGTH or term in common:
            continue
        if not min_occurrences <= len(where) <= ceiling:
            continue
        scored = [
            (candidate, len(where & seen), _dice(len(where & seen), len(where), len(seen)))
            for candidate, seen in target_sets.items()
            if len(candidate) >= MIN_TERM_LENGTH and where & seen
        ]
        scored.sort(key=lambda row: (-row[2], -row[1], row[0]))
        if not scored or scored[0][1] / len(where) < min_dominant:
            continue
        found.append(
            Term(
                term=term,
                occurrences=len(where),
                renderings=tuple((candidate, count) for candidate, count, _ in scored[:4]),
            )
        )
    found.sort(key=lambda t: (t.consistency, -t.occurrences, t.term))
    return found


@dataclass(frozen=True, slots=True)
class Report:
    """One document's lexical stability."""

    document: str
    segments: int
    skip_rate: float
    terms: tuple[Term, ...]

    @property
    def consistency(self) -> float:
        """Occurrence-weighted mean, so a term seen ten times counts for ten."""
        seen = sum(term.occurrences for term in self.terms)
        if not seen:
            return 0.0
        return sum(term.dominant for term in self.terms) / seen

    def as_dict(self) -> dict[str, object]:
        return {
            "document": self.document,
            "segments": self.segments,
            "terms": len(self.terms),
            "skip_rate": round(self.skip_rate, 4),
            "consistency": round(self.consistency, 4),
            "least_stable": [
                {
                    "term": term.term,
                    "occurrences": term.occurrences,
                    "consistency": round(term.consistency, 3),
                    "renderings": [
                        {"form": form, "segments": count} for form, count in term.renderings
                    ],
                }
                for term in self.terms[:20]
            ],
        }


def common_words(texts: Iterable[str], keep: int = COMMON_WORDS) -> frozenset[str]:
    """The `keep` most frequent words of a language, from a corpus of it.

    Taken from the training corpus rather than a shipped stop list so it is the same
    distribution the model was fitted on, and so a third source language costs nothing.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokens(text))
    return frozenset(word for word, _ in counts.most_common(keep))


def measure(
    document: str, source: str, target: str, common: Container[str] = frozenset()
) -> Report:
    """Align a source and its translation, and score the stability of its vocabulary."""
    alignment = align(source, target)
    return Report(
        document=document,
        segments=len(alignment.pairs),
        skip_rate=alignment.skip_rate,
        terms=tuple(renderings(alignment, common=common)),
    )


__all__ = [
    "Alignment",
    "Report",
    "Term",
    "align",
    "common_words",
    "free_state",
    "measure",
    "renderings",
    "tokens",
]
