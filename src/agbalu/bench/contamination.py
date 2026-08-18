"""Measure how much of a benchmark leaked into the training corpus.

Excluding the benchmark by source id is not decontamination. FLORES+ is built from
Wikipedia, Wikinews, Wikivoyage and Wikibooks, and the corpus holds all four
through its crawl and wiki sources, so benchmark sentences can enter under a
source id that is not the benchmark's.

Two tests:

- **exact** — `extract.fingerprint`, the same key the corpus deduplicates on, so a
  hit means the sentence would have been dropped as a duplicate had the benchmark
  been part of the build. Each benchmark sentence is indexed under both its raw and
  its normalised form: the corpus is normalised and FLORES+ `kab_Latn` is not, so
  matching raw-to-normalised misses every leaked sentence carrying the
  Greek-epsilon corruption, 16.3% of them.
- **n-gram** — a shared run of `NGRAM` tokens, which catches a benchmark sentence
  that was edited, re-cased or embedded in a longer passage.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.bench.flores import Sentence
from agbalu.extract import fingerprint
from agbalu.normalise import Normaliser

NGRAM: Final = 8
"""Tokens in a contamination n-gram.

Long enough that natural coincidence is negligible in a 35M-word corpus, short
enough to catch a benchmark sentence that was lightly edited. FLORES+ Kabyle
sentences average 21.5 tokens, so a leaked sentence offers many chances to hit;
the 16 sentences shorter than 8 tokens are covered by the exact test only.
"""


@dataclass(frozen=True, slots=True)
class Leak:
    """One benchmark sentence found in the corpus."""

    id: int
    split: str
    source: str
    kind: str
    text: str
    corpus_text: str


@dataclass
class ContaminationReport:
    corpus_lines: int = 0
    benchmark_sentences: int = 0
    leaked: list[Leak] = field(default_factory=list)
    by_source: Counter[str] = field(default_factory=Counter)
    by_split: Counter[str] = field(default_factory=Counter)
    by_kind: Counter[str] = field(default_factory=Counter)

    @property
    def leak_rate(self) -> float:
        if not self.benchmark_sentences:
            return 0.0
        return len(self.leaked) / self.benchmark_sentences


def _tokens(text: str) -> list[str]:
    return text.casefold().split()


def ngrams(text: str, size: int = NGRAM) -> set[tuple[str, ...]]:
    tokens = _tokens(text)
    if len(tokens) < size:
        return set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def build_index(
    sentences: list[Sentence], normaliser: Normaliser | None = None
) -> tuple[dict[bytes, Sentence], dict[tuple[str, ...], Sentence]]:
    """Exact and n-gram indexes over the benchmark, in both raw and normalised form."""
    engine = normaliser if normaliser is not None else Normaliser()
    exact: dict[bytes, Sentence] = {}
    grams: dict[tuple[str, ...], Sentence] = {}
    for sentence in sentences:
        for variant in {sentence.text, engine.normalise(sentence.text)}:
            exact.setdefault(fingerprint(variant), sentence)
            for gram in ngrams(variant):
                grams.setdefault(gram, sentence)
    return exact, grams


def scan(
    corpus: Path, sentences: list[Sentence], normaliser: Normaliser | None = None
) -> ContaminationReport:
    exact, grams = build_index(sentences, normaliser)
    report = ContaminationReport(benchmark_sentences=len(sentences))
    seen: set[tuple[str, int]] = set()

    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            report.corpus_lines += 1
            row = json.loads(stripped)
            text = str(row["text"])

            hit = exact.get(fingerprint(text))
            kind = "exact"
            if hit is None:
                for gram in ngrams(text):
                    hit = grams.get(gram)
                    if hit is not None:
                        kind = "ngram"
                        break
            if hit is None:
                continue
            # FLORES+ ids restart at 0 in every split; `id` alone is not unique.
            key = (hit.split, hit.id)
            if key in seen:
                continue

            seen.add(key)
            source = str(row.get("source", "?"))
            report.leaked.append(
                Leak(
                    id=hit.id,
                    split=hit.split,
                    source=source,
                    kind=kind,
                    text=hit.text,
                    corpus_text=text,
                )
            )
            report.by_source[source] += 1
            report.by_split[hit.split] += 1
            report.by_kind[kind] += 1

    return report
