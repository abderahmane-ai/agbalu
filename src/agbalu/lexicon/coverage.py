"""Lexicon coverage of the corpus, per source.

The share of a source's tokens the lexicon can analyse. Token coverage weights by
frequency and so tracks function words; type coverage weights each distinct word once
and so tracks unknown vocabulary.

This ranks sources against each other. It is not a threshold and not a quality score:
coverage is also low for a source with genuinely unusual vocabulary.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.lexicon.analyser import Analyser
from agbalu.lexicon.models import LexiconError

DEFAULT_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")

_TOKEN: Final = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
"""Letters only, with hyphenated clitic groups kept whole for the analyser to split."""


@dataclass
class SourceCoverage:
    source: str
    lines: int = 0
    tokens: int = 0
    known_tokens: int = 0
    types: set[str] = field(default_factory=set)
    known_types: set[str] = field(default_factory=set)
    routes: Counter[str] = field(default_factory=Counter)

    @property
    def token_coverage(self) -> float:
        return self.known_tokens / self.tokens if self.tokens else 0.0

    @property
    def type_coverage(self) -> float:
        return len(self.known_types) / len(self.types) if self.types else 0.0


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text)


def scan(
    corpus: Path,
    analyser: Analyser,
    limit: int | None = None,
) -> dict[str, SourceCoverage]:
    """Coverage per corpus source. `limit` caps lines per source, not overall."""
    if not corpus.is_file():
        msg = f"corpus not found: {corpus}"
        raise LexiconError(msg)

    per_source: dict[str, SourceCoverage] = {}
    decided: dict[str, str | None] = {}

    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source = str(row.get("source", "?"))
            report = per_source.get(source)
            if report is None:
                report = SourceCoverage(source=source)
                per_source[source] = report
            if limit is not None and report.lines >= limit:
                continue
            report.lines += 1

            for token in tokenise(str(row["text"])):
                folded = token.casefold()
                report.tokens += 1
                report.types.add(folded)
                route = decided.get(folded, "")
                if route == "":
                    analyses = analyser.analyse(folded)
                    route = analyses[0].route if analyses else None
                    decided[folded] = route
                if route is not None:
                    report.known_tokens += 1
                    report.known_types.add(folded)
                    report.routes[route] += 1
    return per_source


def totals(per_source: dict[str, SourceCoverage]) -> SourceCoverage:
    combined = SourceCoverage(source="ALL")
    for report in per_source.values():
        combined.lines += report.lines
        combined.tokens += report.tokens
        combined.known_tokens += report.known_tokens
        combined.types |= report.types
        combined.known_types |= report.known_types
        combined.routes.update(report.routes)
    return combined


def unknown_types(
    corpus: Path, analyser: Analyser, top: int = 50, limit: int | None = None
) -> list[tuple[str, int]]:
    """The most frequent unanalysable word types."""
    counts: Counter[str] = Counter()
    per_source: dict[str, int] = defaultdict(int)
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            source = str(row.get("source", "?"))
            if limit is not None and per_source[source] >= limit:
                continue
            per_source[source] += 1
            for token in tokenise(str(row["text"])):
                counts[token.casefold()] += 1
    return [(form, n) for form, n in counts.most_common() if not analyser.knows(form)][:top]
