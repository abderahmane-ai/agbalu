"""Build the clean monolingual corpus from everything landed in `data/raw/`.

raw artifact -> records -> Kabyle column -> normalise -> filter -> dedup -> JSONL

Provenance rides with every line: a released sentence can always name the source
it came from and the licence that governs it (CLAUDE.md 2.7).
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TextIO

from agbalu.acquire.manifest import Manifest
from agbalu.extract.columns import choose_field, sample
from agbalu.extract.detect import LATIN_MIN, latin_ratio
from agbalu.extract.readers import READABLE, ReadError, read_records
from agbalu.normalise import Normaliser
from agbalu.registry.models import Registry, Source

log: Final = logging.getLogger("agbalu.extract")

MIN_CHARS: Final = 3
MAX_CHARS: Final = 2000
MIN_LETTER_RATIO: Final = 0.4
"""Below this a line is a table row, a URL list or markup rather than prose."""

SKIP_MODALITIES: Final = frozenset({"speech", "tool", "lexicon"})
"""`lexicon` is excluded because its records are headwords and inflected forms, not
sentences. Admitting them put 196,796 bare word types into the first build. They
are the Phase 6 layer, not text."""
SKIP_SOURCES: Final = frozenset(
    {
        # Benchmarks. Ingesting them contaminates every number we would report
        # against them (CLAUDE.md 2.7 rule 4).
        "hf.flores-plus-kab",
        "hf.sib200-kab",
    }
)


@dataclass
class SourceStats:
    """What one source contributed, and what was dropped on the way."""

    source_id: str
    fields: list[str] = field(default_factory=list)
    field_score: float = 0.0
    read: int = 0
    empty: int = 0
    too_short: int = 0
    too_long: int = 0
    not_prose: int = 0
    wrong_script: int = 0
    duplicate: int = 0
    kept: int = 0
    repaired: int = 0
    files: int = 0
    errors: list[str] = field(default_factory=list)


def _letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def fingerprint(text: str) -> bytes:
    """Key for exact-duplicate detection, invariant to case and punctuation.

    Punctuation becomes a word break rather than vanishing, so `fell-awen` matches
    `fell awen` without also matching `fellawen` to either. Kabyle's obligatory
    clitic hyphen makes that distinction common across sources.
    """
    folded = unicodedata.normalize("NFKD", text).casefold()
    letters = "".join(c if c.isalnum() else " " for c in folded)
    return hashlib.blake2b(" ".join(letters.split()).encode("utf-8"), digest_size=16).digest()


REDISTRIBUTION_RANK: Final[dict[str, int]] = {
    "permissive": 0,
    "share-alike": 1,
    "unclear": 2,
    "non-commercial": 3,
    "none": 4,
}
TIER_RANK: Final[dict[str, int]] = {"core": 0, "supplementary": 1, "reference": 2, "excluded": 3}


def release_priority(source: Source) -> tuple[int, int, str]:
    """Order in which sources claim a shared sentence.

    Dedup keeps the first copy, so this decides which source a sentence is credited
    to and which licence governs it. Ordering by redistribution first puts every
    sentence in the most permissive bucket a real source can justify, which is what
    makes a permissive-only cut a filter rather than a rebuild (CLAUDE.md 2.7 rule 2).
    Alphabetical order instead credited 949,643 sentences to non-commercial NLLB
    derivatives that permissive sources also carry.
    """
    return (
        REDISTRIBUTION_RANK.get(source.redistribution, 4),
        TIER_RANK.get(source.tier, 3),
        source.id,
    )


class CorpusBuilder:
    """Accumulates every source into one deduplicated JSONL stream."""

    def __init__(self, registry: Registry, root: Path) -> None:
        self.registry = registry
        self.root = root
        self.normaliser = Normaliser()
        self.seen: set[bytes] = set()
        self.stats: list[SourceStats] = []

    def _artifacts(self, source: Source) -> list[Path]:
        manifest = Manifest(self.root)
        paths = [
            self.root / e.source_id / e.path
            for e in manifest.entries()
            if e.source_id == source.id and e.target == "local"
        ]
        return [p for p in paths if p.suffix.lower() in READABLE and p.is_file()]

    def _emit(self, source: Source, out: TextIO) -> SourceStats:
        stats = SourceStats(source_id=source.id)
        artifacts = self._artifacts(source)
        stats.files = len(artifacts)

        for path in artifacts:
            try:
                head = sample(read_records(path))
            except ReadError as exc:
                stats.errors.append(str(exc))
                continue
            name, score = choose_field(head)
            if name is None:
                continue
            if name not in stats.fields:
                stats.fields.append(name)
            stats.field_score = score

            try:
                rows = read_records(path)
            except ReadError as exc:
                stats.errors.append(str(exc))
                continue
            for record in rows:
                raw = record.get(name)
                if raw is None:
                    continue
                stats.read += 1
                self._consume(raw, source, stats, out)

        return stats

    def _consume(self, raw: str, source: Source, stats: SourceStats, out: TextIO) -> None:
        text = self.normaliser.normalise(raw)
        if not text:
            stats.empty += 1
            return
        if len(text) < MIN_CHARS:
            stats.too_short += 1
            return
        if len(text) > MAX_CHARS:
            stats.too_long += 1
            return
        if _letter_ratio(text) < MIN_LETTER_RATIO:
            stats.not_prose += 1
            return
        if latin_ratio(text) < LATIN_MIN:
            stats.wrong_script += 1
            return

        key = fingerprint(text)
        if key in self.seen:
            stats.duplicate += 1
            return
        self.seen.add(key)

        if text != raw:
            stats.repaired += 1
        stats.kept += 1
        out.write(
            json.dumps(
                {
                    "text": text,
                    "source": source.id,
                    "licence": source.licence,
                    "redistribution": source.redistribution,
                },
                ensure_ascii=False,
            )
        )
        out.write("\n")

    def sources(self) -> Iterator[Source]:
        for source in sorted(self.registry.sources, key=release_priority):
            if source.tier == "excluded" or source.modality in SKIP_MODALITIES:
                continue
            if source.id in SKIP_SOURCES:
                continue
            yield source

    def build(self, destination: Path) -> list[SourceStats]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        with partial.open("w", encoding="utf-8") as out:
            for source in self.sources():
                stats = self._emit(source, out)
                self.stats.append(stats)
                if stats.read:
                    log.info(
                        "source=%s fields=%s score=%.2f read=%d kept=%d dup=%d repaired=%d",
                        stats.source_id,
                        ",".join(stats.fields),
                        stats.field_score,
                        stats.read,
                        stats.kept,
                        stats.duplicate,
                        stats.repaired,
                    )
        partial.replace(destination)
        return self.stats


def summary(stats: list[SourceStats]) -> dict[str, object]:
    return {
        "normaliser_version": Normaliser().version,
        "sources": len([s for s in stats if s.kept]),
        "read": sum(s.read for s in stats),
        "kept": sum(s.kept for s in stats),
        "duplicate": sum(s.duplicate for s in stats),
        "repaired": sum(s.repaired for s in stats),
        "empty": sum(s.empty for s in stats),
        "too_short": sum(s.too_short for s in stats),
        "too_long": sum(s.too_long for s in stats),
        "not_prose": sum(s.not_prose for s in stats),
        "wrong_script": sum(s.wrong_script for s in stats),
    }
