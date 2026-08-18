"""Build AƔBALU-Parallel v1 from everything landed in `data/raw/`.

raw artifact -> aligned pairs -> normalise Kabyle -> inspect -> decontaminate
             -> dedup -> JSONL per language pair

Both directions are first class. `fr-kab` is larger than `en-kab` in Tatoeba
(93,908 against 58,674, CLAUDE.md 2.1b), so treating French as an afterthought
would discard the bigger half of the human-authored data.

Defective pairs are **kept and labelled**, not dropped. The rate of mechanical
defects is the measurement this phase exists to produce, and a filtered file cannot
report the rate of what was filtered out of it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TextIO

from agbalu.acquire.manifest import Manifest
from agbalu.bench.flores import KABYLE, Sentence
from agbalu.extract.columns import sample
from agbalu.extract.pipeline import fingerprint, release_priority
from agbalu.extract.readers import READABLE, ReadError, read_records
from agbalu.normalise import Normaliser
from agbalu.parallel.columns import choose_pair
from agbalu.parallel.langid import ForeignLang
from agbalu.parallel.quality import inspect
from agbalu.parallel.readers import PairReadError, read_opus_zip, read_record_pairs
from agbalu.registry.models import Registry, Source

log: Final = logging.getLogger("agbalu.parallel")

OPUS_CODE_TO_ISO3: Final[dict[str, ForeignLang]] = {"en": "eng", "fr": "fra"}

SKIP_SOURCES: Final = frozenset(
    {
        # Benchmarks: ingesting them contaminates the evaluation (CLAUDE.md 2.7 r4).
        "hf.flores-plus-kab",
        "hf.sib200-kab",
        # Latin<->Tifinagh transliteration of one text, not a translation pair.
        "hf.abdelhaqueidali.kab-latn-tfng",
    }
)


@dataclass(frozen=True, slots=True)
class RawPair:
    """One aligned pair as read, before normalisation."""

    kab: str
    foreign: str
    language: ForeignLang


@dataclass
class SourceStats:
    source_id: str
    fields: list[str] = field(default_factory=list)
    languages: Counter[str] = field(default_factory=Counter)
    read: int = 0
    kept: int = 0
    duplicate: int = 0
    defective: int = 0
    hard_defective: int = 0
    contaminated: int = 0
    repaired: int = 0
    by_defect: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


class ParallelBuilder:
    """Accumulates every parallel source into one deduplicated stream per language."""

    def __init__(
        self, registry: Registry, root: Path, benchmark: list[Sentence] | None = None
    ) -> None:
        self.registry = registry
        self.root = root
        self.normaliser = Normaliser()
        self.seen: set[bytes] = set()
        self.stats: list[SourceStats] = []
        self.blocked = self._benchmark_index(benchmark or [])

    def _benchmark_index(self, sentences: list[Sentence]) -> set[bytes]:
        keys: set[bytes] = set()
        for item in sentences:
            keys.add(fingerprint(item.text))
            keys.add(fingerprint(self.normaliser.normalise(item.text)))
        return keys

    def _artifacts(self, source: Source) -> list[Path]:
        manifest = Manifest(self.root)
        paths = [
            self.root / e.source_id / e.path
            for e in manifest.entries()
            if e.source_id == source.id and e.target == "local"
        ]
        return [p for p in paths if p.suffix.lower() in READABLE and p.is_file()]

    def sources(self) -> Iterator[Source]:
        # Dedup keeps the first copy, so ordering decides which licence governs a
        # shared pair. Alphabetical order gave bible-uedin — permissive and
        # human-translated — zero pairs, every one of them claimed by a mined
        # non-commercial NLLB derivative that happens to sort earlier.
        for source in sorted(self.registry.sources, key=release_priority):
            if source.tier == "excluded" or source.modality != "parallel":
                continue
            if source.id in SKIP_SOURCES:
                continue
            yield source

    def _emit_pair(
        self, pair: RawPair, source: Source, stats: SourceStats, out: Mapping[str, TextIO]
    ) -> None:
        stats.read += 1
        if pair.language == "other":
            return
        normalised = self.normaliser.normalise(pair.kab)
        if normalised != pair.kab:
            stats.repaired += 1

        key = fingerprint(normalised + "\x00" + pair.foreign)
        if key in self.seen:
            stats.duplicate += 1
            return
        if fingerprint(normalised) in self.blocked:
            stats.contaminated += 1
            return
        self.seen.add(key)

        defects = inspect(normalised, pair.foreign, pair.language)
        if defects.defective:
            stats.defective += 1
            if defects.hard:
                stats.hard_defective += 1
            for kind in defects.kinds:
                stats.by_defect[kind] += 1

        stats.kept += 1
        stats.languages[pair.language] += 1
        out[pair.language].write(
            json.dumps(
                {
                    "kab": normalised,
                    pair.language: pair.foreign,
                    "source": source.id,
                    "licence": source.licence,
                    "redistribution": source.redistribution,
                    "defects": list(defects.kinds),
                    "length_ratio": round(defects.length_ratio, 3),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def _emit(self, source: Source, out: Mapping[str, TextIO]) -> SourceStats:
        stats = SourceStats(source_id=source.id)
        for path in self._artifacts(source):
            try:
                if path.suffix.lower() == ".zip":
                    for kab, foreign, code in read_opus_zip(path):
                        language = OPUS_CODE_TO_ISO3.get(code, "other")
                        self._emit_pair(RawPair(kab, foreign, language), source, stats, out)
                    continue
                head = sample(read_records(path))
                kab_field, foreign_field, language = choose_pair(head)
                if kab_field is None or foreign_field is None or language == "other":
                    continue
                label = f"{kab_field}->{foreign_field}:{language}"
                if label not in stats.fields:
                    stats.fields.append(label)
                for kab, foreign in read_record_pairs(path, kab_field, foreign_field):
                    self._emit_pair(RawPair(kab, foreign, language), source, stats, out)
            except (ReadError, PairReadError) as exc:
                stats.errors.append(str(exc))
        return stats

    def build(self, directory: Path) -> list[SourceStats]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            lang: directory / f"agbalu-parallel-v1.kab-{lang}.jsonl" for lang in ("eng", "fra")
        }
        # Moved into place only on success, as `extract` does: a build that dies over
        # 11.7M records would otherwise leave a truncated corpus that looks complete.
        partials = {lang: path.with_name(path.name + ".part") for lang, path in paths.items()}
        handles = {lang: path.open("w", encoding="utf-8") for lang, path in partials.items()}
        try:
            for source in self.sources():
                stats = self._emit(source, handles)
                self.stats.append(stats)
                if stats.read:
                    log.info(
                        "source=%s read=%d kept=%d dup=%d defective=%d contaminated=%d langs=%s",
                        stats.source_id,
                        stats.read,
                        stats.kept,
                        stats.duplicate,
                        stats.defective,
                        stats.contaminated,
                        dict(stats.languages),
                    )
        finally:
            for handle in handles.values():
                handle.close()
        for lang, partial in partials.items():
            partial.replace(paths[lang])
        return self.stats


@dataclass(frozen=True, slots=True)
class ParallelSummary:
    """Totals across every source, typed so callers need no casts."""

    normaliser_version: str
    benchmark: str
    sources: int
    read: int
    kept: int
    duplicate: int
    contaminated: int
    repaired: int
    defective: int
    defect_rate: float
    hard_defective: int
    hard_defect_rate: float
    by_language: dict[str, int]
    by_defect: dict[str, int]


def summary(stats: list[SourceStats]) -> ParallelSummary:
    kept = sum(s.kept for s in stats)
    defective = sum(s.defective for s in stats)
    hard = sum(s.hard_defective for s in stats)
    defects: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for item in stats:
        defects.update(item.by_defect)
        languages.update(item.languages)
    return ParallelSummary(
        normaliser_version=Normaliser().version,
        benchmark=KABYLE,
        sources=len([s for s in stats if s.kept]),
        read=sum(s.read for s in stats),
        kept=kept,
        duplicate=sum(s.duplicate for s in stats),
        contaminated=sum(s.contaminated for s in stats),
        repaired=sum(s.repaired for s in stats),
        defective=defective,
        defect_rate=round(defective / kept, 4) if kept else 0.0,
        hard_defective=hard,
        hard_defect_rate=round(hard / kept, 4) if kept else 0.0,
        by_language=dict(languages),
        by_defect=dict(defects.most_common()),
    )
