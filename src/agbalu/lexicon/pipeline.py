"""Build AƔBALU-Lexicon v1 from the lexical sources in `data/raw/`.

raw source -> reader -> normalise -> merge identical analyses -> provenance -> JSONL

Forms go through the Phase 2 normaliser: a lexicon spelled differently from the corpus
would report coverage failures that are its own. Duplicates are credited to the most
redistributable source via `release_priority`, as in Phase 3.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.extract.pipeline import release_priority
from agbalu.lexicon.models import Entry, Gloss, LexiconError
from agbalu.lexicon.readers import (
    read_amawal,
    read_hunspell,
    read_tafsut,
    read_toponyms,
    read_verb_forms,
    read_verb_lemmas,
    rebrand,
)
from agbalu.normalise import Normaliser
from agbalu.registry.models import Registry, Source

log: Final = logging.getLogger("agbalu.lexicon")

DEFAULT_OUT: Final = Path("data/processed/lexicon/agbalu-lexicon-v1.jsonl")

EXCLUDED: Final[frozenset[str]] = frozenset({"git.sferhah.digitized-dallet"})
"""Dallet is on disk and not ingested: its MIT file is an unedited template and the
1982 dictionary is still in copyright (Phase 1). Readable, not redistributable."""

READERS: Final[dict[str, tuple[str, str]]] = {
    "hf.boffire.hunspell-kab": ("file", "kab.dic"),
    "hf.boffire.kabyle-verbs": ("verbs", ""),
    "hf.boffire.kabyle-toponyms": ("dir", "data"),
    "hf.agurzil.tafsut-maths-lexicon": ("file", "tafsut_math_lexicon.jsonl"),
    "hf.abdelhaqueidali.amawal-net": ("file", "amawal_net.csv"),
}


@dataclass
class SourceStats:
    source_id: str
    read: int = 0
    kept: int = 0
    merged: int = 0
    repaired: int = 0
    empty: int = 0


@dataclass
class LexiconBuilder:
    registry: Registry
    root: Path
    normaliser: Normaliser = field(default_factory=Normaliser)
    stats: list[SourceStats] = field(default_factory=list)

    def sources(self) -> Iterator[Source]:
        ordered = sorted(
            (s for s in self.registry.sources if s.id in READERS and s.id not in EXCLUDED),
            key=release_priority,
        )
        yield from ordered

    def _entries(self, source: Source) -> Iterator[Entry]:
        kind, name = READERS[source.id]
        base = self.root / source.id
        if kind == "file":
            yield from self._file_reader(source.id, base / name)
        elif kind == "dir":
            yield from read_toponyms(base / name)
        elif kind == "verbs":
            yield from read_verb_lemmas(base / "conjugation-tables")
            yield from read_verb_forms(base / "lemmatizer")

    def _file_reader(self, source_id: str, path: Path) -> Iterator[Entry]:
        if source_id == "hf.boffire.hunspell-kab":
            yield from read_hunspell(path)
        elif source_id == "hf.agurzil.tafsut-maths-lexicon":
            yield from read_tafsut(path)
        elif source_id == "hf.abdelhaqueidali.amawal-net":
            yield from read_amawal(path)

    def build(self, destination: Path) -> list[SourceStats]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")

        seen: dict[tuple[str, str | None, str | None, tuple[tuple[str, str], ...]], int] = {}
        with partial.open("w", encoding="utf-8") as out:
            for source in self.sources():
                stats = SourceStats(source_id=source.id)
                for entry in self._entries(source):
                    stats.read += 1
                    normalised = self._normalise(entry)
                    if normalised is None:
                        stats.empty += 1
                        continue
                    if normalised.form != entry.form or normalised.lemma != entry.lemma:
                        stats.repaired += 1
                    key = (
                        normalised.form,
                        normalised.lemma,
                        normalised.upos,
                        normalised.features,
                    )
                    if key in seen:
                        stats.merged += 1
                        continue
                    seen[key] = 1
                    stamped = rebrand(
                        normalised,
                        source.id,
                        source.licence,
                        source.redistribution,
                    )
                    out.write(json.dumps(_payload(stamped), ensure_ascii=False) + "\n")
                    stats.kept += 1
                self.stats.append(stats)
                if stats.read:
                    log.info(
                        "source=%s read=%d kept=%d merged=%d repaired=%d",
                        stats.source_id,
                        stats.read,
                        stats.kept,
                        stats.merged,
                        stats.repaired,
                    )
        partial.replace(destination)
        if not seen:
            msg = "no lexical entries were produced; check that the sources are on disk"
            raise LexiconError(msg)
        return self.stats

    def _normalise(self, entry: Entry) -> Entry | None:
        form = self.normaliser.normalise(entry.form).strip()
        if not form:
            return None
        lemma = self.normaliser.normalise(entry.lemma).strip() if entry.lemma else None
        return Entry(
            form=form,
            lemma=lemma or None,
            upos=entry.upos,
            features=entry.features,
            glosses=entry.glosses,
            source=entry.source,
            licence=entry.licence,
            redistribution=entry.redistribution,
        )


def _payload(entry: Entry) -> dict[str, object]:
    payload: dict[str, object] = {
        "form": entry.form,
        "lemma": entry.lemma,
        "upos": entry.upos,
        "feats": entry.feats(),
        "source": entry.source,
        "licence": entry.licence,
        "redistribution": entry.redistribution,
    }
    if entry.glosses:
        payload["glosses"] = [{"lang": g.language, "text": g.text} for g in entry.glosses]
    return payload


def read_lexicon(path: Path) -> Iterator[Entry]:
    """Read back a built lexicon."""
    if not path.is_file():
        msg = f"lexicon not found: {path}; run `make lexicon`"
        raise LexiconError(msg)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            feats = str(row.get("feats") or "_")
            features = (
                ()
                if feats == "_"
                else tuple(
                    (k, v) for k, _, v in (part.partition("=") for part in feats.split("|")) if v
                )
            )
            yield Entry(
                form=str(row["form"]),
                lemma=row.get("lemma"),
                upos=row.get("upos"),
                features=features,
                glosses=tuple(
                    Gloss(str(g["lang"]), str(g["text"])) for g in row.get("glosses", [])
                ),
                source=str(row.get("source", "")),
                licence=str(row.get("licence", "")),
                redistribution=str(row.get("redistribution", "")),
            )


def summary(stats: list[SourceStats], path: Path) -> dict[str, object]:
    forms: set[str] = set()
    lemmas: set[str] = set()
    by_upos: Counter[str] = Counter()
    by_redistribution: Counter[str] = Counter()
    glossed = multiword = 0

    for entry in read_lexicon(path):
        forms.add(entry.form)
        if entry.lemma:
            lemmas.add(entry.lemma)
        by_upos[entry.upos or "unlabelled"] += 1
        by_redistribution[entry.redistribution] += 1
        if entry.glosses:
            glossed += 1
        if " " in entry.form:
            multiword += 1

    return {
        "normaliser_version": Normaliser().version,
        "sources": len([s for s in stats if s.kept]),
        "entries": sum(s.kept for s in stats),
        "distinct_forms": len(forms),
        "distinct_lemmas": len(lemmas),
        "multiword_forms": multiword,
        "glossed_entries": glossed,
        "read": sum(s.read for s in stats),
        "merged": sum(s.merged for s in stats),
        "repaired": sum(s.repaired for s in stats),
        "empty": sum(s.empty for s in stats),
        "by_upos": dict(by_upos.most_common()),
        "by_redistribution": dict(by_redistribution.most_common()),
        "by_source": {s.source_id: s.kept for s in stats if s.kept},
    }
