"""Whole-document Kabyle for continued pretraining (task 11.16).

AƔBALU-Text v1 is a sentence corpus built for the encoder, and its 2,000-character cap
dropped every long web page: 6,467 of 15,100 HPLT documents, 3,367 of 11,541 FineWeb-2.
Continued pretraining needs those documents whole, so the document-level sources are read
again from `data/raw` and the sentence corpus contributes only the sources that never had
documents.

Sources are processed in redistribution order, permissive first, because deduplication
keeps the first copy and the first copy decides the licence the text is released under.
Every record read is either written out or written to the rejects file with its reason.
"""

from __future__ import annotations

import bz2
import json
import re
from collections import Counter as Tally
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol, TextIO

from agbalu.llm.corpus import TokenCounter, is_held_out
from agbalu.llm.decontaminate import Decontaminator
from agbalu.llm.dedupe import NearDuplicates, exact_key
from agbalu.llm.progress import Tracker, stage
from agbalu.normalise import Normaliser
from agbalu.registry import Registry

Format = Literal["parquet", "jsonl", "wikitext"]

KABYLE: Final = "kab_Latn"
SENTENCE_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
RAW_ROOT: Final = Path("data/raw")

MIN_DOCUMENT_CHARS: Final = 200
"""Below this a web record is a navigation fragment or a caption, not a document."""

LID_WINDOW_CHARS: Final = 256
"""Dihya's own `max_length`; a longer input is truncated by the identifier anyway."""

LID_WINDOWS: Final = 4
"""Windows spread evenly over a document. One window judges only the opening, which on a
web page is the most likely place for a menu in another language."""

REDISTRIBUTION_RANK: Final[dict[str, int]] = {
    "permissive": 0,
    "share-alike": 1,
    "non-commercial": 2,
    "unclear": 3,
}

WIKI_DROPPED_PREFIXES: Final = ("taggayt:", "category:", "file:", "image:", "afaylu:", "tugna:")
"""Category and media lines `strip_code` leaves behind, by their Kabyle and English names."""


class DocumentError(Exception):
    """A source cannot be read, or the build does not account for its input."""


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """One registered source whose records are whole documents."""

    source_id: str
    patterns: tuple[str, ...]
    format: Format
    fields: tuple[str, ...]
    """Joined by a blank line, in order, skipping empty values: a title, then its body."""


DOCUMENT_SOURCES: Final = (
    DocumentSource("hf.hplt2-cleaned-kab", ("kab_Latn/*.parquet",), "parquet", ("text",)),
    DocumentSource("hf.fineweb2-kab", ("data/kab_Latn/*/*.parquet",), "parquet", ("text",)),
    DocumentSource(
        "hf.finetranslations-kab", ("data/kab_Latn/*.parquet",), "parquet", ("og_full_text",)
    ),
    DocumentSource("hf.glotcc-v1-kab", ("v1.0/kab-Latn/*.parquet",), "parquet", ("content",)),
    DocumentSource("hf.finepdfs-kab", ("data/kab_Latn/*/*.parquet",), "parquet", ("text",)),
    DocumentSource("hf.dcad2000-kab", ("kab_Latn/*_keep.jsonl",), "jsonl", ("text",)),
    DocumentSource("hf.imsidag.kabyle-datasets-mono", ("*.jsonl",), "jsonl", ("title", "content")),
    DocumentSource("wikimedia.kabwiki", ("*-pages-articles.xml.bz2",), "wikitext", ("text",)),
)


@dataclass(frozen=True, slots=True)
class RawDocument:
    source_id: str
    key: str
    """Where the record is: `path:row`, or `path:title` for a wiki page, the path relative to
    the source directory. FineWeb-2 has `train/000_00000.parquet` and `test/000_00000.parquet`,
    so a bare file name collides."""

    text: str


class LanguageScorer(Protocol):
    """Class probabilities per text, one mapping per input, in input order."""

    def probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]: ...


def clean(text: str, normaliser: Normaliser) -> str:
    """Normalised text with uniform line endings and at most one blank line in a row."""
    unified = normaliser.normalise(text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip() for line in unified.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def joined(values: Iterable[object]) -> str:
    return "\n\n".join(str(v).strip() for v in values if isinstance(v, str) and v.strip())


def windows(text: str, size: int = LID_WINDOW_CHARS, count: int = LID_WINDOWS) -> list[str]:
    """Up to `count` slices of `size` characters spread evenly over `text`."""
    if size < 1 or count < 1:
        message = f"window size and count must be positive, got {size} and {count}"
        raise DocumentError(message)
    if len(text) <= size:
        return [text]
    slots = min(count, len(text) // size)
    span = len(text) - size
    starts = sorted({round(i * span / max(1, slots - 1)) for i in range(slots)})
    return [text[start : start + size] for start in starts]


def verdict(rows: Sequence[dict[str, float]]) -> str:
    """The label with the highest mean probability across a document's windows."""
    if not rows:
        message = "no window was scored"
        raise DocumentError(message)
    labels = rows[0].keys()
    return max(labels, key=lambda label: sum(row[label] for row in rows) / len(rows))


def wiki_plaintext(wikitext: str, title: str) -> str:
    """An article's prose: markup stripped, category and media lines dropped."""
    import mwparserfromhell

    stripped = mwparserfromhell.parse(wikitext).strip_code(normalize=True, collapse=True)
    lines = [
        line
        for line in stripped.split("\n")
        if not line.strip().casefold().startswith(WIKI_DROPPED_PREFIXES)
    ]
    return joined([title, "\n".join(lines)])


def _read_parquet(path: Path, fields: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    missing = [name for name in fields if name not in handle.schema_arrow.names]
    if missing:
        message = f"{path}: no column {missing}; has {handle.schema_arrow.names}"
        raise DocumentError(message)
    row = 0
    for batch in handle.iter_batches(columns=list(fields), batch_size=1024):
        columns = [batch.column(name).to_pylist() for name in fields]
        for values in zip(*columns, strict=True):
            yield str(row), joined(values)
            row += 1


def _read_jsonl(path: Path, fields: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                message = f"{path}:{number} is not JSON: {error.msg}"
                raise DocumentError(message) from error
            if not isinstance(record, dict):
                message = f"{path}:{number} is not a JSON object"
                raise DocumentError(message)
            yield str(number), joined(record.get(name) for name in fields)


def _read_wikitext(path: Path) -> Iterator[tuple[str, str]]:
    """Main-namespace articles of a MediaWiki XML dump, redirects excluded."""
    from defusedxml.ElementTree import iterparse

    with bz2.open(path, "rb") as stream:
        title, namespace, text, redirect = "", "", "", False
        for _, element in iterparse(stream, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = element.text or ""
            elif tag == "ns":
                namespace = element.text or ""
            elif tag == "redirect":
                redirect = True
            elif tag == "text":
                text = element.text or ""
            elif tag == "page":
                if namespace == "0" and not redirect:
                    yield title, wiki_plaintext(text, title)
                title, namespace, text, redirect = "", "", "", False
                element.clear()


def read_source(source: DocumentSource, root: Path = RAW_ROOT) -> Iterator[RawDocument]:
    files = sorted(
        {p for pattern in source.patterns for p in (root / source.source_id).glob(pattern)}
    )
    if not files:
        message = f"{source.source_id}: nothing matches {source.patterns} under {root}"
        raise DocumentError(message)
    for path in files:
        relative = path.relative_to(root / source.source_id)
        match source.format:
            case "parquet":
                records = _read_parquet(path, source.fields)
            case "jsonl":
                records = _read_jsonl(path, source.fields)
            case "wikitext":
                records = _read_wikitext(path)
        for locator, text in records:
            yield RawDocument(source.source_id, f"{relative}:{locator}", text)


@dataclass(frozen=True, slots=True)
class SentenceRow:
    text: str
    source_id: str
    licence: str
    redistribution: str
    line: int


def read_sentences(path: Path = SENTENCE_CORPUS) -> Iterator[SentenceRow]:
    if not path.is_file():
        message = f"{path} not found; run `make extract`"
        raise DocumentError(message)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            values = [row.get(k) for k in ("text", "source", "licence", "redistribution")]
            if not all(isinstance(v, str) for v in values):
                message = f"{path}:{number} lacks text, source, licence or redistribution"
                raise DocumentError(message)
            text, source_id, licence, redistribution = (str(v) for v in values)
            yield SentenceRow(text, source_id, licence, redistribution, number)


@dataclass(slots=True)
class SourceStats:
    source_id: str
    licence: str
    redistribution: str
    read: int = 0
    kept: int = 0
    characters: int = 0
    tokens: int = 0
    rejected: Tally[str] = field(default_factory=Tally)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source_id,
            "licence": self.licence,
            "redistribution": self.redistribution,
            "read": self.read,
            "kept": self.kept,
            "characters": self.characters,
            "tokens": self.tokens,
            "rejected": dict(sorted(self.rejected.items())),
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """A record that passed the cheap filters and waits for the language check."""

    text: str
    key: str
    stats: SourceStats
    is_document: bool


class Builder:
    """The filters in order, writing as it goes so the corpus is never held whole."""

    def __init__(
        self,
        scorer: LanguageScorer,
        decontaminator: Decontaminator,
        counter: TokenCounter,
        sink: TextIO,
        rejects: TextIO,
        *,
        batch: int = 256,
    ) -> None:
        self.scorer = scorer
        self.decontaminator = decontaminator
        self.counter = counter
        self.sink = sink
        self.rejects = rejects
        self.batch = batch
        self.normaliser = Normaliser()
        self.seen: set[bytes] = set()
        self.near = NearDuplicates()
        self.pending: list[Candidate] = []
        self.stats: dict[str, SourceStats] = {}

    def reject(self, stats: SourceStats, key: str, reason: str) -> None:
        stats.rejected[reason.split(":", 1)[0]] += 1
        self.rejects.write(
            json.dumps({"source": stats.source_id, "key": key, "reason": reason}) + "\n"
        )

    def offer(self, text: str, key: str, stats: SourceStats, *, is_document: bool) -> None:
        """Run the cheap filters now and queue survivors for the batched language check."""
        stats.read += 1
        if not text:
            self.reject(stats, key, "empty")
            return
        if is_document and len(text) < MIN_DOCUMENT_CHARS:
            self.reject(stats, key, "too_short")
            return
        if not is_document and is_held_out(text):
            self.reject(stats, key, "held_out")
            return
        digest = exact_key(text)
        if digest in self.seen:
            self.reject(stats, key, "duplicate")
            return
        self.seen.add(digest)
        if self.decontaminator.matches(text):
            self.reject(stats, key, "contaminated")
            return
        self.pending.append(Candidate(text, key, stats, is_document))
        if len(self.pending) >= self.batch:
            self.flush()

    def _near_duplicate(self, candidate: Candidate) -> bool:
        return self.near.seen(f"{candidate.stats.source_id}:{candidate.key}", candidate.text)

    def flush(self) -> None:
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        spans: list[tuple[int, int]] = []
        texts: list[str] = []
        for candidate in batch:
            pieces = windows(candidate.text)
            spans.append((len(texts), len(texts) + len(pieces)))
            texts.extend(pieces)
        rows = self.scorer.probabilities(texts)
        if len(rows) != len(texts):
            message = f"the scorer returned {len(rows)} rows for {len(texts)} windows"
            raise DocumentError(message)

        kept: list[Candidate] = []
        for candidate, (start, stop) in zip(batch, spans, strict=True):
            label = verdict(rows[start:stop])
            if label != KABYLE:
                self.reject(candidate.stats, candidate.key, f"language:{label}")
                continue
            if candidate.is_document and self._near_duplicate(candidate):
                self.reject(candidate.stats, candidate.key, "near_duplicate")
                continue
            kept.append(candidate)

        counts = self.counter.count([c.text for c in kept]) if kept else []
        for candidate, tokens in zip(kept, counts, strict=True):
            stats = candidate.stats
            stats.kept += 1
            stats.characters += len(candidate.text)
            stats.tokens += tokens
            row = {
                "text": candidate.text,
                "source": stats.source_id,
                "licence": stats.licence,
                "redistribution": stats.redistribution,
            }
            self.sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    def source_stats(self, source_id: str, licence: str, redistribution: str) -> SourceStats:
        if source_id not in self.stats:
            self.stats[source_id] = SourceStats(source_id, licence, redistribution)
        return self.stats[source_id]


def ordered(sources: Sequence[DocumentSource], registry: Registry) -> list[DocumentSource]:
    """Permissive sources first: the first copy of a duplicate is the one kept."""
    return sorted(
        sources,
        key=lambda s: (
            REDISTRIBUTION_RANK[registry.by_id(s.source_id).redistribution],
            s.source_id,
        ),
    )


def build(
    sources: Sequence[DocumentSource],
    sentences: Iterable[SentenceRow],
    builder: Builder,
    registry: Registry,
    *,
    raw_root: Path = RAW_ROOT,
) -> list[SourceStats]:
    """Every document source, then the sentence rows of sources that have no documents."""
    superseded = {source.source_id for source in sources}
    for source in ordered(sources, registry):
        entry = registry.by_id(source.source_id)
        stats = builder.source_stats(source.source_id, entry.licence, entry.redistribution)
        with stage("documents", source=source.source_id):
            tracker = Tracker(f"documents {source.source_id}", None)
            for document in read_source(source, raw_root):
                text = clean(document.text, builder.normaliser)
                builder.offer(text, document.key, stats, is_document=True)
                tracker.advance(kept=stats.kept)
            builder.flush()

    with stage("sentence rows"):
        tracker = Tracker("sentence rows", None)
        for row in sentences:
            stats = builder.source_stats(row.source_id, row.licence, row.redistribution)
            tracker.advance(source=row.source_id)
            if row.source_id in superseded:
                stats.read += 1
                builder.reject(stats, f"sentences:{row.line}", "superseded")
                continue
            builder.offer(row.text, f"sentences:{row.line}", stats, is_document=False)
        builder.flush()

    for stats in builder.stats.values():
        rejected = sum(stats.rejected.values())
        if stats.read != stats.kept + rejected:
            message = (
                f"{stats.source_id}: read {stats.read} but kept {stats.kept} "
                f"and rejected {rejected}; {stats.read - stats.kept - rejected} unaccounted"
            )
            raise DocumentError(message)
    return list(builder.stats.values())
