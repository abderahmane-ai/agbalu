"""Kabyle beside its own translations, each pair once (task 11.17).

Parallel text is worth more per token than monolingual text: at a fixed 5M-token budget,
multi-way parallel data scored 22.48 on low-resource MMMLU against 19.64 for unaligned
monolingual (arXiv 2505.14045). Each pair becomes one tagged document, Kabyle first on even
rows, so neither direction is the only one taught.

`mt-train` holds every pair twice, one row per direction, and the earlier corpus wrote both:
the aligned tokens were half duplicates. A pair is keyed on its two sides, so the mirror
row is recognised and dropped. Held-out records leave by `corpus.withheld`, and anything
overlapping an evaluation set by the shared decontaminator.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter as Tally
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TextIO

from agbalu.llm.corpus import LANGUAGE_TAG, Record, Source, TokenCounter, records, withheld
from agbalu.llm.decontaminate import Decontaminator
from agbalu.normalise import Normaliser

KABYLE_CODE: Final = LANGUAGE_TAG["kab"]
SEPARATOR: Final = "\n"
"""One newline between the two sides. The pair is one document, not two."""

TRANSLATIONS: Final = Path("data/raw/hf.finetranslations-kab/data/kab_Latn")
"""FineWeb-2's Kabyle documents with a full English translation beside each one."""


class MixtureError(Exception):
    """The aligned corpus cannot be built or does not account for its input."""


def aligned_document(kabyle: str, other: str, code: str, *, kabyle_first: bool) -> str:
    """One parallel pair as a single tagged document."""
    head = (KABYLE_CODE, kabyle) if kabyle_first else (code, other)
    tail = (code, other) if kabyle_first else (KABYLE_CODE, kabyle)
    return f"{head[0]}: {head[1]}{SEPARATOR}{tail[0]}: {tail[1]}"


def pair_key(record: Record) -> bytes:
    """The same pair from either direction's row gives the same key."""
    payload = f"{record.kabyle}\x00{record.code}\x00{record.other}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def translation_records(directory: Path = TRANSLATIONS) -> Iterator[Record]:
    """FineTranslations rows as aligned records, Kabyle original beside its English."""
    import pyarrow.parquet as pq

    files = sorted(directory.glob("*.parquet"))
    if not files:
        message = f"no FineTranslations parquet under {directory}"
        raise MixtureError(message)
    index = 0
    for path in files:
        table = pq.read_table(path, columns=["og_full_text", "translated_text"])
        for kabyle, english in zip(
            table["og_full_text"].to_pylist(), table["translated_text"].to_pylist(), strict=True
        ):
            if isinstance(kabyle, str) and isinstance(english, str) and kabyle and english:
                yield Record(kabyle=kabyle, index=index, other=english, code=LANGUAGE_TAG["eng"])
            index += 1


@dataclass(slots=True)
class SourceTally:
    name: str
    read: int = 0
    kept: int = 0
    tokens: int = 0
    rejected: Tally[str] = field(default_factory=Tally)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.name,
            "read": self.read,
            "kept": self.kept,
            "tokens": self.tokens,
            "rejected": dict(sorted(self.rejected.items())),
        }


@dataclass(slots=True)
class AlignedWriter:
    """The filters for one aligned record, writing survivors in token-counted batches."""

    decontaminator: Decontaminator
    counter: TokenCounter
    sink: TextIO
    rate: int
    batch: int = 2000
    normaliser: Normaliser = field(default_factory=Normaliser)
    seen: set[bytes] = field(default_factory=set)
    pending: list[tuple[str, SourceTally]] = field(default_factory=list)

    def offer(self, record: Record, tally: SourceTally) -> None:
        tally.read += 1
        if withheld(record, rate=self.rate):
            tally.rejected["held_out"] += 1
            return
        kabyle = self.normaliser.normalise(record.kabyle).strip()
        other = record.other.strip()
        if not kabyle or not other:
            tally.rejected["empty"] += 1
            return
        key = pair_key(Record(kabyle=kabyle, index=record.index, other=other, code=record.code))
        if key in self.seen:
            tally.rejected["duplicate_pair"] += 1
            return
        self.seen.add(key)
        if self.decontaminator.matches(kabyle) or self.decontaminator.matches(other):
            tally.rejected["contaminated"] += 1
            return
        text = aligned_document(kabyle, other, record.code, kabyle_first=record.index % 2 == 0)
        self.pending.append((text, tally))
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        counts = self.counter.count([text for text, _ in batch])
        if len(counts) != len(batch):
            message = f"{len(counts)} token counts for {len(batch)} documents"
            raise MixtureError(message)
        for (text, tally), tokens in zip(batch, counts, strict=True):
            tally.kept += 1
            tally.tokens += tokens
            self.sink.write(json.dumps({"text": text, "source": tally.name}, ensure_ascii=False))
            self.sink.write("\n")


def build(
    sources: Sequence[Source], translations: Iterator[Record] | None, writer: AlignedWriter
) -> list[SourceTally]:
    """Every aligned source, then the document-level translations when given."""
    tallies: list[SourceTally] = []
    for source in sources:
        if source.kind != "aligned":
            message = f"{source.name} is {source.kind}; this corpus holds aligned pairs only"
            raise MixtureError(message)
        tally = SourceTally(source.name)
        tallies.append(tally)
        for record in records(source):
            writer.offer(record, tally)
    if translations is not None:
        tally = SourceTally("finetranslations-kab")
        tallies.append(tally)
        for record in translations:
            writer.offer(record, tally)
    writer.flush()

    for tally in tallies:
        rejected = sum(tally.rejected.values())
        if tally.read != tally.kept + rejected:
            message = (
                f"{tally.name}: read {tally.read}, kept {tally.kept}, rejected {rejected}; "
                f"{tally.read - tally.kept - rejected} unaccounted"
            )
            raise MixtureError(message)
    return tallies
