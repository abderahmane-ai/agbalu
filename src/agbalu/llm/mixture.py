"""The continued-pretraining corpus: Kabyle beside its own translations (task 11.4).

Two findings decide the shape of this file.

Parallel text is worth more per token than monolingual text: at a fixed 5M-token budget,
multi-way parallel data scored 22.48 on low-resource MMMLU against 19.64 for unaligned
monolingual and 18.27 untrained (arXiv 2505.14045). And more of the *other* language
improves the target language rather than merely protecting it — 1:1 English:Arabic beat
1:9 on Arabic loss (arXiv 2407.12869). Our parallel corpus is 1:1 by construction, so one
dataset carries the Kabyle signal, the replay, and the alignment supervision at once.

The mixture is balanced **by tokens, not by rows**. A monolingual row averages ~33 base
tokens and an aligned pair ~55, so an equal number of rows would be a 1:1.8 token mixture
while claiming to be 1:1.

Direction alternates by row so the model sees Kabyle first half the time. A fixed order
teaches translation in one direction, which is not what continued pretraining is for.

Held-out records are skipped, by the same predicate that selects them in `holdout`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from agbalu.llm.corpus import (
    HOLDOUT_RATE,
    LANGUAGE_TAG,
    Kind,
    Record,
    Source,
    records,
    withheld,
)

KABYLE_CODE = LANGUAGE_TAG["kab"]
SEPARATOR = "\n"
"""One newline between the two sides. The pair is one document, not two."""


class MixtureError(Exception):
    """The mixture cannot be counted or written."""


class Counter(Protocol):
    """Token counting, injected so the mixture is testable without a tokenizer."""

    def count(self, texts: Sequence[str]) -> list[int]: ...


def aligned_document(kabyle: str, other: str, code: str, *, kabyle_first: bool) -> str:
    """One parallel pair as a single tagged document."""
    head = (KABYLE_CODE, kabyle) if kabyle_first else (code, other)
    tail = (code, other) if kabyle_first else (KABYLE_CODE, kabyle)
    return f"{head[0]}: {head[1]}{SEPARATOR}{tail[0]}: {tail[1]}"


def document(record: Record) -> str:
    """The training document one record contributes."""
    if not record.aligned:
        return record.kabyle
    return aligned_document(
        record.kabyle, record.other, record.code, kabyle_first=record.index % 2 == 0
    )


def documents(source: Source, *, rate: int = HOLDOUT_RATE) -> Iterator[tuple[str, bool]]:
    """Every document of one source, each with whether it is held out of training."""
    for record in records(source):
        yield document(record), withheld(record, rate=rate)


@dataclass(frozen=True, slots=True)
class Tally:
    """What one source contributed, and what was withheld from it."""

    name: str
    kind: Kind
    documents: int
    tokens: int
    held_out: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "documents": self.documents,
            "tokens": self.tokens,
            "held_out": self.held_out,
        }


@dataclass(frozen=True, slots=True)
class Mixture:
    """A built corpus and the ratio it actually achieved."""

    tallies: tuple[Tally, ...]
    epochs: int
    rate: int = HOLDOUT_RATE

    def tokens(self, kind: Kind) -> int:
        return sum(t.tokens for t in self.tallies if t.kind == kind)

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self.tallies)

    @property
    def held_out(self) -> int:
        return sum(t.held_out for t in self.tallies)

    @property
    def aligned_share(self) -> float:
        return 0.0 if not self.total_tokens else self.tokens("aligned") / self.total_tokens

    def as_dict(self) -> dict[str, object]:
        return {
            "sources": [t.as_dict() for t in self.tallies],
            "tokens_per_epoch": self.total_tokens,
            "epochs": self.epochs,
            "tokens_total": self.total_tokens * self.epochs,
            "kabyle_tokens": self.tokens("kabyle"),
            "aligned_tokens": self.tokens("aligned"),
            "aligned_share": round(self.aligned_share, 4),
            "held_out_documents": self.held_out,
            "held_out_rate": self.rate,
        }


def build(
    sources: Sequence[Source],
    counter: Counter,
    out: Path,
    *,
    epochs: int,
    rate: int = HOLDOUT_RATE,
    batch: int = 2000,
) -> Mixture:
    """Write every document that is not held out to `out`, counting tokens by kind.

    Documents are written in source order and shuffled by `modal_app.jugurtha.order` under
    `recipe.SHUFFLE_SEED`, not here: shuffling 5.2M rows in memory would cost more than the
    training step it feeds. Naming the shuffler is the point — while this said only "by the
    loader", no loader did it, and a run trained on one source at a time.
    """
    if epochs < 1:
        message = f"epochs must be positive, got {epochs}"
        raise MixtureError(message)
    if batch < 1:
        message = f"batch must be positive, got {batch}"
        raise MixtureError(message)

    out.parent.mkdir(parents=True, exist_ok=True)
    tallies: list[Tally] = []
    with out.open("w", encoding="utf-8") as sink:
        for source in sources:
            count = 0
            tokens = 0
            excluded = 0
            pending: list[str] = []
            for text, held in documents(source, rate=rate):
                if held:
                    excluded += 1
                    continue
                pending.append(text)
                if len(pending) >= batch:
                    written, counted = _flush(source, pending, counter, sink)
                    count += written
                    tokens += counted
                    pending.clear()
            written, counted = _flush(source, pending, counter, sink)
            tallies.append(
                Tally(source.name, source.kind, count + written, tokens + counted, excluded)
            )

    return Mixture(tallies=tuple(tallies), epochs=epochs, rate=rate)


def _flush(
    source: Source, pending: Sequence[str], counter: Counter, sink: TextIO
) -> tuple[int, int]:
    """Count and write one batch: the documents written and the tokens they hold."""
    if not pending:
        return 0, 0
    counts = counter.count(pending)
    if len(counts) != len(pending):
        message = f"{source.name}: {len(counts)} counts for {len(pending)} documents"
        raise MixtureError(message)
    for text in pending:
        # `kind` is what the replay ratio is measured over: the file is written source by
        # source, so a reader cannot recover it from position, and inferring it from the
        # language tags an aligned document happens to carry would make the format load
        # bearing in two places.
        sink.write(json.dumps({"text": text, "kind": source.kind}, ensure_ascii=False) + "\n")
    return len(pending), sum(counts)
