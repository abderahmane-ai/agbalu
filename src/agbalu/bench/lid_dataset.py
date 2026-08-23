"""Build the Berber language-identification training set (task 7.4, model side).

`lid.py` cuts the held-out set; this module cuts train and dev from the same source
table, minus every held-out line.

Three decisions here are measurements, not preferences:

- **Every source is capped.** Uncapped, one source owned each class — Tarifit was
  98.9% one OCR New Testament — and removing 79,454 of 80,360 Tarifit rows moved
  held-out recall by 1.2 points. A source past the cap buys imbalance, not signal.
- **The split is stratified by `(language, domain)`, not by language.** That group is
  what the sampler balances, what the contrastive loss mines negatives along, and
  what the checkpoint is selected on. A dev split drawn per language reports the
  biggest domain and calls it the language.
- **FLORES+ is not read at all.** It is where the Tamasheq evaluation sentences come
  from, and it was 78.7% of Tamasheq training rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from agbalu.bench.lid import LidError, read_evalset
from agbalu.bench.lid_sources import CLASSES, SOURCES, Source
from agbalu.bench.lid_sources import lines as source_lines
from agbalu.registry.models import redistribution_class

log: Final = logging.getLogger("agbalu.bench.lid_dataset")

DEFAULT_RAW: Final = Path("data/raw")
DEFAULT_EVAL: Final = Path("data/processed/bench/lid-eval.jsonl")
DEFAULT_OUT: Final = Path("data/processed/bench/lid-dataset")
DEFAULT_SEED: Final = 20260823
DEFAULT_DEV_FRACTION: Final = 0.05
DEFAULT_CAP: Final = 8_000
"""Lines kept per source. At 8,000 no source owns a class and every class that has
more than one domain keeps all of them."""

MIN_GROUP_ROWS: Final = 12
"""A `(language, domain)` cell below this is folded into the language's largest cell
rather than becoming a group of its own. Nine lines is not a domain, and a sampler
that draws a whole batch slot from nine lines trains on those nine lines."""


@dataclass(frozen=True, slots=True)
class Record:
    """One labelled line, carrying the provenance a subset is cut on."""

    text: str
    language: str
    domain: str
    source: str
    label_trust: str
    licence: str
    redistribution: str
    group: str = ""
    """The cell this row trains in. Its own `(language, domain)` unless that cell was too
    small to be one, in which case `assign_groups` folded it — and the fold has to be
    written, not only reported, or the trainer draws a batch slot from the nine rows the
    fold exists to absorb."""

    @property
    def cell(self) -> str:
        return f"{self.language}/{self.domain}"

    def to_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "language": self.language,
            "domain": self.domain,
            "group": self.group or self.cell,
            "source": self.source,
            "label_trust": self.label_trust,
            "licence": self.licence,
            "redistribution": self.redistribution,
        }


def normalise_key(text: str) -> str:
    """Dedup and decontamination key: whitespace-folded and case-folded only."""
    return " ".join(text.split()).casefold()


def build_records(
    raw: Path, sources: Sequence[Source] = SOURCES, *, cap: int = DEFAULT_CAP, seed: int = 0
) -> list[Record]:
    """Every usable line from every training source, labelled, at most `cap` per source."""
    records: list[Record] = []
    for source in sources:
        if source.language not in CLASSES:
            msg = f"{source.id}: label {source.language!r} is outside the label set"
            raise LidError(msg)
        redistribution = (
            "unclear" if source.licence == "various" else redistribution_class(source.licence)
        )
        kept = [
            Record(
                text=line,
                language=source.language,
                domain=source.domain,
                source=source.id,
                label_trust=source.trust,
                licence=source.licence,
                redistribution=redistribution,
            )
            for line in source_lines(source, raw, limit=cap, seed=seed)
        ]
        records.extend(kept)
        log.info("read source=%s language=%s lines=%d", source.id, source.language, len(kept))
    return records


@dataclass(frozen=True, slots=True)
class DedupStats:
    """What deduplication removed, so the loss is auditable rather than silent."""

    kept: int
    cross_language_dropped: int
    duplicate_dropped: int


def deduplicate(records: Sequence[Record]) -> tuple[list[Record], DedupStats]:
    """Collapse repeats within a language; drop lines claimed by two languages."""
    by_key: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_key[normalise_key(record.text)].append(record)

    kept: list[Record] = []
    cross_language = 0
    duplicate = 0
    for group in by_key.values():
        if len({record.language for record in group}) > 1:
            cross_language += len(group)
            continue
        kept.append(min(group, key=lambda record: (record.label_trust != "gold", record.source)))
        duplicate += len(group) - 1
    kept.sort(key=lambda record: (record.language, record.source, record.text))
    return kept, DedupStats(len(kept), cross_language, duplicate)


def decontaminate(records: Sequence[Record], eval_path: Path) -> tuple[list[Record], int]:
    """Drop any training line that is a held-out line of the same language."""
    keys = {(example.language, normalise_key(example.text)) for example in read_evalset(eval_path)}
    kept = [r for r in records if (r.language, normalise_key(r.text)) not in keys]
    return kept, len(records) - len(kept)


def assign_groups(records: Sequence[Record]) -> dict[str, str]:
    """Map each record's own `(language, domain)` cell to the group it trains in.

    A cell under `MIN_GROUP_ROWS` is folded into its language's largest cell.
    Tamazight's Pontoon slice is nine lines once the placeholders and the Tifinagh rows
    fall out; left as a group, one slot of every batch would be those nine lines.
    """
    per_cell: Counter[str] = Counter(record.cell for record in records)
    largest: dict[str, str] = {}
    for cell, _ in per_cell.most_common():
        largest.setdefault(cell.split("/", maxsplit=1)[0], cell)
    return {
        cell: cell if count >= MIN_GROUP_ROWS else largest[cell.split("/", maxsplit=1)[0]]
        for cell, count in per_cell.items()
    }


def grouped(records: Sequence[Record]) -> list[Record]:
    """Stamp every row with the group it trains in, folding the cells too small to be one."""
    assignment = assign_groups(records)
    return [replace(record, group=assignment[record.cell]) for record in records]


def split(
    records: Sequence[Record], *, dev_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    """Per-group seeded split into train and dev, disjoint by construction.

    Per group rather than per language: dev is what the checkpoint is selected on,
    and a dev split drawn per language is 98.9% scripture for Tarifit, so it scores
    a scripture model and calls it Tarifit.
    """
    if not 0.0 < dev_fraction < 1.0:
        msg = f"dev_fraction must be in (0, 1), got {dev_fraction}"
        raise LidError(msg)
    by_group: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_group[record.group or record.cell].append(record)

    train: list[Record] = []
    dev: list[Record] = []
    for group in sorted(by_group):
        pool = by_group[group]
        random.Random(f"{seed}:split:{group}").shuffle(pool)
        held = max(1, round(len(pool) * dev_fraction)) if len(pool) > 1 else 0
        dev.extend(pool[:held])
        train.extend(pool[held:])
    return train, dev


def _counts(records: Sequence[Record], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter(getattr(record, key) for record in records)
    return dict(counter.most_common())


def statistics(
    train: Sequence[Record],
    dev: Sequence[Record],
    *,
    eval_path: Path,
    dedup: DedupStats,
    decontaminated: int,
) -> dict[str, object]:
    """Every reported number, measured off the records that will be written."""
    held_out = read_evalset(eval_path)
    train_by_language = _counts(train, "language")
    dev_by_language = _counts(dev, "language")
    per_language = {
        language: {
            "train": train_by_language.get(language, 0),
            "dev": dev_by_language.get(language, 0),
            "eval": sum(1 for example in held_out if example.language == language),
            "sources": len({r.source for r in train if r.language == language}),
            "domains": len({r.domain for r in train if r.language == language}),
        }
        for language in CLASSES
    }
    counts = [per_language[language]["train"] for language in CLASSES]
    smallest = min(counts)
    group_counts = dict(Counter(record.group or record.cell for record in train).most_common())
    return {
        "languages": list(CLASSES),
        "train_total": len(train),
        "dev_total": len(dev),
        "eval_total": len(held_out),
        "per_language": per_language,
        "train_imbalance_ratio": round(max(counts) / smallest, 2) if smallest else None,
        "train_groups": len(group_counts),
        "train_by_group": group_counts,
        "train_by_domain": _counts(train, "domain"),
        "train_by_label_trust": _counts(train, "label_trust"),
        "train_by_redistribution": _counts(train, "redistribution"),
        "train_by_source": _counts(train, "source"),
        "train_tokens": sum(len(record.text.split()) for record in train),
        "deduplication": {
            "kept": dedup.kept,
            "cross_language_dropped": dedup.cross_language_dropped,
            "duplicate_dropped": dedup.duplicate_dropped,
        },
        "decontaminated_against_eval": decontaminated,
    }


def write_jsonl(records: Sequence[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".part")
    with staging.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    staging.replace(path)


@dataclass(frozen=True, slots=True)
class BuildResult:
    train: list[Record]
    dev: list[Record]
    stats: dict[str, object]


def build_dataset(
    raw: Path,
    eval_path: Path,
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = DEFAULT_SEED,
    cap: int = DEFAULT_CAP,
) -> BuildResult:
    """Read capped per source, deduplicate, decontaminate, group, then split."""
    records, dedup = deduplicate(build_records(raw, cap=cap, seed=seed))
    records, removed = decontaminate(records, eval_path)
    records = grouped(records)
    train, dev = split(records, dev_fraction=dev_fraction, seed=seed)
    stats = statistics(train, dev, eval_path=eval_path, dedup=dedup, decontaminated=removed)
    return BuildResult(train=train, dev=dev, stats=stats)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Build the Berber LID training set.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL, dest="eval_path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="lines kept per source")
    args = parser.parse_args(argv)

    result = build_dataset(
        args.raw, args.eval_path, dev_fraction=args.dev_fraction, seed=args.seed, cap=args.cap
    )
    write_jsonl(result.train, args.out / "train.jsonl")
    write_jsonl(result.dev, args.out / "dev.jsonl")
    stats_path = args.out / "lid-dataset.stats.json"
    stats_path.write_text(json.dumps(result.stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote train=%d dev=%d out=%s", len(result.train), len(result.dev), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
