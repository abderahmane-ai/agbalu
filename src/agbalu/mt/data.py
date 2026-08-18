"""The fine-tuning corpus: which pairs, in which directions, split how.

NLLB's own paper reports 72,000 sentences of public Kabyle bitext (Table 12), and 90.1%
of AƔBALU-Parallel is its mined output. Fine-tuning it on that output teaches it what it
already knows, so the default selection keeps only what NLLB did not mine.
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypedDict

from agbalu.parallel.quality import HARD_DEFECTS

Direction = Literal["kab-eng", "eng-kab", "kab-fra", "fra-kab"]

NLLB_MARKER: Final = "nllb"
"""Source ids carrying it are NLLB's mined output, whoever re-uploaded it."""

PARALLEL_DIR: Final = Path("data/interim/parallel")
OUTPUT_DIR: Final = Path("data/processed/mt")

DEV_PAIRS: Final = 2_000
"""Held out for early stopping. FLORES+ is the test set and is never trained against, so
the stopping signal has to come from our own pairs."""


class BuildStats(TypedDict):
    """What `build` wrote, and the counts that explain the difference from `read`."""

    read: int
    mined_excluded: int
    hard_defective: int
    duplicate: int
    kept_pairs: int
    examples: int
    train: int
    dev: int
    include_mined: bool
    seed: int
    by_direction: dict[str, int]
    by_source: dict[str, int]


@dataclass(frozen=True, slots=True)
class Example:
    source: str
    target: str
    direction: Direction
    source_id: str

    def as_row(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "direction": self.direction,
            "source_id": self.source_id,
        }


@dataclass(slots=True)
class Selection:
    read: int = 0
    mined: int = 0
    defective: int = 0
    duplicate: int = 0
    kept_pairs: int = 0
    by_source: Counter[str] = field(default_factory=Counter)
    by_direction: Counter[str] = field(default_factory=Counter)

    @property
    def examples(self) -> int:
        return sum(self.by_direction.values())


def _key(kab: str, other: str) -> tuple[str, str]:
    """Deduplication key. NFC and casefold because the same pair reaches us from several
    uploaders with different capitalisation and composition, and each would otherwise be
    counted as new data."""
    return (
        unicodedata.normalize("NFC", kab).casefold().strip(),
        unicodedata.normalize("NFC", other).casefold().strip(),
    )


def is_mined(source_id: str) -> bool:
    return NLLB_MARKER in source_id.lower()


def directions_for(language: str) -> tuple[Direction, Direction]:
    if language == "eng":
        return ("kab-eng", "eng-kab")
    if language == "fra":
        return ("kab-fra", "fra-kab")
    message = f"no directions for {language!r}; the corpus holds eng and fra"
    raise ValueError(message)


def select(
    parallel_dir: Path = PARALLEL_DIR, *, include_mined: bool = False
) -> tuple[list[Example], Selection]:
    """Every direction of every kept pair, with the counts that explain the difference.

    A pair is kept when it carries no hard defect — soft defects stay in, since dropping
    515,650 number mismatches would cost most of the corpus to remove a signal that is
    mostly footnote markers.
    """
    stats = Selection()
    examples: list[Example] = []

    for path in sorted(parallel_dir.glob("agbalu-parallel-v1.kab-*.jsonl")):
        language = path.stem.split("-")[-1]
        seen: set[tuple[str, str]] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats.read += 1
                if not include_mined and is_mined(row["source"]):
                    stats.mined += 1
                    continue
                if any(defect in HARD_DEFECTS for defect in row["defects"]):
                    stats.defective += 1
                    continue
                key = _key(row["kab"], row[language])
                if key in seen:
                    stats.duplicate += 1
                    continue
                seen.add(key)
                stats.kept_pairs += 1
                stats.by_source[row["source"]] += 1
                forward, backward = directions_for(language)
                for direction, (src, tgt) in (
                    (forward, (row["kab"], row[language])),
                    (backward, (row[language], row["kab"])),
                ):
                    examples.append(Example(src, tgt, direction, row["source"]))
                    stats.by_direction[direction] += 1

    return examples, stats


def split(
    examples: Sequence[Example], *, dev_pairs: int = DEV_PAIRS, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Train and dev, split on the *pair* rather than the example.

    Both directions of one pair are the same sentence pair seen twice: putting one in
    train and the other in dev would leak the target through the source side.
    """
    by_pair: dict[tuple[str, str], list[Example]] = {}
    for example in examples:
        forward = example.direction.startswith("kab-")
        kab, other = (
            (example.source, example.target) if forward else (example.target, example.source)
        )
        by_pair.setdefault(_key(kab, other), []).append(example)

    keys = sorted(by_pair)
    rng = random.Random(seed)  # noqa: S311 — a fixed-seed split, not a secret
    rng.shuffle(keys)

    held = keys[: min(dev_pairs, len(keys))]
    dev_keys = set(held)
    train = [e for key in keys if key not in dev_keys for e in by_pair[key]]
    dev = [e for key in held for e in by_pair[key]]
    return train, dev


def write(examples: Sequence[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_row(), ensure_ascii=False) + "\n")


def read(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                Example(row["source"], row["target"], row["direction"], row["source_id"])
            )
    return examples


def build(
    parallel_dir: Path = PARALLEL_DIR,
    output_dir: Path = OUTPUT_DIR,
    *,
    include_mined: bool = False,
    dev_pairs: int = DEV_PAIRS,
    seed: int = 0,
) -> BuildStats:
    """Write `train.jsonl`, `dev.jsonl` and the stats beside them; return the stats."""
    examples, selection = select(parallel_dir, include_mined=include_mined)
    train, dev = split(examples, dev_pairs=dev_pairs, seed=seed)

    write(train, output_dir / "train.jsonl")
    write(dev, output_dir / "dev.jsonl")

    stats: BuildStats = {
        "read": selection.read,
        "mined_excluded": selection.mined,
        "hard_defective": selection.defective,
        "duplicate": selection.duplicate,
        "kept_pairs": selection.kept_pairs,
        "examples": selection.examples,
        "train": len(train),
        "dev": len(dev),
        "include_mined": include_mined,
        "seed": seed,
        "by_direction": dict(selection.by_direction),
        "by_source": dict(selection.by_source.most_common()),
    }
    path = output_dir / "agbalu-mt-v1.stats.json"
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats
