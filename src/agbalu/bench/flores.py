"""Typed access to the FLORES+ Kabyle and English splits.

`last_updated` is read from the release itself; it is what establishes whether a
language has ever been revised.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DEFAULT_ROOT: Final = Path("data/raw/hf.flores-plus-kab")

Split = Literal["dev", "devtest"]
SPLITS: Final[tuple[Split, ...]] = ("dev", "devtest")

KABYLE: Final = "kab_Latn"
ENGLISH: Final = "eng_Latn"
FRENCH: Final = "fra_Latn"
ARABIC: Final = "arb_Arab"
SPANISH: Final = "spa_Latn"
GERMAN: Final = "deu_Latn"


class FloresError(Exception):
    """The benchmark could not be read."""


@dataclass(frozen=True, slots=True)
class Sentence:
    """One FLORES+ record."""

    id: int
    text: str
    split: Split
    domain: str
    topic: str
    url: str
    last_updated: str
    iso_639_3: str
    iso_15924: str


def split_path(root: Path, split: Split, language: str) -> Path:
    return root / split / f"{language}.jsonl"


def read_split(root: Path, split: Split, language: str = KABYLE) -> list[Sentence]:
    path = split_path(root, split, language)
    if not path.is_file():
        msg = f"FLORES+ split not found: {path}"
        raise FloresError(msg)
    sentences: list[Sentence] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sentences.append(
                Sentence(
                    id=int(row["id"]),
                    text=str(row["text"]),
                    split=split,
                    domain=str(row.get("domain", "")),
                    topic=str(row.get("topic", "")),
                    url=str(row.get("url", "")),
                    last_updated=str(row.get("last_updated", "")),
                    iso_639_3=str(row.get("iso_639_3", "")),
                    iso_15924=str(row.get("iso_15924", "")),
                )
            )
    if not sentences:
        msg = f"FLORES+ split is empty: {path}"
        raise FloresError(msg)
    return sentences


def read_all(root: Path = DEFAULT_ROOT, language: str = KABYLE) -> Iterator[Sentence]:
    for split in SPLITS:
        yield from read_split(root, split, language)


def revisions(sentences: list[Sentence]) -> dict[str, int]:
    """`last_updated` histogram.

    A language whose every sentence reads `1.0` has never been revised since the
    first FLORES+ release, whatever later versions did for its siblings.
    """
    counts: dict[str, int] = {}
    for sentence in sentences:
        counts[sentence.last_updated] = counts.get(sentence.last_updated, 0) + 1
    return counts
