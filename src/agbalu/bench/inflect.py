"""Scoring for morphological inflection and analysis, and the floor both are read against.

`agbalu/KabInflect` is split by **lemma**, not by form, so a test verb's paradigm has
never been seen in any cell. That is what makes the floor meaningful: copying the lemma
unchanged is what a system with no morphology does, and any model that cannot beat it has
learned the corpus rather than the language.

The floor is produced here rather than quoted, for the same reason the character table
lives beside the script-conversion scorer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.bench.strings import StringScore, score_strings

SPLIT_DIR: Final = Path("data/tasks/inflection")

INFLECTION: Final = "inflection"
ANALYSIS: Final = "analysis"
CONFIGS: Final = (INFLECTION, ANALYSIS)


class InflectionError(Exception):
    """A scoring request that cannot produce a number."""


@dataclass(frozen=True, slots=True)
class Entry:
    """One inflected form and the features that select it."""

    lemma: str
    feats: str
    form: str


def read_split(
    split: str, config: str = INFLECTION, directory: Path = SPLIT_DIR, limit: int | None = None
) -> list[Entry]:
    """A built split of `agbalu/KabInflect`."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    if config not in CONFIGS:
        message = f"no config {config!r}; expected one of {CONFIGS}"
        raise InflectionError(message)
    path = directory / config / f"{split}.parquet"
    if not path.is_file():
        message = f"no {split} split at {path}; build it with tools/build_kabinflect.py"
        raise InflectionError(message)
    table = pq.read_table(path, columns=["lemma", "feats", "form"])
    if limit is not None:
        table = table.slice(0, limit)
    return [Entry(**row) for row in table.to_pylist()]


def copy_floor(entries: Sequence[Entry]) -> StringScore:
    """What a system that emits the lemma unchanged scores.

    Not zero, and that is the point of measuring it: Kabyle's imperative singular *is*
    the citation form for most verbs, so a copy is correct in one cell of every paradigm
    and a headline exact match has to be read against that, not against zero.
    """
    return score_strings([entry.form for entry in entries], [entry.lemma for entry in entries])


def score_forms(entries: Sequence[Entry], hypotheses: Sequence[str]) -> StringScore:
    """Predicted surface forms against the gold ones."""
    return score_strings([entry.form for entry in entries], hypotheses)
