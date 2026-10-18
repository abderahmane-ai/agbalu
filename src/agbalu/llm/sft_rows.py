"""SFT rows: supervised turns and replay documents packed into rows on disk (task 11.23).

A segment is one assistant turn rendered by `agbalu.llm.chat` or one replay document, and a
segment is never split across rows. Segments are shuffled under a seed and placed first-fit
into rows of at most `capacity` tokens, so conversations and replay share rows and every
update sees both. The trainer reads a row as one variable-length forward with a boundary at
each segment, exactly as continued pretraining does.

The whole set is a few million tokens, so it is one `.npz` held in memory: the flat tokens,
the flat supervision mask, segment and row offsets, and each segment's kind.
"""

from __future__ import annotations

import random
from collections import Counter as Tally
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from agbalu.llm.chat import IGNORE, Sample
from agbalu.llm.packing import MIN_TOKENS, PackedRow, row
from agbalu.llm.recipe import SFT_ROW_TOKENS, SFT_SEED

NO_THINK: Final = 0
THINK: Final = 1
REPLAY: Final = 2
KIND_NAMES: Final = {NO_THINK: "no_think", THINK: "think", REPLAY: "replay"}

ARRAYS: Final = ("tokens", "supervised", "segment_starts", "row_starts", "kinds")


class RowsError(Exception):
    """Segments cannot be packed, or a rows file does not hold what it should."""


@dataclass(frozen=True, slots=True)
class Segment:
    ids: tuple[int, ...]
    supervised: tuple[bool, ...]
    kind: int
    source: str

    def __post_init__(self) -> None:
        if len(self.ids) < MIN_TOKENS or len(self.ids) != len(self.supervised):
            message = (
                f"{self.source}: {len(self.ids)} ids and {len(self.supervised)} mask values; "
                f"a segment needs at least {MIN_TOKENS} and one mask value per id"
            )
            raise RowsError(message)
        if not any(self.supervised[1:]):
            message = f"{self.source}: nothing is supervised"
            raise RowsError(message)


def from_sample(sample: Sample) -> Segment:
    kind = THINK if sample.think else NO_THINK
    mask = tuple(label != IGNORE for label in sample.labels)
    return Segment(sample.input_ids, mask, kind, f"{sample.conversation}#{sample.turn}")


def from_document(ids: Sequence[int], eos: int, source: str) -> Segment:
    """A replay document, `eos` first as in continued pretraining; every token after it scored."""
    tokens = (eos, *ids)
    return Segment(tokens, (False, *([True] * len(ids))), REPLAY, source)


def first_fit(lengths: Sequence[int], capacity: int, seed: int = SFT_SEED) -> list[list[int]]:
    """Segment indices per row: a seeded shuffle, then each into the first row with room."""
    for index, length in enumerate(lengths):
        if not MIN_TOKENS <= length <= capacity:
            message = f"segment {index} has {length} tokens, outside {MIN_TOKENS}..{capacity}"
            raise RowsError(message)
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    rows: list[list[int]] = []
    free: list[int] = []
    for index in order:
        slot = next((r for r, room in enumerate(free) if room >= lengths[index]), None)
        if slot is None:
            rows.append([index])
            free.append(capacity - lengths[index])
        else:
            rows[slot].append(index)
            free[slot] -= lengths[index]
    return rows


@dataclass(frozen=True, slots=True)
class Rows:
    tokens: npt.NDArray[np.uint32]
    supervised: npt.NDArray[np.bool_]
    segment_starts: npt.NDArray[np.int64]
    row_starts: npt.NDArray[np.int64]
    kinds: npt.NDArray[np.uint8]

    def __len__(self) -> int:
        return len(self.row_starts) - 1

    def packed(self, index: int) -> PackedRow:
        if not 0 <= index < len(self):
            raise IndexError(index)
        begin, end = int(self.row_starts[index]), int(self.row_starts[index + 1])
        inside = self.segment_starts[(self.segment_starts >= begin) & (self.segment_starts < end)]
        tokens = self.tokens[begin:end].astype(np.int64)
        return row(tokens, inside - begin, self.supervised[begin:end])

    @classmethod
    def read(cls, path: Path) -> Rows:
        if not path.is_file():
            message = f"{path} not found; build it with `make data TASK=sft`"
            raise RowsError(message)
        with np.load(path) as archive:
            missing = [name for name in ARRAYS if name not in archive.files]
            if missing:
                message = f"{path}: missing arrays {missing}"
                raise RowsError(message)
            rows = cls(*(np.asarray(archive[name]) for name in ARRAYS))
        rows.check(path)
        return rows

    def check(self, where: Path | str) -> None:
        """Offsets inside the token array, every row starting a segment, one kind each."""
        size = len(self.tokens)
        if len(self.supervised) != size:
            message = f"{where}: {len(self.supervised)} mask values for {size} tokens"
            raise RowsError(message)
        if self.row_starts[0] != 0 or self.row_starts[-1] != size:
            message = f"{where}: rows span {self.row_starts[0]}..{self.row_starts[-1]} of {size}"
            raise RowsError(message)
        if not np.isin(self.row_starts[:-1], self.segment_starts).all():
            message = f"{where}: a row starts inside a segment"
            raise RowsError(message)
        if len(self.kinds) != len(self.segment_starts):
            message = f"{where}: {len(self.kinds)} kinds for {len(self.segment_starts)} segments"
            raise RowsError(message)


def pack(segments: Sequence[Segment], capacity: int = SFT_ROW_TOKENS, seed: int = SFT_SEED) -> Rows:
    """Every segment in exactly one row; the token and mask totals are checked against them."""
    if not segments:
        message = "no segment to pack"
        raise RowsError(message)
    plan = first_fit([len(segment.ids) for segment in segments], capacity, seed)
    tokens: list[int] = []
    supervised: list[bool] = []
    segment_starts: list[int] = []
    row_starts: list[int] = [0]
    kinds: list[int] = []
    for members in plan:
        for index in members:
            segment = segments[index]
            segment_starts.append(len(tokens))
            kinds.append(segment.kind)
            tokens.extend(segment.ids)
            supervised.extend(segment.supervised)
        row_starts.append(len(tokens))
    placed = sorted(index for members in plan for index in members)
    if placed != list(range(len(segments))):
        message = f"{len(placed)} placements for {len(segments)} segments"
        raise RowsError(message)
    expected = sum(len(segment.ids) for segment in segments)
    if len(tokens) != expected:
        message = f"{len(tokens)} tokens written for {expected} in the segments"
        raise RowsError(message)
    rows = Rows(
        np.asarray(tokens, dtype=np.uint32),
        np.asarray(supervised, dtype=np.bool_),
        np.asarray(segment_starts, dtype=np.int64),
        np.asarray(row_starts, dtype=np.int64),
        np.asarray(kinds, dtype=np.uint8),
    )
    rows.check("packed rows")
    return rows


def write(rows: Rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as handle:
        np.savez(handle, **{name: getattr(rows, name) for name in ARRAYS})
    partial.replace(path)


def summary(rows: Rows, capacity: int = SFT_ROW_TOKENS) -> dict[str, object]:
    """Rows, fill, and supervised tokens per kind with their shares."""
    ends = np.append(rows.segment_starts[1:], len(rows.tokens))
    per_kind: Tally[str] = Tally()
    for start, end, kind in zip(rows.segment_starts, ends, rows.kinds, strict=True):
        per_kind[KIND_NAMES[int(kind)]] += int(rows.supervised[int(start) + 1 : int(end)].sum())
    total = sum(per_kind.values())
    lengths = np.diff(rows.row_starts)
    return {
        "rows": len(rows),
        "segments": len(rows.segment_starts),
        "tokens": len(rows.tokens),
        "supervised_tokens": total,
        "supervised_by_kind": dict(per_kind),
        "supervised_share": {kind: round(n / total, 4) for kind, n in per_kind.items()},
        "row_fill": round(float(lengths.mean()) / capacity, 4),
        "capacity": capacity,
    }
