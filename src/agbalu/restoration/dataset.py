"""Read encoded restoration windows into pinned tensor batches."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from agbalu.encoder.corpus import read_json
from agbalu.restoration.annotations import IGNORE
from agbalu.restoration.configuration import Settings
from agbalu.restoration.windows import Window, tile


@dataclass(frozen=True)
class Example:
    ids: NDArray[np.int64]
    spans: NDArray[np.int64]
    marks: NDArray[np.int64]
    casing: NDArray[np.int64]
    structure: NDArray[np.int64]
    edges: tuple[bool, bool]
    owned_words: int | None = None
    case_spans: NDArray[np.int64] | None = None
    case_words: NDArray[np.int64] | None = None
    case_features: NDArray[np.float32] | None = None


@dataclass(frozen=True)
class Batch:
    ids: Tensor
    attention: Tensor
    documents: Tensor
    positions: Tensor
    spans: Tensor
    words: Tensor
    edges: Tensor
    marks: Tensor
    casing: Tensor
    case_spans: Tensor
    case_words: Tensor
    case_features: Tensor
    structure: Tensor
    owned_words: Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            *(getattr(self, field.name).to(device, non_blocking=True) for field in fields(self))
        )

    def pin_memory(self) -> "Batch":
        return Batch(*(getattr(self, field.name).pin_memory() for field in fields(self)))


def case_alignment(
    row: Example,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    if row.case_spans is not None:
        if row.case_words is None or row.case_features is None:
            message = "incomplete component case alignment"
            raise ValueError(message)
        return row.case_spans, row.case_words, row.case_features
    if len(row.casing) != len(row.spans):
        message = f"{len(row.casing)} components require explicit case alignment"
        raise ValueError(message)
    features = np.zeros((len(row.spans), 3), np.float32)
    features[:, 1:] = 1
    return row.spans, np.arange(len(row.spans), dtype=np.int64), features


def collate(
    examples: list[Example], context_tokens: int, *, encoder_tokens: int | None = None
) -> Batch:
    batch = len(examples)
    if not batch:
        msg = "cannot collate an empty batch"
        raise ValueError(msg)
    # Word capacity follows the actual batch, rounded to reduce compiler shape variants.
    word_count = ((max(len(row.spans) for row in examples) + 31) // 32) * 32
    case_count = ((max(len(row.casing) for row in examples) + 31) // 32) * 32
    block_tokens = encoder_tokens if encoder_tokens is not None else context_tokens * min(batch, 8)
    if block_tokens < context_tokens:
        msg = f"encoder block has {block_tokens} tokens, below the {context_tokens}-token context"
        raise ValueError(msg)
    placements: list[tuple[int, int]] = []
    block, cursor = 0, 0
    for index, row in enumerate(examples):
        length = len(row.ids)
        if length > context_tokens:
            msg = f"example {index}: {length} tokens exceed {context_tokens}"
            raise ValueError(msg)
        if cursor + length > block_tokens:
            block += 1
            cursor = 0
        placements.append((block, cursor))
        cursor += length
    ids = np.zeros((block + 1, block_tokens), np.int64)
    attention = np.zeros_like(ids, np.bool_)
    documents = np.full_like(ids, -1)
    positions = np.zeros_like(ids)
    spans = np.zeros((batch, word_count, 2), np.int64)
    words = np.zeros((batch, word_count), np.bool_)
    edges = np.zeros((batch, 2), np.bool_)
    marks = np.full((batch, word_count + 1), IGNORE, np.int64)
    structure = np.full_like(marks, IGNORE)
    casing = np.full((batch, case_count), IGNORE, np.int64)
    case_spans = np.zeros((batch, case_count, 2), np.int64)
    case_words = np.zeros((batch, case_count), np.int64)
    case_features = np.zeros((batch, case_count, 3), np.float32)
    owned_words = np.asarray(
        [len(row.spans) if row.owned_words is None else row.owned_words for row in examples],
        np.int64,
    )
    for index, row in enumerate(examples):
        length, count = len(row.ids), len(row.spans)
        components = len(row.casing)
        block, begin = placements[index]
        ids[block, begin : begin + length] = row.ids
        attention[block, begin : begin + length] = True
        documents[block, begin : begin + length] = index
        positions[block, begin : begin + length] = np.arange(length)
        spans[index, :count] = row.spans + block * block_tokens + begin
        words[index, :count] = True
        edges[index] = row.edges
        marks[index, : count + 1] = row.marks
        structure[index, : count + 1] = row.structure
        casing[index, :components] = row.casing
        aligned, owners, features = case_alignment(row)
        case_spans[index, :components] = aligned + block * block_tokens + begin
        case_words[index, :components] = owners
        case_features[index, :components] = features
    return Batch(
        *(
            torch.from_numpy(value)
            for value in (
                ids,
                attention,
                documents,
                positions,
                spans,
                words,
                edges,
                marks,
                casing,
                case_spans,
                case_words,
                case_features,
                structure,
                owned_words,
            )
        )
    )


def collate_update(examples: list[Example], context_tokens: int, micro_windows: int) -> list[Batch]:
    batches: list[Batch] = []
    pending: list[Example] = []
    tokens = 0
    token_budget = context_tokens * min(micro_windows, 8)
    for row in examples:
        if pending and (len(pending) >= micro_windows or tokens + len(row.ids) > token_budget):
            batches.append(collate(pending, context_tokens, encoder_tokens=token_budget))
            pending = []
            tokens = 0
        pending.append(row)
        tokens += len(row.ids)
    if pending:
        batches.append(collate(pending, context_tokens, encoder_tokens=token_budget))
    return batches


class Windows(Dataset[Example]):
    def __init__(
        self, directory: Path, settings: Settings, split: int, *, streaming: bool = False
    ) -> None:
        self.settings = settings
        self.ids: NDArray[np.uint16] = np.memmap(directory / "tokens.bin", np.uint16, "r")
        self.offsets: NDArray[np.int64] = np.load(directory / "offsets.npy", mmap_mode="r")
        self.documents: NDArray[np.int64] = np.load(directory / "documents.npy", mmap_mode="r")
        self.cases: NDArray[np.int16] = np.memmap(directory / "cases.bin", np.int16, "r")
        self.marks: NDArray[np.int16] = np.memmap(directory / "marks.bin", np.int16, "r")
        self.structures: NDArray[np.int16] = np.memmap(directory / "structure.bin", np.int16, "r")
        self.case_offsets: NDArray[np.int64] | None = None
        self.case_rows: NDArray[np.int32] | None = None
        if read_json(directory / "corpus.stats.json")["format_version"] != 1:
            self.case_offsets = np.load(directory / "case_offsets.npy", mmap_mode="r")
            self.case_rows = np.memmap(directory / "case_spans.bin", np.int32, "r").reshape(-1, 6)
        self.windows: list[tuple[int, Window]] = []
        body = settings.geometry.context_tokens - 2
        core = settings.streaming_core_tokens if streaming else settings.core_tokens
        left = body - core - settings.lookahead_tokens if streaming else settings.left_tokens
        for index in np.flatnonzero(self.documents[:, 4] == split):
            token_start, word_start, count = map(int, self.documents[index, :3])
            offsets = self.offsets[word_start : word_start + count + 1] - token_start
            for window in tile(
                offsets.tolist(),
                body,
                core,
                left,
                right_tokens=settings.lookahead_tokens if streaming else None,
            ):
                self.windows.append((int(index), window))
        if not self.windows:
            msg = f"{directory}: split={split} has no restoration windows"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Example:
        document, window = self.windows[index]
        _, word_start, _, gap_start, _, known_start, known_end = map(int, self.documents[document])
        first = word_start + window.start
        last = word_start + window.stop
        begin, end = int(self.offsets[first]), int(self.offsets[last])
        ids = np.concatenate(([2], self.ids[begin:end], [3])).astype(np.int64)
        spans = np.stack((self.offsets[first:last], self.offsets[first + 1 : last + 1]), 1)
        spans = spans - begin + 1
        marks = np.asarray(
            self.marks[gap_start + window.start : gap_start + window.stop + 1], dtype=np.int64
        ).copy()
        structure = np.asarray(
            self.structures[gap_start + window.start : gap_start + window.stop + 1], dtype=np.int64
        ).copy()
        case_spans: NDArray[np.int64] | None = None
        case_words: NDArray[np.int64] | None = None
        case_features: NDArray[np.float32] | None = None
        if self.case_offsets is not None and self.case_rows is not None:
            case_first, case_last = int(self.case_offsets[first]), int(self.case_offsets[last])
            rows = self.case_rows[case_first:case_last]
            casing = np.asarray(self.cases[case_first:case_last], dtype=np.int64).copy()
            case_spans = np.asarray(rows[:, :2], np.int64) - begin + 1
            case_words = np.asarray(rows[:, 2], np.int64) - first
            features = np.stack((rows[:, 3], rows[:, 4], rows[:, 4] - rows[:, 3]), 1)
            case_features = np.asarray(features / rows[:, 5:6], np.float32)
        else:
            casing = np.asarray(self.cases[first:last], dtype=np.int64).copy()
        core_first = window.core_start - window.start
        core_last = window.core_stop - window.start
        if case_words is None:
            casing[:core_first] = casing[core_last:] = IGNORE
        else:
            casing[(case_words < core_first) | (case_words >= core_last)] = IGNORE
        # Gaps after core words belong to this window; only a true beginning owns gap zero.
        marks[: core_first + 1] = marks[core_last + 1 :] = IGNORE
        structure[: core_first + 1] = structure[core_last + 1 :] = IGNORE
        count = int(self.documents[document, 2])
        real_start = window.start == 0 and bool(known_start)
        real_end = window.stop == count and bool(known_end)
        if real_start and window.core_start == 0:
            marks[0] = self.marks[gap_start]
            structure[0] = self.structures[gap_start]
        if not real_start:
            if case_words is None or case_features is None:
                casing[0] = IGNORE
            else:
                casing[(case_words == 0) & (case_features[:, 0] == 0)] = IGNORE
        if not real_end:
            marks[-1] = structure[-1] = IGNORE
        return Example(
            ids,
            spans,
            marks,
            casing,
            structure,
            (real_start, real_end),
            core_last - core_first,
            case_spans,
            case_words,
            case_features,
        )
