"""Supply reproducible document-isolated batches from memory-mapped text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from agbalu.encoder.corpus import documents
from agbalu.encoder.packing import segments


@dataclass(frozen=True)
class Packed:
    ids: Tensor
    groups: Tensor
    documents: Tensor
    positions: Tensor
    valid: Tensor

    @property
    def tokens(self) -> int:
        return int(self.valid.sum())

    def pin_memory(self) -> Packed:
        return Packed(
            *(
                value.pin_memory()
                for value in (self.ids, self.groups, self.documents, self.positions, self.valid)
            )
        )


class Batches(Dataset[Packed]):
    def __init__(
        self,
        directory: Path,
        context_tokens: int,
        micro_tokens: int,
        seed: int,
        *,
        validation: bool = False,
        limit_documents: int | None = None,
    ) -> None:
        self.directory = directory
        self.micro_tokens = micro_tokens
        self.seed = seed
        self.ids: NDArray[np.uint16] = np.memmap(directory / "tokens.bin", np.uint16, "r")
        self.groups: NDArray[np.int32] = np.memmap(directory / "groups.bin", np.int32, "r")
        table = documents(directory)
        chosen = np.flatnonzero(table[:, 2] == int(validation))
        # The complete partition is permuted before a fixed validation cap is applied.
        chosen = np.random.default_rng(seed).permutation(chosen)
        if limit_documents is not None:
            chosen = chosen[:limit_documents]
        ranges: list[tuple[int, int]] = []
        for document in chosen:
            start, end = int(table[document, 0]), int(table[document, 1])
            ranges.extend(segments(start, end, self.groups, context_tokens - 2))
        self.ranges = np.asarray(ranges, dtype=np.int64)
        self.bins: list[tuple[int, int]] = []
        first = size = 0
        for index, (start, end) in enumerate(ranges):
            length = end - start + 2
            if size and size + length > micro_tokens:
                self.bins.append((first, index))
                first, size = index, 0
            size += length
        if ranges:
            self.bins.append((first, len(ranges)))
        if not self.bins:
            message = f"{directory}: {'validation' if validation else 'train'} partition is empty"
            raise ValueError(message)

    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, index: int) -> Packed:
        if not 0 <= index < len(self.bins):
            raise IndexError(index)
        start_range, end_range = self.bins[index]
        ids = np.zeros(self.micro_tokens, dtype=np.int64)
        groups = np.full(self.micro_tokens, -1, dtype=np.int64)
        doc_ids = np.full(self.micro_tokens, -1, dtype=np.int64)
        positions = np.zeros(self.micro_tokens, dtype=np.int64)
        valid = np.zeros(self.micro_tokens, dtype=np.bool_)
        cursor = 0
        for document, row in enumerate(self.ranges[start_range:end_range]):
            start, stop = int(row[0]), int(row[1])
            length = stop - start
            ids[cursor : cursor + length + 2] = np.concatenate(([2], self.ids[start:stop], [3]))
            ownership = np.asarray(self.groups[start:stop], dtype=np.int64)
            local = ownership - start + cursor + 1
            # A group clipped by a long-document boundary is never partially supervised.
            eligible = (local >= cursor + 1) & (local < cursor + length + 1)
            if start > 0 and self.groups[start - 1] == self.groups[start]:
                eligible &= ownership != ownership[0]
            if stop < len(self.groups) and self.groups[stop - 1] == self.groups[stop]:
                eligible &= ownership != ownership[-1]
            groups[cursor + 1 : cursor + length + 1] = np.where(eligible, local, -1)
            doc_ids[cursor : cursor + length + 2] = document
            positions[cursor : cursor + length + 2] = np.arange(length + 2)
            valid[cursor : cursor + length + 2] = True
            cursor += length + 2
        return Packed(
            *(torch.from_numpy(value) for value in (ids, groups, doc_ids, positions, valid))
        )


def trim_to_budget(batch: Packed, remaining_tokens: int) -> Packed:
    if remaining_tokens >= batch.tokens:
        return batch
    if remaining_tokens < 1:
        message = f"remaining token budget must be positive, got {remaining_tokens}"
        raise ValueError(message)
    ids, groups = batch.ids.clone(), batch.groups.clone()
    valid, documents = batch.valid.clone(), batch.documents.clone()
    boundary_group = groups[remaining_tokens - 1]
    if remaining_tokens < groups.numel() and groups[remaining_tokens] == boundary_group:
        groups[groups == boundary_group] = -1
    ids[remaining_tokens:] = 0
    groups[remaining_tokens:] = -1
    valid[remaining_tokens:] = False
    documents[remaining_tokens:] = -1
    return Packed(ids, groups, documents, batch.positions, valid)
