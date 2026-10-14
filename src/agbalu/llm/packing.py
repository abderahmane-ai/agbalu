"""Packed blocks as one variable-length row: boundaries, positions and labels.

A micro-batch of blocks is flattened into a single row and every document in it becomes a
segment: the full-attention layers read `cu_seqlens` through FlashAttention's varlen path,
the Gated DeltaNet layers reset their state at the same offsets, and positions restart at
zero in each segment. Measured on the 0.8B pilot on 2026-10-14: per-token loss of three
packed documents equals each scored alone, to 0.0000 nats, once the short convolution also
respects the boundaries (`agbalu.llm.kernels`).

A segment starts at every `eos` and at every block edge, since a document cut by the
previous block continues here without one. The label at each segment start is masked: it
would be predicted from the previous segment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

IGNORE_INDEX: Final = -100

MIN_TOKENS: Final = 2
"""The shortest block or document with anything to predict: one token and its successor."""

BLOCK_AXES: Final = 2
"""A micro-batch is `(blocks, length)`."""


class PackingError(Exception):
    """A micro-batch cannot be turned into a packed row."""


@dataclass(frozen=True, slots=True)
class PackedRow:
    """One forward pass worth of tokens, host-side."""

    input_ids: npt.NDArray[np.int64]
    position_ids: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]
    cu_seqlens: npt.NDArray[np.int32]
    max_length: int
    targets: int
    """Labels the loss scores: every position after the first that is not masked."""


def segment_starts(
    tokens: npt.NDArray[np.integer], row_length: int, eos: int
) -> npt.NDArray[np.int64]:
    """Offsets where a segment begins: each block edge and each `eos`."""
    if tokens.ndim != 1:
        message = f"expected a flat token array, got shape {tokens.shape}"
        raise PackingError(message)
    if row_length < 1 or len(tokens) % row_length:
        message = f"{len(tokens)} tokens do not divide into rows of {row_length}"
        raise PackingError(message)
    edges = np.arange(0, len(tokens), row_length, dtype=np.int64)
    separators = np.flatnonzero(tokens == eos).astype(np.int64)
    return np.union1d(edges, separators)


def row(tokens: npt.NDArray[np.int64], starts: npt.NDArray[np.int64]) -> PackedRow:
    """One row from its tokens and the sorted offsets where its segments begin."""
    if len(starts) == 0 or starts[0] != 0:
        message = "the first segment must start at offset 0"
        raise PackingError(message)
    bounds = np.append(starts, len(tokens))
    lengths = np.diff(bounds)
    if (lengths < 1).any():
        message = "segment offsets must be strictly increasing and inside the row"
        raise PackingError(message)
    positions = np.arange(len(tokens), dtype=np.int64) - np.repeat(starts, lengths)
    labels = tokens.copy()
    labels[starts] = IGNORE_INDEX
    return PackedRow(
        input_ids=tokens[None, :],
        position_ids=positions[None, :],
        labels=labels[None, :],
        cu_seqlens=bounds.astype(np.int32),
        max_length=int(lengths.max()),
        targets=int((labels[1:] != IGNORE_INDEX).sum()),
    )


def pack_row(blocks: npt.NDArray[np.integer], eos: int) -> PackedRow:
    """`(n, L)` blocks as one row of `n * L` tokens with its segment metadata."""
    if blocks.ndim != BLOCK_AXES or blocks.shape[0] < 1 or blocks.shape[1] < MIN_TOKENS:
        message = f"expected (blocks, length) with length >= {MIN_TOKENS}, got {blocks.shape}"
        raise PackingError(message)
    tokens = blocks.reshape(-1).astype(np.int64)
    return row(tokens, segment_starts(tokens, blocks.shape[1], eos))


def plan_rows(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """Document indices per row, filled greedily in input order up to `capacity` tokens."""
    if capacity < MIN_TOKENS:
        message = f"capacity must hold eos and one token, got {capacity}"
        raise PackingError(message)
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, length in enumerate(lengths):
        if not 1 <= length <= capacity:
            message = f"document {index} has {length} tokens, outside 1..{capacity}"
            raise PackingError(message)
        if current and used + length > capacity:
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        groups.append(current)
    return groups


def concatenate(documents: Sequence[Sequence[int]]) -> PackedRow:
    """Documents as one row, each its own segment."""
    lengths = np.fromiter((len(d) for d in documents), dtype=np.int64, count=len(documents))
    starts = np.cumsum(lengths) - lengths
    tokens = np.fromiter(
        (token for document in documents for token in document),
        dtype=np.int64,
        count=int(lengths.sum()),
    )
    return row(tokens, starts)


def pack_documents(documents: list[list[int]], eos: int, capacity: int) -> list[PackedRow]:
    """Whole documents, `eos`-prefixed, greedily filled into rows of at most `capacity`.

    For scoring: no document is split, so the summed loss over the rows is the summed loss
    of each document scored alone. A document longer than `capacity` is truncated to it.
    """
    if any(not ids for ids in documents):
        message = "a document encoded to no tokens"
        raise PackingError(message)
    prefixed = [[eos, *ids][:capacity] for ids in documents]
    groups = plan_rows([len(document) for document in prefixed], capacity)
    return [concatenate([prefixed[i] for i in group]) for group in groups]
