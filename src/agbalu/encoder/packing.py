"""Align byte-offset pieces to atomic word groups and segments."""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def word_groups(text: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    byte_offsets = [0]
    for character in text:
        byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))
    words = [(byte_offsets[m.start()], byte_offsets[m.end()]) for m in re.finditer(r"\S+", text)]
    if not words:
        return [-1] * len(offsets)
    starts, ends = zip(*words, strict=True)
    owner: list[int] = []
    previous_end = -1
    anchor = 0
    for index, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= byte_offsets[-1]:
            message = (
                f"piece {index}: byte offsets {(start, end)} exceed text bytes {byte_offsets[-1]}"
            )
            raise ValueError(message)
        first = min(bisect.bisect_right(ends, start), len(words) - 1)
        last = min(max(first, bisect.bisect_left(starts, end) - 1), len(words) - 1)
        if first > previous_end:
            anchor = index
        owner.append(anchor)
        previous_end = max(previous_end, last)
    return owner


def segments(
    start: int, end: int, groups: Sequence[int] | NDArray[np.int32], body_tokens: int
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + body_tokens, end)
        if stop < end and groups[stop] == groups[stop - 1]:
            boundary = stop
            while boundary > cursor and groups[boundary - 1] == groups[stop]:
                boundary -= 1
            if boundary > cursor:
                stop = boundary
        result.append((cursor, stop))
        cursor = stop
    return result
