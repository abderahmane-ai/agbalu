"""Tile word-aligned supervision with bounded context and complete coverage."""

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    core_start: int
    core_stop: int


def tile(
    offsets: Sequence[int],
    body_tokens: int,
    core_tokens: int,
    left_tokens: int,
    *,
    right_tokens: int | None = None,
) -> list[Window]:
    if not offsets or offsets[0] != 0 or any(b <= a for a, b in pairwise(offsets)):
        msg = f"invalid word offsets: {list(offsets)[:20]}"
        raise ValueError(msg)
    if core_tokens < 1 or left_tokens < 0 or body_tokens <= core_tokens + left_tokens:
        msg = "invalid core/context token budget"
        raise ValueError(msg)
    if any(b - a > body_tokens for a, b in pairwise(offsets)):
        msg = "one word exceeds the full lexical context budget"
        raise ValueError(msg)
    count = len(offsets) - 1
    result: list[Window] = []
    cursor = 0
    while cursor < count:
        core_stop = max(cursor + 1, bisect.bisect_right(offsets, offsets[cursor] + core_tokens) - 1)
        core_stop = min(core_stop, count)
        start = bisect.bisect_left(offsets, max(0, offsets[cursor] - left_tokens))
        if cursor and start == cursor:
            start = cursor - 1
        stop = min(count, bisect.bisect_right(offsets, offsets[start] + body_tokens) - 1)
        if stop < core_stop or (core_stop < count and stop == core_stop):
            # A complete right neighbor is required to supervise the final core gap.
            needed = min(count, core_stop + 1)
            start = bisect.bisect_left(offsets, max(0, offsets[needed] - body_tokens))
            stop = min(count, bisect.bisect_right(offsets, offsets[start] + body_tokens) - 1)
        if start > cursor or stop < core_stop:
            msg = f"cannot fit word-aligned core [{cursor},{core_stop})"
            raise ValueError(msg)
        if right_tokens is not None:
            stop = min(stop, bisect.bisect_right(offsets, offsets[core_stop] + right_tokens) - 1)
        result.append(Window(start, stop, cursor, core_stop))
        cursor = core_stop
    return result
