"""Decode ordered boundary edits while preserving every lexical word."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from agbalu.restoration.annotations import (
    MARKS,
    OPENERS,
    PRIMARY,
    TERMINAL_LABELS,
    CasePrediction,
    apply_case,
)


@dataclass
class StructureState:
    stack: list[str] = field(default_factory=list)
    suppressed: int = 0

    def constrain(self, value: str) -> str:
        result: list[str] = []
        for original in value:
            char = original
            if char == '"':
                char = "»" if self.stack and self.stack[-1] == "»" else "«"
            if char in OPENERS:
                self.stack.append(OPENERS[char])
            elif char in OPENERS.values():
                if not self.stack or self.stack[-1] != char:
                    self.suppressed += 1
                    continue
                self.stack.pop()
            result.append(char)
        return "".join(result)

    def flush(self) -> str:
        closers = "".join(reversed(self.stack))
        self.stack.clear()
        return closers


def joint_logits(presence: NDArray[np.float32], types: NDArray[np.float32]) -> NDArray[np.float32]:
    log_present = -np.logaddexp(0, -presence)
    log_none = -np.logaddexp(0, presence)
    shifted = types - types.max(-1, keepdims=True)
    log_types = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    return np.concatenate((log_none[:, None], log_present[:, None] + log_types), axis=1)


def render_gap(value: str, *, initial: bool = False) -> str:
    if not value:
        return "" if initial else " "
    opener = next(
        (index for index, char in enumerate(value) if char in OPENERS or char in "—–-"), None
    )
    if opener is None:
        return value + " "
    before, after = value[:opener], value[opener:]
    return (
        before
        + (" " if before or not initial else "")
        + after
        + (" " if after[-1:] in "—–-" else "")
    )


def render(
    words: tuple[str, ...],
    marks: tuple[int, ...],
    casing: tuple[CasePrediction, ...],
    structures: tuple[str, ...],
    state: StructureState | None = None,
    *,
    continuing: bool = False,
) -> str:
    if len(marks) != len(words) + 1 or len(casing) != len(words) or len(structures) != len(marks):
        msg = "render labels do not cover every word and gap"
        raise ValueError(msg)
    active = state if state is not None else StructureState()
    pieces: list[str] = []
    for index, word in enumerate(words):
        value = structures[index].replace(PRIMARY, MARKS[marks[index]])
        pieces.append(render_gap(active.constrain(value), initial=index == 0 and not continuing))
        word_casing = casing[index]
        if marks[index] in TERMINAL_LABELS:
            if isinstance(word_casing, tuple):
                word_casing = (max(word_casing[0], 1), *word_casing[1:])
            else:
                word_casing = max(word_casing, 1)
        pieces.append(apply_case(word, word_casing))
    terminal = structures[-1].replace(PRIMARY, MARKS[marks[-1]])
    pieces.append(active.constrain(terminal))
    return "".join(pieces).rstrip() if continuing else "".join(pieces).strip()
