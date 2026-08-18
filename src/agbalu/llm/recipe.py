"""The continued-pretraining recipe: data path, schedule, and the constants that size a run.

Full continued pretraining of every parameter, one stage. LoRA is refused on measurement
rather than preference: in CPT at rank 256 it needs 20B tokens to match full fine-tuning at
4B ([arXiv 2405.09673](https://arxiv.org/abs/2405.09673), TMLR 2024), and a five-fold sample
inefficiency is unaffordable against a corpus of 230M tokens an epoch. Adapter architectures
are refused for the same reason from the other side: their validation is at 80B tokens, and
[AfriqueLLM](https://arxiv.org/abs/2601.06395) — twenty African languages, 26B tokens —
measures data composition as the primary driver of CPT gains and architecture as secondary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Final

MIN_BLOCK: Final = 2
"""One token of context and one to predict, the shortest packed block that teaches
anything."""

SEQUENCE_LENGTH: Final = 1024

SHUFFLE_SEED: Final = 20260811
"""Nothing else in the path shuffles. `mixture` writes the corpus source by source and
`pack` packs in file order, so without this permutation every batch would come from one
source and the loss would track the source rather than the model."""

WARMUP_FRACTION: Final = 0.01
PEAK_LR: Final = 2e-5
FINAL_LR_FRACTION: Final = 0.1
WEIGHT_DECAY: Final = 0.1
"""Full CPT updates every weight, so the schedule is a continuation of pretraining rather
than the 3e-4 an adapter over a frozen base can take: at 3e-4 on 1.88B parameters the run
overwrites what the base knows. The decay is the usual 0.1 for the same reason — arXiv
2606.06888's ~1.0 is measured for training a fixed corpus from scratch, not for continuing."""


class RecipeError(Exception):
    """The corpus cannot be read, or a schedule cannot be built."""


def documents(path: Path) -> Iterator[str]:
    """Every document of the CPT corpus, in file order, streamed."""
    if not path.is_file():
        message = f"CPT corpus not found at {path}; run `make llm TASK=mixture`"
        raise RecipeError(message)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "text" not in row:
                message = f"{path}:{number} is not a document"
                raise RecipeError(message)
            yield str(row["text"])


def pack(streams: Iterable[Sequence[int]], length: int, separator: int) -> Iterator[list[int]]:
    """Concatenate token sequences and cut fixed-length blocks, dropping the remainder.

    The corpus averages 43.9 tokens a document against a 1024-token sequence, so padding
    each document into its own sequence would spend 96% of the arithmetic on a token that
    teaches nothing.
    """
    if length < MIN_BLOCK:
        message = f"sequence length must be at least {MIN_BLOCK}, got {length}"
        raise RecipeError(message)
    buffer: list[int] = []
    for ids in streams:
        buffer.extend(ids)
        buffer.append(separator)
        while len(buffer) >= length:
            yield buffer[:length]
            del buffer[:length]


def learning_rate(step: int, total: int, *, peak: float = PEAK_LR) -> float:
    """Linear warmup then cosine decay to `FINAL_LR_FRACTION` of peak.

    `step` is zero-based and step 0 is not zero: a first step at lr 0 wastes an optimizer
    step, and on a short run a measurable share of the budget.
    """
    if total < 1:
        message = f"total steps must be positive, got {total}"
        raise RecipeError(message)
    if not 0 <= step < total:
        message = f"step {step} is outside a run of {total}"
        raise RecipeError(message)
    warmup = max(1, int(total * WARMUP_FRACTION))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine)
