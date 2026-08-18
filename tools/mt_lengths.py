"""Measure the padded-token cost of one MT fine-tuning step.

The step cost is not `batch x max_source_length`: `DataCollatorForSeq2Seq` pads to the
longest member of each micro-batch, and the corpus is far shorter than the 128-token cap.
This is what sets `train_sampling_strategy` in `modal_app.mt`, and what the fine-tune's
throughput projection has to be built on.

    python3 -m tools.mt_lengths [--sample N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Final

TRAIN: Final = Path("data/processed/mt/train.jsonl")
TOKENIZER: Final = "facebook/nllb-200-distilled-600M"
"""Every NLLB-200 size shares one vocabulary, so the distilled 600M gives the 1.3B's
token counts. The trim keeps segmentation and only remaps ids, so it does not move them."""

MAX_LEN: Final = 128
MICRO: Final = 32
ACCUM: Final = 64
MEGA_BATCH_MULT: Final = 50
"""`transformers` sorts within megabatches of this many batches, not globally."""


def measure(sample: int, micro: int, accum: int) -> dict[str, float]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    sources: list[str] = []
    targets: list[str] = []
    with TRAIN.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= sample:
                break
            row = json.loads(line)
            sources.append(row["source"])
            targets.append(row["target"])

    def lengths(texts: list[str]) -> list[int]:
        encoded = tokenizer(texts, truncation=True, max_length=MAX_LEN)
        return [len(ids) for ids in encoded["input_ids"]]

    source_lengths, target_lengths = lengths(sources), lengths(targets)

    shuffled = list(range(len(sources)))
    random.Random(0).shuffle(shuffled)  # noqa: S311 — a fixed-seed measurement, not a secret

    def cost(order: list[int]) -> tuple[float, float]:
        padded = real = batches = 0
        for start in range(0, len(order) - micro, micro):
            chunk = order[start : start + micro]
            padded += micro * max(source_lengths[i] for i in chunk)
            padded += micro * max(target_lengths[i] for i in chunk)
            real += sum(source_lengths[i] + target_lengths[i] for i in chunk)
            batches += 1
        return padded / batches, real / padded

    random_micro, random_real = cost(shuffled)
    grouped_micro, grouped_real = cost(_length_grouped(shuffled, source_lengths, micro))
    return {
        "pairs": float(len(sources)),
        "source_mean": statistics.mean(source_lengths),
        "target_mean": statistics.mean(target_lengths),
        "source_median": float(statistics.median(source_lengths)),
        "random_padded_per_step": random_micro * accum,
        "random_real_fraction": random_real,
        "grouped_padded_per_step": grouped_micro * accum,
        "grouped_real_fraction": grouped_real,
        "grouped_speedup": random_micro / grouped_micro,
    }


def _length_grouped(order: list[int], lengths: list[int], micro: int) -> list[int]:
    """`transformers.trainer_pt_utils.get_length_grouped_indices`, reproduced.

    Megabatches of `50 x batch_size` are sorted by length descending, so a batch of
    `batch_size` consecutive indices holds sequences of near-equal length. It keys on the
    source `input_ids` only; the target side follows by correlation, not by construction.
    """
    mega = min(len(order) // (micro * 4), MEGA_BATCH_MULT) or 1
    size = mega * micro
    grouped: list[int] = []
    for start in range(0, len(order), size):
        grouped.extend(sorted(order[start : start + size], key=lambda i: lengths[i], reverse=True))
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=120_000)
    parser.add_argument("--micro", type=int, default=MICRO)
    parser.add_argument("--accum", type=int, default=ACCUM)
    args = parser.parse_args(argv)

    if not TRAIN.is_file():
        message = f"{TRAIN} is missing; run `make mt-data` first"
        raise SystemExit(message)

    for key, value in measure(args.sample, args.micro, args.accum).items():
        print(f"{key}: {value:,.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
