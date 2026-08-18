"""Restoring marks and capitals on text the ASR model produced.

The words come back unchanged: this predicts two labels per word and never edits the
spelling, so it cannot introduce a transcription error into text Fadhma got right.

Words past `max_length` are dropped by the encoder, so a prediction covers a prefix of the
sentence. `predict` returns the words it actually covered rather than padding the tail with
a guess, and the metrics are computed against that same prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from agbalu.punctuation.dataset import MAX_LENGTH, Tokenizer, encode_words
from agbalu.punctuation.labels import Annotation, annotate, restore
from agbalu.punctuation.model import Restorer
from agbalu.tokenizer.spec import PAD_ID

BATCH: int = 128


@dataclass(frozen=True, slots=True)
class Restoration:
    words: tuple[str, ...]
    punctuation: tuple[int, ...]
    case: tuple[int, ...]

    @property
    def text(self) -> str:
        return restore(self.words, self.punctuation, self.case)


def _pad(rows: list[list[int]], device: torch.device) -> tuple[Tensor, Tensor]:
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention[index, : len(row)] = True
    return input_ids.to(device), attention.to(device)


@torch.inference_mode()
def predict(
    model: Restorer,
    tokenizer: Tokenizer,
    texts: list[str],
    *,
    device: torch.device | None = None,
    batch_size: int = BATCH,
    max_length: int = MAX_LENGTH,
) -> list[Restoration]:
    """One `Restoration` per input, in order. Empty inputs come back empty, never dropped."""
    target = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    encoded: list[tuple[tuple[str, ...], list[int], list[int]]] = []
    for text in texts:
        words = annotate(text).words
        ids, first = encode_words(words, tokenizer, max_length)
        encoded.append((words, ids, first))

    results: list[Restoration] = []

    for start in range(0, len(encoded), batch_size):
        window = encoded[start : start + batch_size]
        live = [entry for entry in window if entry[2]]
        if not live:
            results.extend(Restoration((), (), ()) for _ in window)
            continue

        input_ids, attention = _pad([entry[1] for entry in live], target)
        heads = model(input_ids, attention)
        punctuation = heads.punctuation.argmax(-1).to("cpu")
        case = heads.case.argmax(-1).to("cpu")

        row = 0
        for words, _, first in window:
            if not first:
                results.append(Restoration((), (), ()))
                continue
            index = torch.tensor(first, dtype=torch.long)
            results.append(
                Restoration(
                    tuple(words[: len(first)]),
                    tuple(int(value) for value in punctuation[row].index_select(0, index)),
                    tuple(int(value) for value in case[row].index_select(0, index)),
                )
            )
            row += 1

    if was_training:
        model.train()
    return results


def truncated_gold(text: str, covered: int) -> Annotation:
    """The reference cut to the words a prediction covered, so the two are comparable."""
    gold = annotate(text)
    return Annotation(gold.words[:covered], gold.punctuation[:covered], gold.case[:covered])
