"""Tokenisation, and the alignment that makes a word-level task a token-level one.

A label belongs to a word, the encoder consumes subwords, and only the first subword of each
word carries the label; every other position is `IGNORE_INDEX` and the loss never sees it.
Encoding a word at a time rather than the whole sentence is what makes that index exact — a
sentence-level encode gives no way back from a piece to the word it came from.

The corpus is encoded once into flat arrays rather than re-tokenised each epoch: 1.3M
sentences is 40 s of SentencePiece per pass, and the packed form costs about 100 MB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from agbalu.punctuation.corpus import Row
from agbalu.punctuation.labels import CASE, IGNORE_INDEX, PUNCTUATION, Annotation, annotate
from agbalu.tokenizer.spec import CLS_ID, PAD_ID, SEP_ID

MAX_LENGTH: Final = 128


class Tokenizer(Protocol):
    def encode(self, text: str, out_type: type[int]) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class EncodedCorpus:
    """Subword ids and per-position labels for every row, concatenated with an offset index."""

    ids: NDArray[np.int32]
    punctuation: NDArray[np.int8]
    case: NDArray[np.int8]
    offsets: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.offsets.size) - 1

    def example(self, index: int) -> tuple[NDArray[np.int32], NDArray[np.int8], NDArray[np.int8]]:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.ids[start:stop], self.punctuation[start:stop], self.case[start:stop]


def encode_words(
    words: tuple[str, ...], tokenizer: Tokenizer, max_length: int = MAX_LENGTH
) -> tuple[list[int], list[int]]:
    """`[CLS] … [SEP]`, and the position of each word's first subword.

    Words that would overflow `max_length` are dropped rather than truncated mid-word, so a
    returned position always indexes the start of a whole word. The caller reads how many
    words survived from the length of the second list.
    """
    ids = [CLS_ID]
    first: list[int] = []
    budget = max_length - 2

    for word in words:
        pieces = tokenizer.encode(word, out_type=int)
        if not pieces or len(ids) - 1 + len(pieces) > budget:
            break
        first.append(len(ids))
        ids.extend(pieces)

    ids.append(SEP_ID)
    return ids, first


def encode_annotation(
    annotation: Annotation, tokenizer: Tokenizer, max_length: int = MAX_LENGTH
) -> tuple[list[int], list[int], list[int]]:
    """Subword ids with each word's label on its first subword, `IGNORE_INDEX` elsewhere."""
    ids, first = encode_words(annotation.words, tokenizer, max_length)
    punctuation = [IGNORE_INDEX] * len(ids)
    case = [IGNORE_INDEX] * len(ids)
    for index, position in enumerate(first):
        punctuation[position] = annotation.punctuation[index]
        case[position] = annotation.case[index]
    return ids, punctuation, case


def encode_corpus(
    rows: list[Row], tokenizer: Tokenizer, max_length: int = MAX_LENGTH
) -> EncodedCorpus:
    ids: list[int] = []
    punctuation: list[int] = []
    case: list[int] = []
    offsets: list[int] = [0]

    for row in rows:
        row_ids, row_punctuation, row_case = encode_annotation(
            annotate(row.text), tokenizer, max_length
        )
        ids.extend(row_ids)
        punctuation.extend(row_punctuation)
        case.extend(row_case)
        offsets.append(len(ids))

    return EncodedCorpus(
        np.asarray(ids, dtype=np.int32),
        np.asarray(punctuation, dtype=np.int8),
        np.asarray(case, dtype=np.int8),
        np.asarray(offsets, dtype=np.int64),
    )


def label_counts(corpus: EncodedCorpus) -> tuple[dict[str, int], dict[str, int]]:
    """Counted off the encoded arrays, not the source rows, so truncation is reflected."""

    def tally(values: NDArray[np.int8], names: tuple[str, ...]) -> dict[str, int]:
        labelled = values[values != IGNORE_INDEX]
        counts = np.bincount(labelled.astype(np.int64), minlength=len(names))
        return {name: int(counts[index]) for index, name in enumerate(names)}

    return tally(corpus.punctuation, PUNCTUATION), tally(corpus.case, CASE)


@dataclass(frozen=True, slots=True)
class Batch:
    input_ids: Tensor
    attention_mask: Tensor
    punctuation: Tensor
    case: Tensor

    def to(self, device: torch.device) -> Batch:
        """Blocking, deliberately. These tensors wrap numpy buffers that are not pinned and
        are freed as soon as this returns, so an async copy reads memory that is already
        gone — on MPS that surfaces much later as an out-of-range index, not as a fault."""
        return Batch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.punctuation.to(device),
            self.case.to(device),
        )

    @property
    def labelled(self) -> int:
        return int((self.punctuation != IGNORE_INDEX).sum())


def collate(corpus: EncodedCorpus, indices: NDArray[np.int64]) -> Batch:
    """Pad to the longest row in the batch, never to `max_length`."""
    rows = [corpus.example(int(index)) for index in indices]
    width = max(row[0].size for row in rows)

    input_ids = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    attention = np.zeros((len(rows), width), dtype=bool)
    punctuation = np.full((len(rows), width), IGNORE_INDEX, dtype=np.int64)
    case = np.full((len(rows), width), IGNORE_INDEX, dtype=np.int64)

    for position, (row_ids, row_punctuation, row_case) in enumerate(rows):
        length = row_ids.size
        input_ids[position, :length] = row_ids
        attention[position, :length] = True
        punctuation[position, :length] = row_punctuation
        case[position, :length] = row_case

    return Batch(
        torch.from_numpy(input_ids),
        torch.from_numpy(attention),
        torch.from_numpy(punctuation),
        torch.from_numpy(case),
    )


def iter_batches(
    corpus: EncodedCorpus,
    batch_size: int,
    *,
    shuffle: bool,
    generator: np.random.Generator | None = None,
) -> list[NDArray[np.int64]]:
    """Length-bucketed batch indices.

    Sentences here average 10.5 subwords with a 92-subword tail, so batching at random pads
    to 3.18x the real token count and costs 6x the step time. Sorting by length first takes
    the padding to 1.01x; shuffling the order of the batches keeps the epoch stochastic, and
    breaking length ties at random keeps a row from meeting the same neighbours every epoch.
    """
    lengths = np.diff(corpus.offsets)
    if shuffle:
        if generator is None:
            msg = "shuffling needs a generator, so the epoch order is reproducible"
            raise ValueError(msg)
        order = np.lexsort((generator.random(len(corpus)), lengths))
    else:
        order = np.argsort(lengths, kind="stable")

    batches = [order[start : start + batch_size] for start in range(0, order.size, batch_size)]
    if shuffle and generator is not None:
        batches = [batches[index] for index in generator.permutation(len(batches))]
    return batches
