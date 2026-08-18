"""Corpus to training tensors."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from agbalu.model.config import ModelError, TrainConfig
from agbalu.model.masking import SpanMasker
from agbalu.tokenizer.corpus import read_texts
from agbalu.tokenizer.evaluate import load
from agbalu.tokenizer.spec import CLS_ID, SEP_ID

log: Final = logging.getLogger("agbalu.model")

type BatchTensors = tuple[Tensor, Tensor, Tensor]
"""`(input_ids, attention_mask, labels)`, each `[rows, seq_length]`."""

TOKEN_DTYPE: Final = np.uint16
MAX_VOCAB: Final = 1 << 16

VALIDATION_SHARE: Final = 0.005
"""0.5% held out — ~15,000 sentences, enough for a stable validation loss and small
enough that the training set is untouched in practice."""

_HASH_SPACE: Final = 1 << 32


def _is_validation(sentence: str, share: float) -> bool:
    digest = hashlib.sha256(sentence.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / _HASH_SPACE < share


@dataclass(frozen=True, slots=True)
class TokenisedCorpus:
    train: Path
    validation: Path
    stats: Path

    def load_split(self, split: str) -> NDArray[np.uint16]:
        path = self.train if split == "train" else self.validation
        return np.fromfile(path, dtype=TOKEN_DTYPE)


def tokenise_corpus(
    corpus: Path,
    tokenizer_model: Path,
    out_dir: Path,
    *,
    validation_share: float = VALIDATION_SHARE,
) -> TokenisedCorpus:
    """Write `train.bin` and `validation.bin`, plus the counts that describe them."""
    processor = load(tokenizer_model)
    if processor.get_piece_size() > MAX_VOCAB:
        msg = f"vocabulary {processor.get_piece_size()} exceeds uint16; widen TOKEN_DTYPE"
        raise ModelError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: out_dir / f"{split}.bin" for split in ("train", "validation")}
    staging = {split: path.with_suffix(".bin.partial") for split, path in paths.items()}
    handles = {split: path.open("wb") for split, path in staging.items()}
    counts = {"train": 0, "validation": 0}
    sentences = {"train": 0, "validation": 0}

    try:
        for text in read_texts(corpus):
            split = "validation" if _is_validation(text, validation_share) else "train"
            pieces = processor.encode(text, out_type=int)
            if not pieces:
                continue
            row = np.asarray([*pieces, SEP_ID], dtype=TOKEN_DTYPE)
            handles[split].write(row.tobytes())
            counts[split] += int(row.size)
            sentences[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for split, path in paths.items():
        staging[split].replace(path)

    if counts["train"] == 0:
        msg = f"no training tokens produced from {corpus}"
        raise ModelError(msg)

    stats_path = out_dir / "tokenised.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "corpus": str(corpus),
                "tokenizer": str(tokenizer_model),
                "vocab_size": processor.get_piece_size(),
                "validation_share": validation_share,
                "sentences": sentences,
                "tokens": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info(
        "tokenised %s -> train %d tokens / %d sentences, validation %d / %d",
        corpus,
        counts["train"],
        sentences["train"],
        counts["validation"],
        sentences["validation"],
    )
    return TokenisedCorpus(paths["train"], paths["validation"], stats_path)


class PackedDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Fixed-length windows over the token stream, masked on access.

    Deterministic given `(seed, epoch, index)`: the window offset, the span lengths and
    the replacement draws are all derived from that triple, so a resumed run sees the
    same examples as an uninterrupted one.
    """

    def __init__(
        self,
        tokens: NDArray[np.uint16],
        config: TrainConfig,
        vocab_size: int,
        *,
        mask_token_id: int,
        n_special_tokens: int,
    ) -> None:
        if tokens.size <= config.seq_length:
            msg = f"stream of {tokens.size} tokens is shorter than one {config.seq_length} window"
            raise ModelError(msg)
        self.tokens = tokens
        self.config = config
        self.window = config.seq_length - 1
        self.masker = SpanMasker(
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            n_special_tokens=n_special_tokens,
            random_p=config.mask_random_p,
            keep_p=config.mask_keep_p,
            max_span_length=config.max_span_length,
        )
        self.step = 0

    def set_step(self, step: int) -> None:
        """Drives the inverse masking schedule. Called by the trainer, not the loader."""
        self.step = step

    def __len__(self) -> int:
        return int(self.tokens.size // self.window)

    def windows(self, indices: Tensor) -> Tensor:
        """`[len(indices), seq_length]`, each row `[CLS]` then a window of the stream.

        One fancy-index into the memmapped stream, not one slice per row.
        """
        starts = indices.numpy() * self.window
        offsets = np.arange(self.window)
        chunk = self.tokens[starts[:, None] + offsets].astype(np.int64)
        cls = np.full((chunk.shape[0], 1), CLS_ID, dtype=np.int64)
        return torch.from_numpy(np.concatenate((cls, chunk), axis=1))

    def batch(self, indices: Tensor, generator: torch.Generator) -> BatchTensors:
        ids = self.windows(indices)
        masked, labels = self.masker(
            ids, self.config.mask_probability(self.step), generator=generator
        )
        return masked, torch.ones_like(masked, dtype=torch.bool), labels

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        generator = torch.Generator().manual_seed(self.config.seed * 1_000_003 + index)
        ids, attention, labels = self.batch(torch.tensor([index]), generator)
        return ids[0], attention[0], labels[0]


def iter_batches(
    dataset: PackedDataset,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    start_batch: int = 0,
    drop_last: bool = True,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Shuffled batches, resumable at `start_batch` without replaying the skipped work.

    `drop_last` is True for training, where a ragged tail would perturb the gradient
    accumulation arithmetic, and False for validation, where dropping it can silently
    yield *no batches at all* on a set smaller than one batch — which reads downstream as
    an infinite validation loss rather than as an error.
    """
    if batch_size < 1:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ModelError(msg)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + epoch))
    total = len(dataset) // batch_size if drop_last else math.ceil(len(dataset) / batch_size)
    generator = torch.Generator()
    for batch in range(start_batch, total):
        indices = order[batch * batch_size : (batch + 1) * batch_size]
        if indices.numel():
            generator.manual_seed(seed * 1_000_003 + epoch * 7_919 + batch)
            yield dataset.batch(indices, generator)
