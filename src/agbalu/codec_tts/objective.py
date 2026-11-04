"""Compute token-summed codec loss without weighting short microbatches equally."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional

from agbalu.codec_tts.layout import CODEBOOKS, IGNORE, VOCABULARY


@dataclass(frozen=True)
class CodecLoss:
    stream_nats: Tensor
    stream_counts: Tensor

    @property
    def numerator(self) -> Tensor:
        return self.stream_nats.sum()

    @property
    def count(self) -> Tensor:
        return self.stream_counts.sum()


def codec_loss(logits: Tensor, labels: Tensor) -> CodecLoss:
    if logits.shape[:-1] != labels.shape or logits.shape[-2:] != (CODEBOOKS, VOCABULARY):
        message = (
            f"codec loss shape mismatch logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
        raise ValueError(message)
    valid = labels.ne(IGNORE)
    if not bool(valid.any().item()):
        message = "codec loss has zero supervised targets"
        raise ValueError(message)
    losses = functional.cross_entropy(
        logits.float().reshape(-1, VOCABULARY),
        labels.reshape(-1),
        ignore_index=IGNORE,
        reduction="none",
    ).reshape(labels.shape)
    axes = tuple(range(labels.ndim - 1))
    return CodecLoss(losses.sum(dim=axes), valid.sum(dim=axes))
