"""Span masking, following `ltgoslo/gpt-bert` `pretraining/dataset.py`."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Generator, Tensor

from agbalu.model.config import ModelError
from agbalu.model.modeling import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class SpanMasker:
    vocab_size: int
    mask_token_id: int
    n_special_tokens: int
    random_p: float
    keep_p: float
    max_span_length: int

    def __post_init__(self) -> None:
        if self.max_span_length < 1:
            msg = f"max_span_length must be positive, got {self.max_span_length}"
            raise ModelError(msg)
        if self.n_special_tokens >= self.vocab_size:
            msg = "n_special_tokens must leave at least one ordinary piece"
            raise ModelError(msg)
        if not 0.0 <= self.random_p + self.keep_p <= 1.0:
            msg = "random_p + keep_p must be a proportion"
            raise ModelError(msg)

    def span_indices(self, rows: int, length: int, generator: Generator) -> Tensor:
        """A span id per position, spans being 1..max_span_length long. Shape `[rows, length]`.

        Span `k` covers `[cumulative[k-1], cumulative[k])`, so the id at a position is the
        number of span ends at or before it — a batched `searchsorted`, which replaces the
        per-row scatter-and-cummax that made this the training loop's bottleneck.
        """
        lengths = torch.randint(
            1, self.max_span_length + 1, (rows, length), dtype=torch.long, generator=generator
        )
        cumulative = lengths.cumsum(1)
        positions = torch.arange(length, dtype=torch.long).expand(rows, length).contiguous()
        return torch.searchsorted(cumulative, positions, right=True)

    def __call__(
        self, tokens: Tensor, mask_probability: float, *, generator: Generator
    ) -> tuple[Tensor, Tensor]:
        """Mask a `[rows, length]` batch, or a single `[length]` window.

        Returns `(input_ids, labels)` with `IGNORE_INDEX` wherever nothing was masked.
        Every operation is over the whole batch: no `.item()`, no per-row Python.
        """
        if tokens.ndim not in (1, 2):
            msg = f"expected a 1-D window or a 2-D batch, got shape {tuple(tokens.shape)}"
            raise ModelError(msg)
        flat = tokens.ndim == 1
        batch = tokens.unsqueeze(0) if flat else tokens
        rows, length = int(batch.size(0)), int(batch.size(1))

        indices = self.span_indices(rows, length, generator)
        ratios = torch.rand((rows, length), generator=generator).gather(1, indices)
        replacement_draw = torch.rand((rows, length), generator=generator).gather(1, indices)

        protected = batch < self.n_special_tokens
        ratios = ratios.masked_fill(protected, float("inf"))

        wanted = max(int(length * mask_probability), 1)
        threshold = ratios.topk(wanted, dim=1, largest=False).values[:, -1:]
        masked = (ratios <= threshold) & ~protected
        """`& ~protected` also covers `wanted` exceeding the eligible count: the surplus
        selects `inf` ratios, which are exactly the protected positions."""

        random_ids = torch.randint(
            self.n_special_tokens,
            self.vocab_size,
            (rows, length),
            dtype=batch.dtype,
            generator=generator,
        )
        inputs = torch.where(masked & (replacement_draw < self.random_p), random_ids, batch)
        inputs = torch.where(
            masked & (replacement_draw > self.random_p + self.keep_p),
            torch.full_like(batch, self.mask_token_id),
            inputs,
        )
        labels = torch.where(masked, batch, torch.full_like(batch, IGNORE_INDEX))
        return (inputs.squeeze(0), labels.squeeze(0)) if flat else (inputs, labels)
