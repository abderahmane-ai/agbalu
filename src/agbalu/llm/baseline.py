"""Held-out likelihood of a causal language model (task 11.2).

The number every later Phase 11 claim has to beat, and the number the retention half of the
exit criterion is measured against. Both are computed the same way on three languages, so
"better at Kabyle" and "no worse at English and French" are one measurement, not two.

Bits per character is reported beside perplexity because perplexity per token is a property
of the tokenizer as much as of the model: task 11.12 is a vocabulary ablation, and a model
whose tokenizer emits fewer tokens per sentence scores a lower perplexity for free. Bits per
character is invariant to that, and is the only one of the two safe to compare across arms.

A leading token is prepended explicitly, because no tokenizer here adds one where it is
needed — measured, not assumed. A scoring loop that trusts `add_special_tokens` scores the
first real token with no context and reports a perplexity wrong in the direction of looking
bad. Which id that is comes from `leading_and_pad`, since the base carries no `<bos>` at all.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    import torch

MAX_LENGTH: Final = 512
"""The sequence length the CPT recipe trains at (`docs/llm_adaptation_design.md` §7)."""

BATCH_SIZE: Final = 8

MIN_WINDOW: Final = 2
"""`<bos>` plus one token, which is the shortest input with anything to predict."""

IGNORE_INDEX: Final = -100
"""What `cross_entropy` skips. Padding must not enter the sum, and it must not enter the
token count either — a padded token scored at any loss corrupts both the numerator and the
denominator of every figure in this module."""


class SpecialTokenError(Exception):
    """A checkpoint carries no token that can separate or pad documents."""


def leading_and_pad(bos_id: int | None, eos_id: int | None, pad_id: int | None) -> tuple[int, int]:
    """The ids to prefix a document with and to pad a batch with.

    The base carries no `<bos>` at all, and not every checkpoint does. Scoring
    still needs one token of context before the first real one, and `eos` is what a causal
    model has actually seen in that position, so it stands in for both. Refusing a missing
    `bos` instead fails on the container, after the checkpoint has been downloaded.
    """
    if eos_id is None:
        message = "no eos token, so documents can be neither separated nor padded"
        raise SpecialTokenError(message)
    return int(bos_id if bos_id is not None else eos_id), int(
        pad_id if pad_id is not None else eos_id
    )


LOSS_CHUNK_BYTES: Final = 256 * 1024 * 1024
"""How much fp32 logit surface to upcast at once, in bytes.

With a large vocabulary the logits — not the weights — are the memory bound, and the factor
is easy to miss because it does not scale with the model. The base's vocabulary is 248,320,
so a *single* token's fp32 logit row is 0.95 MiB and a batch of 8 × 385 tokens costs
2.9 GiB to upcast. Scoring the whole batch at once cost that twice — once for `.float()` and
again for the contiguous copy a transposed input forces — and was what put an A10G with
12 GiB free over the edge.

The loss is therefore accumulated over slices of this size, upcasting to fp32 only inside a
slice. fp32 is not negotiable: `log_softmax` over 248,320 classes in bf16 loses precision in
the figure being reported."""


class BaselineError(Exception):
    """A population cannot be scored, or was scored inconsistently."""


@dataclass(frozen=True, slots=True)
class Likelihood:
    """One document: summed negative log-likelihood in nats, and the tokens it covers."""

    nll: float
    tokens: int
    truncated: bool

    def __post_init__(self) -> None:
        if self.tokens < 1:
            message = f"a scored document has {self.tokens} tokens"
            raise BaselineError(message)


@dataclass(frozen=True, slots=True)
class Scored:
    """One evaluation set under one model."""

    name: str
    language: str
    documents: int
    tokens: int
    characters: int
    nll: float
    truncated: int

    def __post_init__(self) -> None:
        if self.tokens < 1 or self.characters < 1:
            message = f"{self.name}: {self.tokens} tokens over {self.characters} characters"
            raise BaselineError(message)

    @property
    def nll_per_token(self) -> float:
        return self.nll / self.tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll_per_token)

    @property
    def bits_per_character(self) -> float:
        """Comparable across tokenizers, unlike perplexity. Exact only while `truncated`
        is zero — a truncated document contributes all of its characters and only some of
        its tokens."""
        return self.nll / (math.log(2) * self.characters)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "language": self.language,
            "documents": self.documents,
            "tokens": self.tokens,
            "characters": self.characters,
            "nll": round(self.nll, 4),
            "truncated": self.truncated,
            "nll_per_token": round(self.nll_per_token, 6),
            "perplexity": round(self.perplexity, 4),
            "bits_per_character": round(self.bits_per_character, 6),
        }


def measure(
    name: str, language: str, texts: Sequence[str], likelihoods: Sequence[Likelihood]
) -> Scored:
    """Aggregate per-document likelihoods into one set's figures.

    The two sequences are paired positionally, so a mismatch is a bug that would otherwise
    surface as a plausible number over the wrong characters.
    """
    if len(texts) != len(likelihoods):
        message = f"{name}: {len(likelihoods)} likelihoods for {len(texts)} documents"
        raise BaselineError(message)
    if not texts:
        message = f"{name}: nothing to score"
        raise BaselineError(message)
    return Scored(
        name=name,
        language=language,
        documents=len(texts),
        tokens=sum(item.tokens for item in likelihoods),
        characters=sum(len(text) for text in texts),
        nll=math.fsum(item.nll for item in likelihoods),
        truncated=sum(1 for item in likelihoods if item.truncated),
    )


class CausalOutput(Protocol):
    logits: torch.Tensor


class CausalModel(Protocol):
    """The slice of a `transformers` causal LM this module drives."""

    def __call__(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> CausalOutput: ...


def batches(items: Sequence[list[int]], size: int) -> Iterator[Sequence[list[int]]]:
    if size < 1:
        message = f"batch size must be positive, got {size}"
        raise BaselineError(message)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def prepare(
    texts: Sequence[str], encode: Callable[[list[str]], list[list[int]]], bos_id: int, limit: int
) -> list[list[int]]:
    """Every document as ids, `<bos>`-prefixed and truncated to `limit`."""
    if limit < MIN_WINDOW:
        message = f"max length must leave at least one scored token, got {limit}"
        raise BaselineError(message)
    encoded = encode(list(texts))
    if len(encoded) != len(texts):
        message = f"the tokenizer returned {len(encoded)} rows for {len(texts)} documents"
        raise BaselineError(message)
    if any(not ids for ids in encoded):
        message = "a document encoded to no tokens"
        raise BaselineError(message)
    return [[bos_id, *ids][:limit] for ids in encoded]


def chunk_rows(vocabulary: int, budget: int = LOSS_CHUNK_BYTES) -> int:
    """How many token positions to upcast at once, given the vocabulary."""
    if vocabulary < 1:
        message = f"vocabulary must be positive, got {vocabulary}"
        raise BaselineError(message)
    return max(1, budget // (vocabulary * 4))


def summed_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-row summed negative log-likelihood, without upcasting the whole batch.

    Sliced one row at a time because `logits[row]` is contiguous and `[:-1]` on its first
    dimension keeps it so — the slice stays a view, where `logits[:, :-1]` would copy. Each
    slice is upcast to fp32 only as far as `chunk_rows` allows, and the sums accumulate on
    the device, so there is exactly one host synchronisation per batch rather than one per
    chunk.
    """
    import torch

    if logits.shape[0] != targets.shape[0]:
        message = f"{logits.shape[0]} logit rows against {targets.shape[0]} target rows"
        raise BaselineError(message)
    if logits.shape[1] - 1 != targets.shape[1]:
        message = (
            f"{logits.shape[1]} logit positions cannot score {targets.shape[1]} targets; "
            f"the targets are the inputs shifted by one, so this must be off by exactly one"
        )
        raise BaselineError(message)

    span = chunk_rows(int(logits.shape[-1]))
    totals = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    for row in range(logits.shape[0]):
        row_logits = logits[row, :-1]
        row_targets = targets[row]
        for start in range(0, row_targets.shape[0], span):
            stop = start + span
            totals[row] += torch.nn.functional.cross_entropy(
                row_logits[start:stop].float(),
                row_targets[start:stop],
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
    return totals


def likelihoods(
    texts: Sequence[str],
    encode: Callable[[list[str]], list[list[int]]],
    model: CausalModel,
    *,
    bos_id: int,
    pad_id: int,
    device: str = "cpu",
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
) -> list[Likelihood]:
    """Per-document negative log-likelihood, in input order.

    Padding is applied here rather than by the tokenizer so the ignored positions are the
    ones this function chose: the loss mask, the token count and the padding are one
    decision, and splitting them across two libraries is how a padded token ends up scored.
    """
    import torch

    prepared = prepare(texts, encode, bos_id, max_length)
    scored: list[Likelihood] = []
    for batch in batches(prepared, batch_size):
        width = max(len(ids) for ids in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = 1

        input_ids = input_ids.to(device)
        attention = attention.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits

        targets = input_ids[:, 1:].clone()
        targets[attention[:, 1:] == 0] = IGNORE_INDEX
        totals = summed_nll(logits, targets).tolist()
        counts = (targets != IGNORE_INDEX).sum(dim=1).tolist()
        scored.extend(
            Likelihood(nll=float(total), tokens=int(count), truncated=len(ids) >= max_length)
            for total, count, ids in zip(totals, counts, batch, strict=True)
        )

    if len(scored) != len(texts):
        message = f"{len(scored)} likelihoods from {len(texts)} documents"
        raise BaselineError(message)
    return scored
