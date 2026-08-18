"""Linear-probe POS tagging with a frozen encoder (task 8.7).

A single `[hidden -> 17]` layer over the encoder's contextual vectors, trained on the
treebank's own train split and scored by `agbalu.bench.pos` — the same scorer that
produced every other number in task 7.5, so the systems are comparable.

The encoder is frozen throughout. What the probe measures is what pretraining already put
in the representation, not what a task head can learn given enough capacity; a fine-tuned
encoder would answer a different question and would not be comparable to the lexicon or
the most-frequent-tag floor.

A word's label is read at its **first subword**, as `taggers.NeuralTagger` does. The
treebank fixes the tokenisation under both settings, so a word that the vocabulary splits
still gets exactly one position and one label.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import torch
from torch import Tensor, nn

from agbalu.model.config import ModelConfig
from agbalu.tokenizer.spec import CLS_ID, SEP_ID
from agbalu.treebank import Sentence as TreebankSentence

log: Final = logging.getLogger("agbalu.bench")

UPOS_TAGS: Final[tuple[str, ...]] = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)
"""The 17 universal POS tags. Fixed by UD, not by this treebank, so a tag absent from
Kabyle still occupies its column and the confusion matrix stays comparable."""

TAG_ID: Final[dict[str, int]] = {tag: index for index, tag in enumerate(UPOS_TAGS)}
IGNORE_INDEX: Final = -100
MAX_PIECES: Final = 510
"""Leaves room for `[CLS]` and `[SEP]` inside the encoder's 512-position window."""


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """How the head is fitted. Nothing here touches the encoder, which stays frozen."""

    epochs: int = 20
    learning_rate: float = 2e-3
    batch_size: int = 32
    weight_decay: float = 0.01
    max_gradient: float = 1.0
    seed: int = 0
    """Fixes both the head's initialisation and the batch order, so a reported accuracy
    reproduces. Seeding only the second leaves ~0.5 points of run-to-run drift."""


class Tokenizer(Protocol):
    """The slice of the SentencePiece API this module drives, as in `model.infer`."""

    def encode(self, text: str, out_type: type[int]) -> list[int]: ...

    def unk_id(self) -> int: ...


class Encoder(Protocol):
    config: ModelConfig

    def contextualise(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...


Example = tuple[Tensor, list[int], list[int]]
"""(piece ids, first-subword position per word, label id per word)."""


def encode_words(
    words: Sequence[str], tokenizer: Tokenizer, *, max_pieces: int = MAX_PIECES
) -> tuple[Tensor, list[int]]:
    """Piece ids for a pre-split sentence, and where each word's first piece landed.

    A word the vocabulary cannot produce falls back to `unk`, so the returned positions
    always number one per word and stay alignable to gold.
    """
    pieces: list[int] = [CLS_ID]
    first: list[int] = []
    for word in words:
        subwords = tokenizer.encode(word, out_type=int) or [tokenizer.unk_id()]
        first.append(len(pieces))
        pieces.extend(subwords)
    pieces.append(SEP_ID)
    if len(pieces) > max_pieces + 2:
        pieces = pieces[: max_pieces + 2]
    return torch.tensor(pieces, dtype=torch.long), first


def examples_for(
    sentences: Sequence[TreebankSentence], tokenizer: Tokenizer
) -> tuple[list[Example], int]:
    """Training examples in the gold-words view, and how many sentences were skipped.

    A sentence whose word count and first-subword count disagree cannot be aligned, and is
    dropped rather than trained on a shifted labelling.
    """
    examples: list[Example] = []
    skipped = 0
    for sentence in sentences:
        words = [word.form for word in sentence.words]
        labels = [TAG_ID.get(word.upos, IGNORE_INDEX) for word in sentence.words]
        pieces, first = encode_words(words, tokenizer)
        if len(first) != len(words):
            skipped += 1
            continue
        examples.append((pieces, first, labels))
    return examples, skipped


class Head(nn.Module):
    """One linear layer. Deliberately the whole probe: depth here would measure the head."""

    def __init__(
        self, hidden_size: int, num_labels: int = len(UPOS_TAGS), *, seed: int = 0
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        # Xavier draws from the global RNG. Seeding only the batch order left the reported
        # accuracy varying by ~0.5 points between runs of the same checkpoint.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            weight = torch.empty_like(self.linear.weight, device="cpu")
            nn.init.xavier_uniform_(weight, generator=generator)
            self.linear.weight.copy_(weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        logits: Tensor = self.linear(hidden)
        return logits


def _batch(examples: Sequence[Example], device: torch.device) -> tuple[Tensor, Tensor]:
    width = max(pieces.shape[0] for pieces, _, _ in examples)
    ids = torch.zeros(len(examples), width, dtype=torch.long, device=device)
    mask = torch.zeros(len(examples), width, dtype=torch.bool, device=device)
    for row, (pieces, _, _) in enumerate(examples):
        length = pieces.shape[0]
        ids[row, :length] = pieces.to(device)
        mask[row, :length] = True
    return ids, mask


def _selected(
    hidden: Tensor, examples: Sequence[Example], device: torch.device
) -> tuple[Tensor, Tensor] | None:
    """First-subword vectors and their labels, dropping positions past the window."""
    vectors: list[Tensor] = []
    labels: list[int] = []
    for row, (_, first, gold) in enumerate(examples):
        for position, label in zip(first, gold, strict=True):
            if position < hidden.shape[1]:
                vectors.append(hidden[row, position])
                labels.append(label)
    if not vectors:
        return None
    return torch.stack(vectors), torch.tensor(labels, dtype=torch.long, device=device)


def train_head(
    encoder: Encoder,
    head: Head,
    examples: Sequence[Example],
    *,
    device: torch.device,
    config: ProbeConfig,
) -> list[float]:
    """Fit the head, returning mean loss per epoch. Only the head receives gradients."""
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    objective = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    order = list(range(len(examples)))
    rng = random.Random(config.seed)  # noqa: S311 — a fixed-seed shuffle, not a secret
    history: list[float] = []

    for epoch in range(1, config.epochs + 1):
        head.train()
        rng.shuffle(order)
        total, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            chunk = [examples[index] for index in order[start : start + config.batch_size]]
            ids, mask = _batch(chunk, device)
            with torch.no_grad():
                hidden = encoder.contextualise(ids, mask)
            picked = _selected(hidden, chunk, device)
            if picked is None:
                continue
            vectors, labels = picked
            loss = objective(head(vectors), labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), config.max_gradient)
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        history.append(total / max(batches, 1))
        log.info("probe epoch %2d/%d | loss %.4f", epoch, config.epochs, history[-1])
    return history


class ProbeTagger:
    """A frozen encoder and its trained head, behind `agbalu.bench.pos.Tagger`."""

    def __init__(
        self, encoder: Encoder, head: Head, tokenizer: Tokenizer, device: torch.device, step: int
    ) -> None:
        self._encoder = encoder
        self._head = head
        self._tokenizer = tokenizer
        self._device = device
        self._step = step

    @classmethod
    def fit(
        cls,
        encoder: Encoder,
        tokenizer: Tokenizer,
        sentences: Sequence[TreebankSentence],
        *,
        device: torch.device,
        step: int,
        config: ProbeConfig | None = None,
    ) -> ProbeTagger:
        examples, skipped = examples_for(sentences, tokenizer)
        if skipped:
            log.warning("probe skipped %d train sentences that could not be aligned", skipped)
        if not examples:
            message = "no alignable training sentences; the probe cannot be fitted"
            raise ValueError(message)
        settings = config or ProbeConfig()
        head = Head(encoder.config.hidden_size, seed=settings.seed).to(device)
        train_head(encoder, head, examples, device=device, config=settings)
        return cls(encoder, head, tokenizer, device, step)

    @property
    def name(self) -> str:
        return "encoder-probe"

    @property
    def revision(self) -> str:
        return f"step={self._step}"

    @torch.inference_mode()
    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]:
        self._head.eval()
        tagged: list[list[str | None]] = []
        for sentence in sentences:
            words = list(sentence)
            if not words:
                tagged.append([])
                continue
            pieces, first = encode_words(words, self._tokenizer)
            ids = pieces.unsqueeze(0).to(self._device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            hidden = self._encoder.contextualise(ids, mask)
            logits = self._head(hidden[0])
            width = hidden.shape[1]
            tagged.append(
                [UPOS_TAGS[int(logits[at].argmax())] if at < width else None for at in first]
            )
        return tagged
