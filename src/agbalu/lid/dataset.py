"""Reading the built splits, byte encoding, and the group-balanced sampler.

The sampler draws `(language, domain)` cells, not classes. Drawing classes over a
corpus where each class arrives from one source balances the source and reports it as
the language, and drawing rows balances neither: 5,000 steps of nine-per-class made
21.6 passes over Tamazight's 2,080 rows and 0.52 over Kabyle's 86,247, so the tail was
memorised and the head was never finished.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import torch
from torch import Tensor

from agbalu.lid.config import BYTE_OFFSET, PAD_ID, ModelConfig

MIN_AUGMENT_WORDS: Final = 4
MIN_PHRASE_WORDS: Final = 2


class CorpusError(Exception):
    """A split that cannot be read, or that carries no usable row."""


def encode(text: str, max_length: int) -> list[int]:
    """UTF-8 bytes shifted past the two reserved ids, truncated to `max_length`.

    Truncation is on bytes and may cut a multi-byte character in half. That is left
    alone deliberately: a partial `ɣ` is a real input at inference, where text arrives
    already truncated, and teaching the model that its last byte is always complete
    would be teaching it something false.
    """
    return [b + BYTE_OFFSET for b in text.encode("utf-8")[:max_length]]


@dataclass(frozen=True, slots=True)
class Augmentation:
    """How a training row is perturbed before it is encoded.

    One value rather than three parameters, because the three only ever travel
    together and a corpus is either augmented or it is not.
    """

    phrase_prob: float = 0.0
    case_prob: float = 0.0
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.phrase_prob or self.case_prob)


@dataclass(frozen=True, slots=True)
class Record:
    """One row of a built split."""

    text: str
    language: str
    domain: str
    group: str
    source: str


@dataclass(frozen=True, slots=True)
class Batch:
    """A padded batch, with the label and group index of every row."""

    input_ids: Tensor
    pad_mask: Tensor
    labels: Tensor
    groups: Tensor

    def to(self, device: torch.device) -> Self:
        return type(self)(
            input_ids=self.input_ids.to(device, non_blocking=True),
            pad_mask=self.pad_mask.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            groups=self.groups.to(device, non_blocking=True),
        )

    def __len__(self) -> int:
        return int(self.labels.numel())


def read_split(path: Path) -> list[Record]:
    """Read one built split, keeping the provenance the sampler groups on."""
    if not path.is_file():
        msg = f"split not found: {path} — run `make lid-dataset`"
        raise CorpusError(msg)
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            language, domain = row["language"], row.get("domain", "unknown")
            records.append(
                Record(
                    text=row["text"],
                    language=language,
                    domain=domain,
                    group=row.get("group", f"{language}/{domain}"),
                    source=row.get("source", "unknown"),
                )
            )
    if not records:
        msg = f"{path} carried no rows"
        raise CorpusError(msg)
    return records


class Corpus:
    """Encoded rows plus the label and group indices the sampler and the losses need.

    `groups` is fixed by the training corpus and passed to every other split, so a
    group that only the dev split contains cannot silently shift every index by one.
    """

    def __init__(
        self,
        records: Sequence[Record],
        config: ModelConfig,
        *,
        groups: Sequence[str] | None = None,
        augmentation: Augmentation | None = None,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.augmentation = augmentation or Augmentation()
        self.rng = random.Random(self.augmentation.seed)
        self.classes = tuple(config.classes)
        self.groups = tuple(groups) if groups is not None else self.observed_groups(records)

        class_index = {name: i for i, name in enumerate(self.classes)}
        group_index = {name: i for i, name in enumerate(self.groups)}
        unknown = sorted({record.language for record in self.records} - set(class_index))
        if unknown:
            msg = f"rows carry labels outside the class list: {', '.join(unknown)}"
            raise CorpusError(msg)
        self.labels = [class_index[record.language] for record in self.records]
        self.group_ids = [
            group_index[record.group if record.group in group_index else self._fallback(record)]
            for record in self.records
        ]

    @staticmethod
    def observed_groups(records: Sequence[Record]) -> tuple[str, ...]:
        return tuple(sorted({record.group for record in records}))

    def _fallback(self, record: Record) -> str:
        """A group the training corpus never saw maps to any group of the same language.

        Dev and the held-out set are cut from the same build as train, so this is a
        guard rather than a path: it fires only when a split is rebuilt against a
        different source table, and mapping to the language keeps group-DRO meaningful
        instead of crashing an evaluation.
        """
        for name in self.groups:
            if name.startswith(f"{record.language}/"):
                return name
        msg = f"no group for language {record.language!r} in {self.groups}"
        raise CorpusError(msg)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def class_counts(self) -> list[int]:
        counts = Counter(self.labels)
        return [counts.get(index, 0) for index in range(len(self.classes))]

    @property
    def group_counts(self) -> list[int]:
        counts = Counter(self.group_ids)
        return [counts.get(index, 0) for index in range(len(self.groups))]

    def _augmented(self, text: str) -> str:
        """A random phrase of the line, and sometimes lowercased.

        Both fight a length and shape cue rather than adding data: interface strings
        are four words and scripture is fifteen, so a model given whole lines can read
        the domain off the length and the domain names the language.
        """
        words = text.split()
        if len(words) > MIN_AUGMENT_WORDS and self.rng.random() < self.augmentation.phrase_prob:
            span = self.rng.randint(MIN_PHRASE_WORDS, len(words))
            start = self.rng.randint(0, len(words) - span)
            text = " ".join(words[start : start + span])
        if self.rng.random() < self.augmentation.case_prob:
            text = text.lower()
        return text

    def encoded(self, index: int) -> list[int]:
        text = self.records[index].text
        if self.augmentation.enabled:
            text = self._augmented(text)
        return encode(text, self.config.max_length)

    def collate(self, indices: Sequence[int]) -> Batch:
        """Pad to the longest row in the batch, never to `max_length`."""
        rows = [self.encoded(index) or [PAD_ID] for index in indices]
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        pad_mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for position, row in enumerate(rows):
            input_ids[position, : len(row)] = torch.tensor(row, dtype=torch.long)
            pad_mask[position, : len(row)] = True
        return Batch(
            input_ids=input_ids,
            pad_mask=pad_mask,
            labels=torch.tensor([self.labels[i] for i in indices], dtype=torch.long),
            groups=torch.tensor([self.group_ids[i] for i in indices], dtype=torch.long),
        )


def group_batches(
    corpus: Corpus, *, groups_per_batch: int, samples_per_group: int, batches: int, seed: int
) -> Iterator[list[int]]:
    """Batches of `groups_per_batch` cells, `samples_per_group` rows each, with replacement.

    Cells are drawn without replacement within a batch so a batch never contrasts a
    group with itself, and the row draw is with replacement because the smallest cell
    holds nine rows.
    """
    if groups_per_batch < 1 or samples_per_group < 1:
        msg = f"batch shape must be positive, got {groups_per_batch}x{samples_per_group}"
        raise CorpusError(msg)
    members: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(corpus.group_ids):
        members[group].append(index)
    populated = sorted(members)
    if not populated:
        msg = "corpus has no populated groups"
        raise CorpusError(msg)
    width = min(groups_per_batch, len(populated))
    rng = random.Random(seed)
    for _ in range(batches):
        chosen = rng.sample(populated, width)
        batch = [rng.choice(members[group]) for group in chosen for _ in range(samples_per_group)]
        rng.shuffle(batch)
        yield batch


def sequential_batches(corpus: Corpus, batch_size: int) -> Iterator[list[int]]:
    """Every row once, longest-first within a batch, for evaluation.

    Sorted by length so padding does not dominate: unsorted, the dev pass over a corpus
    whose rows run from four words to a hundred and twenty pays for the longest row in
    every batch.
    """
    order = sorted(range(len(corpus)), key=lambda index: len(corpus.records[index].text))
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]
