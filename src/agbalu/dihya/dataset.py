"""Dataset collation and cell-balanced group sampling for Dihya-22M.

Converts arbitrary UTF-8 text directly into byte token sequences without subword
tokenization and provides stratified sampling across (variety, domain, script) cells.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from agbalu.dihya.config import BYTE_OFFSET, PAD_ID, ModelConfig
from agbalu.dihya.taxonomy import (
    COMPOSITE_CLASSES,
    SCRIPTS,
    is_amazigh_label,
    script_error,
)

LABEL_TO_ID: Final[dict[str, int]] = {cls: idx for idx, cls in enumerate(COMPOSITE_CLASSES)}
SCRIPT_TO_ID: Final[dict[str, int]] = {script: idx for idx, script in enumerate(SCRIPTS)}


@dataclass(frozen=True, slots=True)
class Example:
    """A single annotated text record."""

    text: str
    label: str
    domain: str
    source: str
    script: str = ""

    def __post_init__(self) -> None:
        fields = (self.text, self.label, self.domain, self.source, self.script)
        if not all(isinstance(value, str) for value in fields):
            msg = f"Example fields must be strings, got {fields!r}"
            raise TypeError(msg)
        error = script_error(self.text, self.label, self.script)
        if error or not self.domain or not self.source:
            msg = f"source={self.source!r} label={self.label!r}: {error or 'missing metadata'}"
            raise ValueError(msg)

    @property
    def group_key(self) -> str:
        """The (variety, domain, script) cell identifier."""
        return f"{self.label}/{self.domain}"


@dataclass(frozen=True, slots=True)
class Batch:
    """Collation tensor batch for Dihya-22M."""

    input_ids: Tensor
    pad_mask: Tensor
    labels: Tensor
    script_labels: Tensor
    filter_labels: Tensor
    domains: Tensor
    groups: Tensor


def text_to_bytes(text: str, max_length: int) -> list[int]:
    """Convert unicode string to byte token IDs with offset and truncation.

    Args:
        text: Input string.
        max_length: Maximum allowed bytes.

    Returns:
        List of integer token IDs.
    """
    raw_bytes = text.encode("utf-8")[:max_length]
    if max_length < 1:
        msg = f"max_length must be positive, got {max_length}"
        raise ValueError(msg)
    return [b + BYTE_OFFSET for b in raw_bytes]


class TextDataset(Dataset[Example]):
    """Memory-mapped or list-backed text dataset for language identification."""

    def __init__(self, examples: Sequence[Example], max_length: int = 512) -> None:
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    @classmethod
    def load_jsonl(cls, path: Path, max_length: int = 512) -> TextDataset:
        """Load examples from a JSONL file."""
        examples: list[Example] = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    example = Example(
                        text=row["text"],
                        label=row["label"],
                        domain=row["domain"],
                        source=row["source"],
                        script=row.get("script", ""),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    msg = f"{path}:{line_number}: {error}"
                    raise ValueError(msg) from error
                examples.append(example)
        return cls(examples, max_length=max_length)

    @classmethod
    def load(cls, directory: Path, split: str, max_length: int = 512) -> TextDataset:
        """Load the committed Parquet split and verify its content digest."""
        from agbalu.dihya.prepare import read_split

        return cls(read_split(directory, split), max_length=max_length)


class CellSampler(Sampler[list[int]]):
    """Stratified sampler drawing balanced groups of (variety, domain, script) cells."""

    def __init__(
        self,
        dataset: TextDataset,
        groups_per_batch: int,
        samples_per_group: int,
        seed: int = 20261101,
        min_group_support: int = 20,
    ) -> None:
        super().__init__()
        if min(groups_per_batch, samples_per_group, len(dataset)) < 1:
            msg = "CellSampler requires a nonempty dataset and positive batch counts"
            raise ValueError(msg)
        self.groups_per_batch = groups_per_batch
        self.samples_per_group = samples_per_group
        self.rng = random.Random(seed)

        raw_cells: dict[str, list[int]] = defaultdict(list)
        for idx, ex in enumerate(dataset.examples):
            raw_cells[ex.group_key].append(idx)
        largest: dict[str, str] = {}
        for cell in sorted(raw_cells):
            label = dataset[raw_cells[cell][0]].label
            if label not in largest or len(raw_cells[cell]) > len(raw_cells[largest[label]]):
                largest[label] = cell
        self.cell_aliases: dict[str, str] = {}
        self.cell_indices: dict[str, list[int]] = defaultdict(list)
        for cell, indices in raw_cells.items():
            label = dataset[indices[0]].label
            target = largest[label] if len(indices) < min_group_support else cell
            self.cell_aliases[cell] = target
            self.cell_indices[target].extend(indices)

        self.cells = sorted(self.cell_indices.keys())
        self.total_samples = len(dataset)
        self.batch_size = min(groups_per_batch, len(self.cells)) * samples_per_group
        self.num_batches = max(1, (self.total_samples + self.batch_size - 1) // self.batch_size)

    def set_epoch(self, epoch: int, seed: int) -> None:
        """Replay an epoch without depending on earlier iterator consumption."""
        self.rng.seed(seed + epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_batches):
            k = min(self.groups_per_batch, len(self.cells))
            selected_cells = self.rng.sample(self.cells, k)
            batch_indices: list[int] = []
            for cell in selected_cells:
                pool = self.cell_indices[cell]
                if len(pool) >= self.samples_per_group:
                    batch_indices.extend(self.rng.sample(pool, self.samples_per_group))
                else:
                    batch_indices.extend(self.rng.choices(pool, k=self.samples_per_group))
            yield batch_indices


def collate_examples(
    examples: Sequence[Example],
    config: ModelConfig,
    domain_map: dict[str, int],
    group_map: dict[str, int],
) -> Batch:
    """Collate and dynamically pad examples into batch tensors.

    Args:
        examples: Sequence of text examples.
        config: Model configuration.
        domain_map: Mapping from domain string to integer ID.
        group_map: Mapping from group key to integer ID.

    Returns:
        Collation Batch instance.
    """
    token_lists = [text_to_bytes(ex.text, config.max_length) for ex in examples]
    max_len = max(len(toks) for toks in token_lists)
    max_len = max(max_len, 1)

    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    pad_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    labels = torch.empty(batch_size, dtype=torch.long)
    script_labels = torch.empty(batch_size, dtype=torch.long)
    filter_labels = torch.empty(batch_size, dtype=torch.float)
    domains = torch.empty(batch_size, dtype=torch.long)
    groups = torch.empty(batch_size, dtype=torch.long)

    for i, (ex, toks) in enumerate(zip(examples, token_lists, strict=True)):
        input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        pad_mask[i, : len(toks)] = True

        labels[i] = config.classes.index(ex.label)
        script = ex.label.rsplit("_", 1)[1]
        script_labels[i] = config.scripts.index(script)
        filter_labels[i] = 1.0 if is_amazigh_label(ex.label) else 0.0

        domains[i] = domain_map[ex.domain] if domain_map else 0
        groups[i] = group_map[ex.group_key] if group_map else 0

    return Batch(
        input_ids=input_ids,
        pad_mask=pad_mask,
        labels=labels,
        script_labels=script_labels,
        filter_labels=filter_labels,
        domains=domains,
        groups=groups,
    )
