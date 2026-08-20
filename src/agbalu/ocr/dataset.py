"""PyTorch Dataset and dynamic collator for document OCR training."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from agbalu.ocr.synthetic import image_to_tensor, render_text_line
from agbalu.ocr.vocabulary import encode

DEFAULT_MIN_CHARS: Final[int] = 15
DEFAULT_MAX_CHARS: Final[int] = 75
MIN_FONT_SIZE: Final[int] = 26
MAX_FONT_SIZE: Final[int] = 36
RENDER_SEED: Final[int] = 20_250_820
"""Fixes the un-augmented render. Any value works; it must not change between two runs
whose CER is compared."""


def chunk_text_into_lines(
    sentences: Sequence[str],
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    seed: int = RENDER_SEED,
) -> list[str]:
    """Slice long sentences or paragraphs into realistic single document text lines.

    Seeded, because the line boundaries decide the train/holdout split: on the module
    generator the same corpus chunks differently on every call, so a resumed run holds out
    different lines from the one it is continuing.
    """
    rng = random.Random(seed)
    lines: list[str] = []
    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue

        current_line: list[str] = []
        current_len = 0
        target_len = rng.randint(min_chars, max_chars)

        for word in words:
            if current_len + len(word) + 1 > target_len and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
                target_len = rng.randint(min_chars, max_chars)
            else:
                current_line.append(word)
                current_len += len(word) + 1

        if current_line:
            lines.append(" ".join(current_line))

    return lines


def load_corpus_sentences(file_path: str | Path, max_sentences: int | None = None) -> list[str]:
    """Load raw sentences from JSONL, TXT, or Parquet corpus file."""
    path = Path(file_path)
    if not path.is_file():
        return []

    sentences: list[str] = []
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        col_names = table.column_names
        target_col = (
            "text_tfng"
            if "text_tfng" in col_names
            else ("text_latn" if "text_latn" in col_names else col_names[0])
        )
        data = table[target_col].to_pylist()
        for item in data:
            if item and str(item).strip():
                sentences.append(str(item).strip())
                if max_sentences and len(sentences) >= max_sentences:
                    break
        return sentences

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                    val = obj.get(
                        "text",
                        obj.get("kab", obj.get("text_latn", obj.get("text_tfng", s))),
                    )
                    text = str(val).strip()
                    if text:
                        sentences.append(text)
                except json.JSONDecodeError:
                    # A line that opens with `{` and does not parse is a malformed record,
                    # not a sentence. Appending it puts raw JSON into the training targets.
                    continue
            else:
                sentences.append(s)
            if max_sentences and len(sentences) >= max_sentences:
                break
    return sentences


def build_dual_script_lines(
    latin_sentences: Sequence[str],
    tifinagh_sentences: Sequence[str],
    tifinagh_ratio: float = 0.25,
    max_lines: int | None = None,
    seed: int = 17,
) -> list[str]:
    """Create a balanced mixed dual-script line dataset (default: 75% Latin, 25% Tifinagh)."""
    latin_lines = chunk_text_into_lines(latin_sentences, seed=seed)
    tifinagh_lines = chunk_text_into_lines(tifinagh_sentences, seed=seed)

    rng = random.Random(seed)
    rng.shuffle(latin_lines)
    rng.shuffle(tifinagh_lines)

    if not tifinagh_lines or tifinagh_ratio <= 0.0:
        return latin_lines[:max_lines] if max_lines else latin_lines

    if not latin_lines:
        return tifinagh_lines[:max_lines] if max_lines else tifinagh_lines

    target_total = (
        min(max_lines, len(latin_lines) + len(tifinagh_lines))
        if max_lines
        else (len(latin_lines) + len(tifinagh_lines))
    )
    n_tif = min(int(target_total * tifinagh_ratio), len(tifinagh_lines))
    n_lat = min(target_total - n_tif, len(latin_lines))

    selected_lat = latin_lines[:n_lat]
    selected_tif = tifinagh_lines[:n_tif]

    combined = selected_lat + selected_tif
    rng.shuffle(combined)
    return combined


class SyntheticDataset(Dataset[tuple[torch.Tensor, list[int], str]]):
    """Renders document line images on demand.

    With `augment=False` the render is keyed on `seed + index`, so a held-out line is the
    same image at every evaluation and a CER difference between two checkpoints is the
    model's. With `augment=True` it draws from the module generator, which the dataloader
    seeds per worker per epoch — that variation is the point.
    """

    def __init__(
        self,
        lines: Sequence[str],
        augment: bool = True,
        seed: int = RENDER_SEED,
    ) -> None:
        self.lines = list(lines)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int], str]:
        text = self.lines[idx]
        rng = None if self.augment else random.Random(self.seed + idx)
        font_size = (rng or random).randint(MIN_FONT_SIZE, MAX_FONT_SIZE)

        img = render_text_line(text, font_size=font_size, augment=self.augment, rng=rng)
        pixel_values = image_to_tensor(img)
        token_ids = encode(text, add_special_tokens=True)
        return pixel_values, token_ids, text


def collate_lines(
    batch: list[tuple[torch.Tensor, list[int], str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate function padding variable-length label sequences with -100."""
    pixel_tensors = [item[0] for item in batch]
    token_sequences = [item[1] for item in batch]
    texts = [item[2] for item in batch]

    pixel_values = torch.stack(pixel_tensors, dim=0)

    # Pad labels to max sequence length in batch
    max_len = max(len(seq) for seq in token_sequences)
    padded_labels = torch.full((len(token_sequences), max_len), -100, dtype=torch.long)

    for i, seq in enumerate(token_sequences):
        padded_labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "labels": padded_labels,
        "texts": texts,
    }
