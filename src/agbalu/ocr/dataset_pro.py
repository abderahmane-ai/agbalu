"""Dataset definitions, reservoir sampling, and bucketed collators for Feraoun-Pro."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from agbalu.ocr.synthetic_pro import LineRenderer
from agbalu.ocr.vocabulary_pro import PAD_ID, encode_aux, encode_ctc

LABEL_IGNORE_INDEX: Final[int] = -100


@dataclass(frozen=True, slots=True)
class BatchOCRPro:
    pixel_values: Tensor  # (B, 3, 64, W_max)
    input_lengths: Tensor  # (B,) CTC sequence length (W // 4)
    ctc_targets: Tensor  # (B, L_max) or flat 1D tensor
    target_lengths: Tensor  # (B,) lengths of ctc_targets
    aux_input_ids: Tensor  # (B, L_aux) shifted right for teacher forcing
    aux_labels: Tensor  # (B, L_aux) with LABEL_IGNORE_INDEX
    texts: tuple[str, ...]


def mix_dual_script(
    latin: Sequence[str],
    tifinagh: Sequence[str],
    tifinagh_ratio: float,
    max_lines: int | None = None,
    seed: int = 17,
) -> list[str]:
    """Interleave Latin and Neo-Tifinagh sentences at `tifinagh_ratio`, shuffled together.

    Sentences, not pre-chunked lines: `SyntheticOCRDatasetPro.__getitem__` draws a
    variable-length substring per sample, so chunking here would be sampled twice.

    Mixing happens before the train/validation split, so both sides carry the same script
    proportions. A validation set drawn after a Latin-only mix has no Tifinagh support, and a
    per-script rate over an empty bucket is not a measurement.
    """
    rng = random.Random(seed)
    latin_pool = list(latin)
    tifinagh_pool = list(tifinagh)
    rng.shuffle(latin_pool)
    rng.shuffle(tifinagh_pool)

    if tifinagh_ratio <= 0.0 or not tifinagh_pool:
        return latin_pool[:max_lines] if max_lines else latin_pool
    if not latin_pool:
        return tifinagh_pool[:max_lines] if max_lines else tifinagh_pool

    available = len(latin_pool) + len(tifinagh_pool)
    total = min(max_lines, available) if max_lines else available
    wanted_tifinagh = min(round(total * tifinagh_ratio), len(tifinagh_pool))
    wanted_latin = min(total - wanted_tifinagh, len(latin_pool))

    combined = latin_pool[:wanted_latin] + tifinagh_pool[:wanted_tifinagh]
    rng.shuffle(combined)
    return combined


def sample_line_substring(text: str, rng: random.Random) -> str:
    """Sample a short, typical, or long substring to simulate natural line variations."""
    cleaned = " ".join(text.split()).strip()
    n = len(cleaned)
    if n <= 25:
        return cleaned

    r_val = rng.random()
    if r_val < 0.15:
        # Short (5 to min(25, n))
        target_len = rng.randint(5, min(25, n))
    elif r_val < 0.85:
        # Typical (25 to min(90, n))
        max_bound = min(90, n)
        min_bound = min(25, max_bound)
        target_len = rng.randint(min_bound, max_bound)
    else:
        # Long (min(90, n) to min(140, n))
        max_bound = min(140, n)
        min_bound = min(90, max_bound)
        target_len = rng.randint(min_bound, max_bound)

    if n <= target_len:
        return cleaned

    start_max = n - target_len
    start_pos = rng.randint(0, start_max)
    sub = cleaned[start_pos : start_pos + target_len].strip()
    return sub if sub else cleaned[:target_len]


class SyntheticOCRDatasetPro(Dataset[tuple[Image.Image, str]]):
    """Generates synthetic rendered line samples online from a source text corpus.

    With `augment=False` the substring, typeface and degradation are keyed on `seed + index`,
    so a held-out line is the same image at every evaluation and a CER difference between two
    checkpoints is the model's rather than the renderer's.

    With `augment=True` the draw comes from the module generator, which the dataloader seeds
    per worker per epoch. An index-keyed generator would hand epoch 2 the identical image it
    rendered in epoch 1, which is a training set of one crop per line however many epochs are
    paid for.
    """

    def __init__(
        self,
        texts: Sequence[str],
        renderer: LineRenderer | None = None,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.texts = list(texts)
        self.renderer = renderer or LineRenderer()
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        rng = None if self.augment else random.Random(self.seed + index)
        raw_text = self.texts[index]
        line_text = sample_line_substring(raw_text, rng or random.Random())
        img, final_text = self.renderer.render_line(
            line_text, rng=rng, apply_augmentation=self.augment
        )
        return img, final_text


class BucketedBatchSampler(Sampler[list[int]]):
    """Groups lines of similar character length together to minimize right-padding waste."""

    def __init__(
        self, texts: Sequence[str], batch_size: int, seed: int = 42, shuffle: bool = True
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Sort indices by text length
        indexed_lengths = [(i, len(t)) for i, t in enumerate(texts)]
        indexed_lengths.sort(key=lambda x: x[1])
        self.sorted_indices = [idx for idx, _ in indexed_lengths]

    def __len__(self) -> int:
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        batches = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)

        self.epoch += 1
        yield from batches


class CollateOCRPro:
    """Collates variable-width line images and targets into a BatchOCRPro."""

    def __call__(self, batch: list[tuple[Image.Image, str]]) -> BatchOCRPro:
        images = [item[0] for item in batch]
        texts = tuple(item[1] for item in batch)

        # 1. Determine maximum canvas width in the batch (divisible by 4 for CNN stem)
        max_w = max(img.width for img in images)
        max_w = int(np.ceil(max_w / 4.0) * 4)

        b_size = len(images)
        tensors: list[Tensor] = []
        input_lengths: list[int] = []

        for img in images:
            w = img.width
            arr = np.array(img, dtype=np.float32) / 255.0  # (64, W, 3)
            # Normalize to [-1, 1]
            arr = (arr - 0.5) / 0.5
            ch_tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 64, W)

            # Pad right
            pad_w = max_w - w
            if pad_w > 0:
                padded = torch.full((3, 64, max_w), 1.0, dtype=torch.float32)  # white padding
                padded[:, :, :w] = ch_tensor
                tensors.append(padded)
            else:
                tensors.append(ch_tensor)

            # CTC output sequence length is W // 4
            input_lengths.append(w // 4)

        pixel_values = torch.stack(tensors, dim=0)
        input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)

        # 2. Encode CTC and auxiliary targets
        ctc_encoded = [encode_ctc(t) for t in texts]
        target_lengths = torch.tensor([len(c) for c in ctc_encoded], dtype=torch.long)

        max_ctc_len = max(1, *(len(c) for c in ctc_encoded))
        ctc_padded = torch.full((b_size, max_ctc_len), PAD_ID, dtype=torch.long)
        for i, c in enumerate(ctc_encoded):
            if c:
                ctc_padded[i, : len(c)] = torch.tensor(c, dtype=torch.long)

        # Auxiliary targets: <s> ... </s>
        aux_encoded = [encode_aux(t) for t in texts]
        max_aux_len = max(len(a) for a in aux_encoded)
        aux_input_ids = torch.full((b_size, max_aux_len), PAD_ID, dtype=torch.long)
        aux_labels = torch.full((b_size, max_aux_len), LABEL_IGNORE_INDEX, dtype=torch.long)

        for i, a in enumerate(aux_encoded):
            a_tensor = torch.tensor(a, dtype=torch.long)
            aux_input_ids[i, : len(a)] = a_tensor
            aux_labels[i, : len(a)] = a_tensor

        return BatchOCRPro(
            pixel_values=pixel_values,
            input_lengths=input_lengths_tensor,
            ctc_targets=ctc_padded,
            target_lengths=target_lengths,
            aux_input_ids=aux_input_ids,
            aux_labels=aux_labels,
            texts=texts,
        )


def load_corpus_lines(path: str | Path, max_lines: int | None = None) -> list[str]:
    """Load text lines from a JSONL, text, or Parquet file without heavy dependency imports."""
    p = Path(path)
    if not p.is_file():
        return []

    lines: list[str] = []
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(p)
        col_names = table.column_names
        target_col = (
            "text_tfng"
            if "text_tfng" in col_names
            else ("text_latn" if "text_latn" in col_names else col_names[0])
        )
        data = table[target_col].to_pylist()
        for item in data:
            if item and str(item).strip():
                lines.append(str(item).strip())
                if max_lines and len(lines) >= max_lines:
                    break
        return lines

    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            s = raw_line.strip()
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
                        lines.append(text)
                except json.JSONDecodeError:
                    continue
            else:
                lines.append(s)
            if max_lines and len(lines) >= max_lines:
                break
    return lines
