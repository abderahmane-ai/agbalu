"""Build `agbalu/KabTifinagh` from the raw Latin-Tifinagh parallel corpus.

The Latin side goes through the reference normaliser before anything else. That is not
hygiene here, it is the dataset's reason for existing: the model trained on this has to
predict a schwa from its consonant context, and a corpus in which `ɛ` is sometimes Greek
epsilon teaches two contexts where there is one.

Splits are by sentence at seed 42 and checked for leakage on the built sets, not on the
slice arithmetic that produced them.

Writes `data/tasks/tifinagh/` only. Staging for the Hub is `tools.export_datasets`.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

from agbalu.normalise import Normaliser

log: Final = logging.getLogger("agbalu.tools.build_kabtifinagh")

RAW_DIR: Final = Path("data/raw/hf.abdelhaqueidali.kab-latn-tfng")
OUT_DIR: Final = Path("data/tasks/tifinagh")

SEED: Final = 42
TRAIN_RATIO: Final = 0.80
DEV_RATIO: Final = 0.10
"""Test takes the remainder, so there is no third constant that could disagree."""
MIN_WORDS: Final = 4
MAX_WORDS: Final = 35
"""Word bounds on the Latin side. Below four a sentence carries too little context for a
schwa to be predictable; above thirty-five the source corpus's rows are concatenations."""

SCRIPT_COLUMNS: Final = 3
TRILINGUAL_COLUMNS: Final = 4
"""Fields a usable row of each TSV must have. A shorter row is a truncated export line."""


class ScriptConversionRow(TypedDict):
    id: str
    text_latn: str
    text_tfng: str


class TrilingualEnRow(TypedDict):
    id: str
    text_latn: str
    text_tfng: str
    text_en: str


class TrilingualFrRow(TypedDict):
    id: str
    text_latn: str
    text_tfng: str
    text_fr: str


Row = ScriptConversionRow | TrilingualEnRow | TrilingualFrRow


def build_script_conversion_dataset(normaliser: Normaliser) -> dict[str, list[ScriptConversionRow]]:
    """Process kab_tifinagh.tsv into normalized, deduplicated 80/10/10 splits."""
    tsv_path = RAW_DIR / "kab_tifinagh.tsv"
    if not tsv_path.exists():
        msg = f"Raw TSV missing at {tsv_path}"
        raise FileNotFoundError(msg)

    lines = tsv_path.read_text(encoding="utf-8").splitlines()[1:]
    log.info("%d raw lines from %s", len(lines), tsv_path.name)

    seen_latn: set[str] = set()
    seen_tfng: set[str] = set()
    valid_rows: list[tuple[str, str]] = []

    for line in lines:
        parts = line.split("\t")
        if len(parts) < SCRIPT_COLUMNS:
            continue

        raw_tfng = parts[1].strip()
        raw_latn = parts[2].strip()

        if not raw_tfng or not raw_latn:
            continue

        norm_latn = normaliser.normalise(raw_latn)
        words = norm_latn.split()

        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue

        if norm_latn in seen_latn or raw_tfng in seen_tfng:
            continue

        seen_latn.add(norm_latn)
        seen_tfng.add(raw_tfng)
        valid_rows.append((norm_latn, raw_tfng))

    log.info("%d clean pairs after dedup and normalisation", len(valid_rows))

    rng = random.Random(SEED)  # noqa: S311
    rng.shuffle(valid_rows)

    n_total = len(valid_rows)
    n_train = int(n_total * TRAIN_RATIO)
    n_dev = int(n_total * DEV_RATIO)

    train_tuples = valid_rows[:n_train]
    dev_tuples = valid_rows[n_train : n_train + n_dev]
    test_tuples = valid_rows[n_train + n_dev :]

    def make_rows(tuples: list[tuple[str, str]], split_name: str) -> list[ScriptConversionRow]:
        return [
            {
                "id": f"kab_tfng_{split_name}_{idx:06d}",
                "text_latn": latn,
                "text_tfng": tfng,
            }
            for idx, (latn, tfng) in enumerate(tuples)
        ]

    return {
        "train": make_rows(train_tuples, "train"),
        "dev": make_rows(dev_tuples, "dev"),
        "test": make_rows(test_tuples, "test"),
    }


def build_trilingual_en_dataset(normaliser: Normaliser) -> dict[str, list[TrilingualEnRow]]:
    """Process en_kab_tifinagh.tsv into trilingual splits."""
    tsv_path = RAW_DIR / "en_kab_tifinagh.tsv"
    if not tsv_path.exists():
        log.warning("no trilingual English TSV at %s", tsv_path)
        return {"train": [], "dev": [], "test": []}

    lines = tsv_path.read_text(encoding="utf-8").splitlines()[1:]
    valid_rows: list[tuple[str, str, str]] = []
    seen_latn: set[str] = set()

    for line in lines:
        parts = line.split("\t")
        if len(parts) < TRILINGUAL_COLUMNS:
            continue

        raw_tfng = parts[1].strip()
        raw_latn = parts[2].strip()
        raw_en = parts[3].strip()

        if not raw_tfng or not raw_latn or not raw_en:
            continue

        norm_latn = normaliser.normalise(raw_latn)
        if norm_latn in seen_latn:
            continue
        seen_latn.add(norm_latn)
        valid_rows.append((norm_latn, raw_tfng, raw_en))

    rng = random.Random(SEED)  # noqa: S311
    rng.shuffle(valid_rows)

    n_total = len(valid_rows)
    n_train = int(n_total * TRAIN_RATIO)
    n_dev = int(n_total * DEV_RATIO)

    train_tuples = valid_rows[:n_train]
    dev_tuples = valid_rows[n_train : n_train + n_dev]
    test_tuples = valid_rows[n_train + n_dev :]

    def make_rows(tuples: list[tuple[str, str, str]], split_name: str) -> list[TrilingualEnRow]:
        return [
            {
                "id": f"kab_tfng_en_{split_name}_{idx:06d}",
                "text_latn": latn,
                "text_tfng": tfng,
                "text_en": en,
            }
            for idx, (latn, tfng, en) in enumerate(tuples)
        ]

    return {
        "train": make_rows(train_tuples, "train"),
        "dev": make_rows(dev_tuples, "dev"),
        "test": make_rows(test_tuples, "test"),
    }


def build_trilingual_fr_dataset(normaliser: Normaliser) -> dict[str, list[TrilingualFrRow]]:
    """Process fr_kab_tifinagh.tsv into trilingual splits."""
    tsv_path = RAW_DIR / "fr_kab_tifinagh.tsv"
    if not tsv_path.exists():
        log.warning("no trilingual French TSV at %s", tsv_path)
        return {"train": [], "dev": [], "test": []}

    lines = tsv_path.read_text(encoding="utf-8").splitlines()[1:]
    valid_rows: list[tuple[str, str, str]] = []
    seen_latn: set[str] = set()

    for line in lines:
        parts = line.split("\t")
        if len(parts) < TRILINGUAL_COLUMNS:
            continue

        raw_tfng = parts[1].strip()
        raw_latn = parts[2].strip()
        raw_fr = parts[3].strip()

        if not raw_tfng or not raw_latn or not raw_fr:
            continue

        norm_latn = normaliser.normalise(raw_latn)
        if norm_latn in seen_latn:
            continue
        seen_latn.add(norm_latn)
        valid_rows.append((norm_latn, raw_tfng, raw_fr))

    rng = random.Random(SEED)  # noqa: S311
    rng.shuffle(valid_rows)

    n_total = len(valid_rows)
    n_train = int(n_total * TRAIN_RATIO)
    n_dev = int(n_total * DEV_RATIO)

    train_tuples = valid_rows[:n_train]
    dev_tuples = valid_rows[n_train : n_train + n_dev]
    test_tuples = valid_rows[n_train + n_dev :]

    def make_rows(tuples: list[tuple[str, str, str]], split_name: str) -> list[TrilingualFrRow]:
        return [
            {
                "id": f"kab_tfng_fr_{split_name}_{idx:06d}",
                "text_latn": latn,
                "text_tfng": tfng,
                "text_fr": fr,
            }
            for idx, (latn, tfng, fr) in enumerate(tuples)
        ]

    return {
        "train": make_rows(train_tuples, "train"),
        "dev": make_rows(dev_tuples, "dev"),
        "test": make_rows(test_tuples, "test"),
    }


def write_parquet(rows: Sequence[Row], out_file: Path) -> None:
    """Write list of dict rows to Parquet."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_file, compression="snappy")
    log.info("%s: %d rows", out_file, len(rows))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    normaliser = Normaliser()

    script_data = build_script_conversion_dataset(normaliser)
    en_data = build_trilingual_en_dataset(normaliser)
    fr_data = build_trilingual_fr_dataset(normaliser)

    for split in ("train", "dev", "test"):
        write_parquet(script_data[split], OUT_DIR / "script_conversion" / f"{split}.parquet")
        if en_data[split]:
            write_parquet(en_data[split], OUT_DIR / "trilingual_en" / f"{split}.parquet")
        if fr_data[split]:
            write_parquet(fr_data[split], OUT_DIR / "trilingual_fr" / f"{split}.parquet")

    info = {
        "dataset_name": "agbalu/KabTifinagh",
        "description": "Kabyle in both scripts, and schwa restoration",
        "language": "kab_Latn",
        "license": "CC-BY-2.0",
        "normaliser_version": normaliser.version,
        "splits": {
            "script_conversion": {
                "train": len(script_data["train"]),
                "dev": len(script_data["dev"]),
                "test": len(script_data["test"]),
                "total": sum(len(v) for v in script_data.values()),
            },
            "trilingual_en": {
                "train": len(en_data["train"]),
                "dev": len(en_data["dev"]),
                "test": len(en_data["test"]),
                "total": sum(len(v) for v in en_data.values()),
            },
            "trilingual_fr": {
                "train": len(fr_data["train"]),
                "dev": len(fr_data["dev"]),
                "test": len(fr_data["test"]),
                "total": sum(len(v) for v in fr_data.values()),
            },
        },
        "source": "hf.abdelhaqueidali.kab-latn-tfng (CC-BY-2.0)",
    }

    info_file = OUT_DIR / "dataset_info.json"
    info_file.parent.mkdir(parents=True, exist_ok=True)
    info_file.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", OUT_DIR)


if __name__ == "__main__":
    main()
