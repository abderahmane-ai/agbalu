"""Build and stage agbalu/KabLiterary.

Processes the 11 translated and preserved literary masterworks into a structured,
strongly-typed dataset in both Apache Parquet and JSON Lines formats.

Splits are partitioned at book boundaries to prevent intra-document leakage:
  - train: 9 books (27,619 paragraphs, ~630K words)
  - dev: The Prophet / Nnbi (352 paragraphs, ~4.2K words)
  - test: Alice in Wonderland / Tirmit n Alice (936 paragraphs, ~23K words)

Usage:
    python3 -m tools.build_kabliterary --stage artifacts/release/KabLiterary
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA_DIR: Final = Path("data/kabliterary")
DEFAULT_RELEASE_DIR: Final = Path("artifacts/release/KabLiterary")
CARD_PATH: Final = Path("docs/cards/kabliterary.md")
MAX_TITLE_WORDS: Final = 6


class BookConfig(TypedDict):
    slug: str
    title: str
    kabyle_title: str
    author: str
    year: int
    genre: str
    source_language: str
    target_language: str
    split: str
    trans_path: str
    doc_path: str
    is_line_aligned: bool


class StagedFileStat(TypedDict):
    config: str
    split: str
    rows: int
    path: str
    bytes: int


class StagedStats(TypedDict):
    repo: str
    files: list[StagedFileStat]
    rows: int
    total_words: int
    total_characters: int


BOOKS: Final[tuple[BookConfig, ...]] = (
    BookConfig(
        slug="alice-in-wonderland",
        title="Alice's Adventures in Wonderland",
        kabyle_title="Tirmit n Alice deg Tmurt n Lɛeǧayeb",
        author="Lewis Carroll",
        year=1865,
        genre="children_fiction",
        source_language="eng",
        target_language="kab",
        split="test",
        trans_path="artifacts/translations/eng-kab/alice-in-wonderland.txt",
        doc_path="data/documents/eng/alice-in-wonderland.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="the-prophet",
        title="The Prophet",
        kabyle_title="Nnbi",
        author="Kahlil Gibran",
        year=1923,
        genre="poetic_philosophy",
        source_language="eng",
        target_language="kab",
        split="dev",
        trans_path="artifacts/translations/eng-kab/the-prophet.txt",
        doc_path="data/documents/eng/the-prophet.txt",
        is_line_aligned=True,
    ),
    BookConfig(
        slug="the-art-of-war",
        title="The Art of War",
        kabyle_title="Taẓuri n Ṭṭraḍ",
        author="Sun Tzu",
        year=1910,
        genre="military_philosophy",
        source_language="eng",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/eng-kab/the-art-of-war.txt",
        doc_path="data/documents/eng/the-art-of-war.txt",
        is_line_aligned=True,
    ),
    BookConfig(
        slug="fables-of-bidpai",
        title="The Fables of Bidpai",
        kabyle_title="Timucuha n Bidpai",
        author="Bidpai / Ibn al-Muqaffa",
        year=1884,
        genre="fables_folklore",
        source_language="eng",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/eng-kab/fables-of-bidpai.txt",
        doc_path="data/documents/eng/fables-of-bidpai.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="dracula",
        title="Dracula",
        kabyle_title="Dracula, Amudir n Yiḍ",
        author="Bram Stoker",
        year=1897,
        genre="gothic_horror",
        source_language="eng",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/eng-kab/dracula.txt",
        doc_path="data/documents/eng/dracula.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="confessions-of-st-augustine",
        title="The Confessions of Saint Augustine",
        kabyle_title="Isteɛṛufen n Uɣustin Aqeddis",
        author="Saint Augustine of Hippo",
        year=397,
        genre="theological_philosophy",
        source_language="eng",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/eng-kab/confessions-of-st-augustine.txt",
        doc_path="data/documents/eng/confessions-of-st-augustine.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="the-arabian-nights",
        title="The Arabian Nights",
        kabyle_title="Alf Yiḍ d Yiḍ",
        author="Anonymous / Andrew Lang",
        year=1898,
        genre="folklore_tales",
        source_language="eng",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/eng-kab/the-arabian-nights.txt",
        doc_path="data/documents/eng/the-arabian-nights.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="odyssey",
        title="The Odyssey",
        kabyle_title="Asikel n Ulises",
        author="Homer",
        year=-800,
        genre="epic_poetry",
        source_language="eng",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/eng-kab/odyssey.txt",
        doc_path="data/documents/eng/odyssey.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="le-comte-de-monte-cristo",
        title="Le Comte de Monte-Cristo",
        kabyle_title="Lkont n Monte-Cristo",
        author="Alexandre Dumas",
        year=1844,
        genre="adventure_fiction",
        source_language="fra",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/fra-kab/le-comte-de-monte-cristo.txt",
        doc_path="data/documents/fra/le-comte-de-monte-cristo.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="les-miserables",
        title="Les Misérables",
        kabyle_title="Imeɣban",
        author="Victor Hugo",
        year=1862,
        genre="historical_fiction",
        source_language="fra",
        target_language="kab",
        split="train",
        trans_path="artifacts/translations/fra-kab/les-miserables.txt",
        doc_path="data/documents/fra/les-miserables.txt",
        is_line_aligned=False,
    ),
    BookConfig(
        slug="isefra-n-si-mohand",
        title="The Poems of Si Mohand u Mhand",
        kabyle_title="Isefra n Si Mohand u Mhand",
        author="Si Mohand u Mhand",
        year=1904,
        genre="classical_poetry",
        source_language="kab",
        target_language="eng",
        split="train",
        trans_path="data/documents/kab/isefra-n-si-mohand.txt",
        doc_path="artifacts/translations/kab-eng/isefra-n-si-mohand.txt",
        is_line_aligned=True,
    ),
)


@dataclass(frozen=True, slots=True)
class LiteraryRow:
    id: str
    book_id: str
    book_title: str
    book_title_kab: str
    author: str
    year: int
    genre: str
    source_language: str
    target_language: str
    chapter: str
    paragraph_id: int
    text: str
    source_text: str
    word_count: int
    char_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "book_title_kab": self.book_title_kab,
            "author": self.author,
            "year": self.year,
            "genre": self.genre,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "chapter": self.chapter,
            "paragraph_id": self.paragraph_id,
            "text": self.text,
            "source_text": self.source_text,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }


def is_chapter_header(slug: str, line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if re.match(r"^(TAGURT|AḤRIC WIS|AḤRIC|Adlis)\b", line):
        return True
    words = line.split()
    if len(words) > MAX_TITLE_WORDS:
        return False
    if slug == "le-comte-de-monte-cristo":
        return ".-- " in line or (line.endswith(".") and not line.endswith("..."))
    if slug in ("fables-of-bidpai", "isefra-n-si-mohand"):
        return not line.endswith((".", "!", "?", ":", ";", ","))
    return False


def parse_book_rows(book: BookConfig, base_path: Path) -> list[LiteraryRow]:
    trans_file = base_path / book["trans_path"]
    if not trans_file.is_file():
        msg = f"Translation file not found: {trans_file}"
        raise FileNotFoundError(msg)

    raw_lines = trans_file.read_text(encoding="utf-8").splitlines()
    text_lines = [line.strip() for line in raw_lines if line.strip()]

    source_lines: list[str] = []
    if book["is_line_aligned"]:
        doc_file = base_path / book["doc_path"]
        if not doc_file.is_file():
            msg = f"Source doc file not found: {doc_file}"
            raise FileNotFoundError(msg)
        raw_doc_lines = doc_file.read_text(encoding="utf-8").splitlines()
        source_lines = [line.strip() for line in raw_doc_lines if line.strip()]
        if len(source_lines) != len(text_lines):
            msg = (
                f"{book['slug']}: mismatch in line counts "
                f"({len(source_lines)} vs {len(text_lines)})"
            )
            raise ValueError(msg)

    rows: list[LiteraryRow] = []
    current_chapter = "Tazwara"
    for idx, line in enumerate(text_lines, start=1):
        if is_chapter_header(book["slug"], line):
            current_chapter = line

        src_text = source_lines[idx - 1] if source_lines else ""
        row_id = f"{book['slug'].replace('-', '_')}_{idx:05d}"
        words = len(line.split())
        chars = len(line)

        rows.append(
            LiteraryRow(
                id=row_id,
                book_id=book["slug"],
                book_title=book["title"],
                book_title_kab=book["kabyle_title"],
                author=book["author"],
                year=book["year"],
                genre=book["genre"],
                source_language=book["source_language"],
                target_language=book["target_language"],
                chapter=current_chapter,
                paragraph_id=idx,
                text=line,
                source_text=src_text,
                word_count=words,
                char_count=chars,
            )
        )
    return rows


def build_splits(
    base_path: Path,
) -> dict[str, list[LiteraryRow]]:
    splits: dict[str, list[LiteraryRow]] = {"train": [], "dev": [], "test": []}
    for book in BOOKS:
        rows = parse_book_rows(book, base_path)
        splits[book["split"]].extend(rows)
    return splits


def write_split_files(
    split_name: str, rows: Sequence[LiteraryRow], out_dir: Path
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dict_rows = [r.as_dict() for r in rows]

    jsonl_path = out_dir / f"{split_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as sink:
        for r in dict_rows:
            sink.write(json.dumps(r, ensure_ascii=False) + "\n")

    parquet_path = out_dir / f"{split_name}.parquet"
    table = pa.Table.from_pylist(dict_rows)
    pq.write_table(table, parquet_path, compression="snappy")

    return jsonl_path.stat().st_size, parquet_path.stat().st_size


def copy_raw_books(target_dir: Path, base_path: Path) -> None:
    books_dir = target_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)
    for book in BOOKS:
        source_path = base_path / book["trans_path"]
        dest_path = books_dir / f"{book['slug']}.txt"
        shutil.copyfile(source_path, dest_path)


def stage_release(
    data_dir: Path,
    release_dir: Path,
    splits: dict[str, list[LiteraryRow]],
    card_path: Path,
    base_path: Path,
) -> StagedStats:
    release_dir.mkdir(parents=True, exist_ok=True)

    file_stats: list[StagedFileStat] = []
    for split_name in ("train", "dev", "test"):
        parquet_src = data_dir / f"{split_name}.parquet"
        parquet_dst = release_dir / f"{split_name}.parquet"
        shutil.copyfile(parquet_src, parquet_dst)

        jsonl_src = data_dir / f"{split_name}.jsonl"
        jsonl_dst = release_dir / f"{split_name}.jsonl"
        shutil.copyfile(jsonl_src, jsonl_dst)

        file_stats.append(
            {
                "config": "default",
                "split": split_name,
                "rows": len(splits[split_name]),
                "path": f"{split_name}.parquet",
                "bytes": parquet_dst.stat().st_size,
            }
        )

    copy_raw_books(release_dir, base_path)

    readme_dst = release_dir / "README.md"
    shutil.copyfile(card_path, readme_dst)

    total_rows = sum(len(r) for r in splits.values())
    stats_data: StagedStats = {
        "repo": "KabLiterary",
        "files": file_stats,
        "rows": total_rows,
        "total_words": sum(r.word_count for sp in splits.values() for r in sp),
        "total_characters": sum(r.char_count for sp in splits.values() for r in sp),
    }

    stats_path = release_dir / "dataset.stats.json"
    stats_path.write_text(json.dumps(stats_data, indent=2) + "\n", encoding="utf-8")
    return stats_data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stage", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--base-path", type=Path, default=Path())
    args = parser.parse_args(argv)

    print("Building KabLiterary splits from 11 verified works...")
    splits = build_splits(args.base_path)

    total_words = 0
    total_rows = 0
    for split_name, rows in splits.items():
        w_count = sum(r.word_count for r in rows)
        c_count = sum(r.char_count for r in rows)
        total_words += w_count
        total_rows += len(rows)
        print(
            f"  {split_name:<5}: {len(rows):>6,} rows | {w_count:>8,} words | {c_count:>10,} chars"
        )
        write_split_files(split_name, rows, args.data_dir)

    copy_raw_books(args.data_dir, args.base_path)
    print(f"Total: {total_rows:,} rows across {total_words:,} words.")
    print(f"Saved split artifacts to {args.data_dir}")

    if args.stage:
        stats = stage_release(args.data_dir, args.stage, splits, CARD_PATH, args.base_path)
        print(f"Staged dataset release at {args.stage} with {len(stats['files'])} split files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
