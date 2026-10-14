"""Unit tests for tools.build_kabliterary."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from tools.build_kabliterary import (
    BOOKS,
    CARD_PATH,
    BookConfig,
    build_splits,
    is_chapter_header,
    parse_book_rows,
    stage_release,
    write_split_files,
)


class TestBookConfiguration:
    def test_books_count_and_slugs_unique(self) -> None:
        assert len(BOOKS) == 11
        slugs = [b["slug"] for b in BOOKS]
        assert len(slugs) == len(set(slugs))

    def test_all_books_have_existing_source_files(self) -> None:
        for book in BOOKS:
            assert Path(book["trans_path"]).is_file(), f"Missing trans file for {book['slug']}"
            assert Path(book["doc_path"]).is_file(), f"Missing doc file for {book['slug']}"

    def test_all_books_have_valid_splits(self) -> None:
        valid_splits = {"train", "dev", "test"}
        for book in BOOKS:
            assert book["split"] in valid_splits

    def test_splits_composition(self) -> None:
        train_books = [b["slug"] for b in BOOKS if b["split"] == "train"]
        dev_books = [b["slug"] for b in BOOKS if b["split"] == "dev"]
        test_books = [b["slug"] for b in BOOKS if b["split"] == "test"]

        assert len(train_books) == 9
        assert dev_books == ["the-prophet"]
        assert test_books == ["alice-in-wonderland"]


class TestChapterDetection:
    def test_detects_standard_headings(self) -> None:
        assert is_chapter_header("alice-in-wonderland", "TAGURT I.")
        assert is_chapter_header("the-art-of-war", "AḤRIC I. ASBEDDI N YIƔAWASEN")
        assert is_chapter_header("odyssey", "Adlis I.")
        assert is_chapter_header("fables-of-bidpai", "Azrem d yizerriwen")

    def test_rejects_regular_prose(self) -> None:
        assert not is_chapter_header("alice-in-wonderland", "Tella Alice tebda teɛya aṭas.")
        assert not is_chapter_header("dracula", "3 mayyu. Bistritz.-- Yeǧǧa Munich.")
        assert not is_chapter_header("odyssey", "")


class TestParseBookRows:
    def test_parse_aligned_book(self) -> None:
        prophet_cfg = next(b for b in BOOKS if b["slug"] == "the-prophet")
        rows = parse_book_rows(prophet_cfg, Path())
        assert len(rows) == 352
        first = rows[0]
        assert first.id == "the_prophet_00001"
        assert first.book_id == "the-prophet"
        assert first.author == "Kahlil Gibran"
        assert first.text == "NNBI"
        assert first.source_text == "THE PROPHET"
        assert first.word_count == 1
        assert first.char_count == 4

    def test_parse_prose_book(self) -> None:
        alice_cfg = next(b for b in BOOKS if b["slug"] == "alice-in-wonderland")
        rows = parse_book_rows(alice_cfg, Path())
        assert len(rows) == 936
        assert all(r.book_id == "alice-in-wonderland" for r in rows)
        assert rows[0].source_text == ""


class TestBuildSplitsAndWrite:
    def test_split_counts_match_exact_specification(self) -> None:
        splits = build_splits(Path())
        assert len(splits["train"]) == 27619
        assert len(splits["dev"]) == 352
        assert len(splits["test"]) == 936
        assert sum(len(v) for v in splits.values()) == 28907

    def test_write_and_read_parquet(self, tmp_path: Path) -> None:
        prophet_cfg = next(b for b in BOOKS if b["slug"] == "the-prophet")
        rows = parse_book_rows(prophet_cfg, Path())

        jsonl_size, parquet_size = write_split_files("sample", rows, tmp_path)
        assert jsonl_size > 0
        assert parquet_size > 0

        parquet_file = tmp_path / "sample.parquet"
        assert parquet_file.is_file()
        table = pq.read_table(parquet_file)
        assert table.num_rows == 352
        assert "text" in table.column_names
        assert "book_id" in table.column_names
        assert "word_count" in table.column_names

    def test_stage_release_structure(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        release_dir = tmp_path / "release"

        prophet_cfg: BookConfig = BookConfig(
            slug="the-prophet",
            title="The Prophet",
            kabyle_title="Nnbi",
            author="Kahlil Gibran",
            year=1923,
            genre="poetic_philosophy",
            source_language="eng",
            target_language="kab",
            split="train",
            trans_path="artifacts/translations/eng-kab/the-prophet.txt",
            doc_path="data/documents/eng/the-prophet.txt",
            is_line_aligned=True,
        )
        sample_rows = parse_book_rows(prophet_cfg, Path())
        splits = {"train": sample_rows, "dev": sample_rows[:10], "test": sample_rows[:20]}

        for sp_name, rows in splits.items():
            write_split_files(sp_name, rows, data_dir)

        stats = stage_release(data_dir, release_dir, splits, CARD_PATH, Path())

        assert (release_dir / "README.md").is_file()
        assert (release_dir / "dataset.stats.json").is_file()
        assert (release_dir / "train.parquet").is_file()
        assert (release_dir / "train.jsonl").is_file()
        assert (release_dir / "books").is_dir()
        assert stats["rows"] == 352 + 10 + 20
