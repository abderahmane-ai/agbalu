"""Unit tests for Adlis book reader and scanner pipeline."""

from __future__ import annotations

from pathlib import Path

from agbalu.ocr.adlis import discover_adlis_books, load_book_pages


def test_discover_and_load_adlis_books() -> None:
    raw_dir = Path("data/raw/hf.boffire.adlis-pdfs-ocr-kab")
    if not raw_dir.is_dir():
        return

    books = discover_adlis_books(raw_dir)
    assert len(books) >= 1

    book_dir = books[0]
    pages = load_book_pages(book_dir)
    assert len(pages) == 42
    assert pages[0].page_number == 1
    assert pages[0].image_path.is_file()
