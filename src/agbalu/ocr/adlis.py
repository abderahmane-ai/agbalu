"""Whole scanned books through Feraoun, page by page.

Reads a book directory of `crops/page_*.png`, cuts each page into lines and transcribes
them. **The model was trained on rendered text and has never been scored on a scan**, so
what comes out of here is unmeasured — and `adlis-pdfs` copyright is unsettled
(`docs/corpus_landscape.md`), which is why nothing from it enters the corpus.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from agbalu.ocr.infer import Recognizer

log: Final = logging.getLogger("agbalu.ocr.adlis")
BLANK_PAGE_PIXEL_FLOOR: Final[int] = 240


@dataclass(frozen=True, slots=True)
class ScannedPage:
    """A single scanned book page image and optional baseline text."""

    page_number: int
    image_path: Path
    baseline_text: str | None = None


@dataclass(frozen=True, slots=True)
class BookTranscription:
    """Full-book OCR transcription result."""

    doc_id: str
    num_pages: int
    pages: list[tuple[int, str]]
    full_text: str


def discover_adlis_books(root_dir: Path) -> list[Path]:
    """Find all processed Adlis book directories containing crops."""
    book_dirs = [
        p.parent for p in root_dir.glob("**/crops") if p.is_dir() and any(p.glob("page_*.png"))
    ]
    return sorted(book_dirs)


def load_book_pages(book_dir: Path) -> list[ScannedPage]:
    """Load all scanned page images and optional ground-truth text from a book directory."""
    crops_dir = book_dir / "crops"
    if not crops_dir.is_dir():
        msg = f"No crops/ directory found in {book_dir}"
        raise FileNotFoundError(msg)

    baselines: dict[int, str] = {}
    pages_jsonl = book_dir / "pages.jsonl"
    if pages_jsonl.is_file():
        with pages_jsonl.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    page_num = obj.get("page_number")
                    text = obj.get("text", "")
                    if page_num is not None:
                        baselines[int(page_num)] = text
                except json.JSONDecodeError:
                    continue

    pages: list[ScannedPage] = []
    for img_path in sorted(crops_dir.glob("page_*.png")):
        stem = img_path.stem
        try:
            num_part = stem.split("_")[-1]
            page_num = int(num_part)
        except ValueError:
            page_num = len(pages) + 1

        pages.append(
            ScannedPage(
                page_number=page_num,
                image_path=img_path,
                baseline_text=baselines.get(page_num),
            )
        )

    return sorted(pages, key=lambda p: p.page_number)


def transcribe_book(
    book_dir: Path,
    ocr: Recognizer,
    output_path: Path | None = None,
) -> BookTranscription:
    """Transcribe an entire scanned book using Feraoun-36M."""
    doc_id = book_dir.name
    pages = load_book_pages(book_dir)
    log.info("Transcribing book %s (%d scanned pages)...", doc_id, len(pages))

    total_pages = len(pages)
    book_start = time.time()
    transcriptions: list[tuple[int, str]] = []
    full_text_blocks: list[str] = []

    for idx, page in enumerate(pages, 1):
        img = Image.open(page.image_path)
        gray = img.convert("L")
        extrema = gray.getextrema()
        if extrema is not None:
            min_val = extrema[0] if isinstance(extrema, tuple) else extrema
            if isinstance(min_val, (int, float)) and min_val > BLANK_PAGE_PIXEL_FLOOR:
                log.info(
                    "  [%2d/%2d] Page %d: [Blank page skipped]", idx, total_pages, page.page_number
                )
                continue

        p_start = time.time()
        text = ocr.recognize_page(img)
        dur = time.time() - p_start

        if text.strip():
            transcriptions.append((page.page_number, text))
            full_text_blocks.append(f"--- [Page {page.page_number}] ---\n{text}")
            lines_count = len([ln for ln in text.splitlines() if ln.strip()])
            first_line = text.splitlines()[0][:50] if text.splitlines() else ""
            log.info(
                "  [%2d/%2d] Page %2d: %2d lines transcribed (%.2fs, %d chars) | %s...",
                idx,
                total_pages,
                page.page_number,
                lines_count,
                dur,
                len(text),
                first_line,
            )
        else:
            log.info(
                "  [%2d/%2d] Page %2d: [No text detected] (%.2fs)",
                idx,
                total_pages,
                page.page_number,
                dur,
            )

    elapsed = time.time() - book_start
    full_text = "\n\n".join(full_text_blocks)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(full_text)
        log.info(
            "Saved transcription to %s (%d chars, %d pages, total %.1fs)",
            output_path,
            len(full_text),
            len(transcriptions),
            elapsed,
        )

    return BookTranscription(
        doc_id=doc_id,
        num_pages=len(pages),
        pages=transcriptions,
        full_text=full_text,
    )
