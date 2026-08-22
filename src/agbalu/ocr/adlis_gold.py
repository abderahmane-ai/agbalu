"""Adlis real-scan line sampler and gold evaluation manifest builder for Feraoun-Pro."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ADLIS_RAW_DIR: Final = Path("data/raw/hf.boffire.adlis-pdfs-ocr-kab")
GOLD_OUTPUT_PATH: Final = Path("data/processed/bench/feraoun-pro-real-heldout.json")


def segment_lines_from_page(page: Image.Image, min_height: int = 15) -> list[Image.Image]:
    """Segment a page image into line strips based on ink density."""
    array = np.array(page.convert("L"), dtype=np.uint8)
    ink = (array < 200).astype(np.float32).sum(axis=1)
    active = ink > 0.015 * array.shape[1]

    lines: list[Image.Image] = []
    start = 0
    inside = False

    for idx, is_ink in enumerate(active):
        if is_ink and not inside:
            start = idx
            inside = True
        elif not is_ink and inside:
            inside = False
            if idx - start >= min_height:
                top = max(0, start - 2)
                bottom = min(page.height, idx + 2)
                lines.append(page.crop((0, top, page.width, bottom)).convert("RGB"))

    if inside and len(active) - start >= min_height:
        lines.append(page.crop((0, start, page.width, len(active))).convert("RGB"))

    return lines


def harvest_real_line_crops(
    scan_dir: Path = ADLIS_RAW_DIR,
    output_dir: Path = Path("data/processed/ocr/adlis_real_lines"),
    num_samples: int = 500,
    seed: int = 42,
) -> list[Path]:
    """Extract and persist real line crops from Adlis scans."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = sorted(list(scan_dir.glob("**/*.png")) + list(scan_dir.glob("**/*.jpg")))
    if not all_images:
        log.warning("No scan images found in %s", scan_dir)
        return []

    rng = random.Random(seed)
    rng.shuffle(all_images)

    extracted_paths: list[Path] = []
    line_idx = 0

    for img_file in all_images:
        if line_idx >= num_samples:
            break
        try:
            page = Image.open(img_file)
            lines = segment_lines_from_page(page)
            for line_img in lines:
                if line_img.width >= 100 and line_img.height >= 20:
                    out_path = output_dir / f"line_{line_idx:05d}.png"
                    line_img.save(out_path)
                    extracted_paths.append(out_path)
                    line_idx += 1
                    if line_idx >= num_samples:
                        break
        except Exception as e:
            log.debug("Error processing %s: %s", img_file, e)

    return extracted_paths


def build_gold_manifest(
    line_paths: list[Path], gold_labels: list[str], output_path: Path = GOLD_OUTPUT_PATH
) -> None:
    """Write out the benchmark manifest JSON linking real line crops to gold transcriptions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for path, label in zip(line_paths, gold_labels, strict=False):
        records.append(
            {
                "image_path": str(path),
                "text": label,
                "source": "adlis_real_scan",
            }
        )

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %d gold evaluation records to %s", len(records), output_path)
