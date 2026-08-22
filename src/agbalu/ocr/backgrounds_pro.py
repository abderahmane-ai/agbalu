"""Real-paper background texture harvesting and compositing for Feraoun-Pro."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ADLIS_RAW_DIR: Final = Path("data/raw/hf.boffire.adlis-pdfs-ocr-kab")
MIN_TILE_HEIGHT: Final = 32
MIN_TILE_WIDTH: Final = 128


class BackgroundPool:
    """Manages paper texture tiles harvested from blank regions of archival scans."""

    def __init__(self, tiles: list[Image.Image] | None = None) -> None:
        self.tiles: list[Image.Image] = tiles or []

    @classmethod
    def harvest_from_directory(
        cls, scan_dir: Path = ADLIS_RAW_DIR, max_tiles: int = 200
    ) -> BackgroundPool:
        """Extract blank inter-line and margin regions from scan images on disk."""
        tiles: list[Image.Image] = []
        if not scan_dir.exists() or not scan_dir.is_dir():
            return cls(tiles=[])

        image_files = sorted(list(scan_dir.glob("**/*.png")) + list(scan_dir.glob("**/*.jpg")))[:50]
        for img_path in image_files:
            if len(tiles) >= max_tiles:
                break
            try:
                page = Image.open(img_path).convert("L")
                page_arr = np.array(page, dtype=np.uint8)
                ink = page_arr < 200
                row_ink = ink.mean(axis=1)

                # Identify horizontal bands where ink percentage is near zero (< 0.005)
                blank_rows = np.where(row_ink < 0.005)[0]
                if len(blank_rows) == 0:
                    continue

                # Find contiguous blank segments
                diffs = np.diff(blank_rows)
                split_indices = np.where(diffs > 1)[0] + 1
                segments = np.split(blank_rows, split_indices)

                for seg in segments:
                    if len(seg) >= MIN_TILE_HEIGHT:
                        top = int(seg[0])
                        bottom = int(seg[-1])
                        tile = page.crop((0, top, page.width, bottom)).convert("RGB")
                        tiles.append(tile)
                        if len(tiles) >= max_tiles:
                            break
            except Exception as e:
                log.debug("Failed harvesting backgrounds from %s: %s", img_path, e)

        return cls(tiles=tiles)

    def sample_tile(self, width: int, height: int, rng: random.Random | None = None) -> Image.Image:
        """Sample or generate a paper background texture of the requested dimensions."""
        r = rng or random.Random()
        if self.tiles:
            tile = r.choice(self.tiles)
            # Crop or tile to requested width and height
            if tile.height < height:
                tile = tile.resize((tile.width, height), resample=Image.Resampling.BILINEAR)
            if tile.width < width:
                # Repeat horizontally if tile is narrower than canvas
                repeats = (width // tile.width) + 1
                extended = Image.new("RGB", (tile.width * repeats, tile.height))
                for i in range(repeats):
                    extended.paste(tile, (i * tile.width, 0))
                tile = extended

            max_x = max(0, tile.width - width)
            max_y = max(0, tile.height - height)
            x = r.randint(0, max_x) if max_x > 0 else 0
            y = r.randint(0, max_y) if max_y > 0 else 0
            return tile.crop((x, y, x + width, y + height))

        # Synthetic paper texture generator
        return self._generate_synthetic_paper(width, height, r)

    @staticmethod
    def _generate_synthetic_paper(width: int, height: int, r: random.Random) -> Image.Image:
        base_gray = r.randint(235, 252)
        tint_r = base_gray + r.randint(-3, 3)
        tint_g = base_gray + r.randint(-2, 2)
        tint_b = base_gray - r.randint(2, 8)  # slight warm/aged tint

        arr = np.full((height, width, 3), [tint_r, tint_g, tint_b], dtype=np.float32)
        # Seeded from the caller's generator, not `np.random`: the grain is most of the
        # pixels in the image, and drawing it from the global generator made an
        # un-augmented render different on every call. A held-out line was then a
        # different image at every evaluation, which makes `best.pt` a coin toss between
        # two checkpoints rather than a comparison of them.
        noise = np.random.default_rng(r.getrandbits(63)).normal(0, 3.0, (height, width, 3))
        grained = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(grained, mode="RGB")
