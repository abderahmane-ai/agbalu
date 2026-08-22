"""Font pool management and typeface selection for Feraoun-Pro synthetic rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import ImageFont

log = logging.getLogger(__name__)

TIFINAGH_FONT_PATH: Final = Path("resources/fonts/NotoSansTifinagh-Regular.ttf")
FONTS_DIR: Final = Path("resources/fonts/latin_pool")

# Standard fallback fonts commonly available across Linux/macOS
SYSTEM_FALLBACKS: Final[tuple[str, ...]] = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/LucidaGrande.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


@dataclass(frozen=True, slots=True)
class FontEntry:
    family: str
    path: Path
    category: str


KABYLE_PROBE_CHARS: Final[tuple[str, ...]] = ("ɣ", "ɛ", "ḍ", "ḥ", "ṭ", "ẓ", "ţ")


class FontPoolError(RuntimeError):
    """No usable face, so no line can be rendered."""


NOTDEF: Final = "￿"
"""A noncharacter, so no face has a glyph for it and every face draws .notdef."""

PROBE_SIZE: Final = 28


def _supports_kabyle_glyphs(font_path: Path) -> bool:
    """Whether the face has a real glyph for every Kabyle probe character.

    A face without the glyph still renders, and `getbbox` measures the .notdef box it draws
    instead — Georgia passes that test with neither `ɣ` nor `ɛ`, and a renderer built on it
    supervises the model to read a box as a letter. The rendered mask is compared against
    .notdef's instead, which agrees with the fonts' own cmap on every face here.
    """
    try:
        font = ImageFont.truetype(str(font_path), size=PROBE_SIZE)
    except OSError:
        return False
    notdef = bytes(font.getmask(NOTDEF, mode="L"))
    return all(bytes(font.getmask(char, mode="L")) != notdef for char in KABYLE_PROBE_CHARS)


class FontPool:
    """Manages the pool of verified Latin and Tifinagh fonts for synthetic rendering."""

    def __init__(self, fonts: list[FontEntry], tifinagh_path: Path | None = None) -> None:
        self.fonts = fonts
        self.tifinagh_path = tifinagh_path or TIFINAGH_FONT_PATH
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    @classmethod
    def discover(cls, pool_dir: Path = FONTS_DIR) -> FontPool:
        """Scan font directory or fall back to available system fonts with Berber glyph support."""
        entries: list[FontEntry] = []
        if pool_dir.exists() and pool_dir.is_dir():
            for ttf_path in pool_dir.rglob("*.ttf"):
                if _supports_kabyle_glyphs(ttf_path):
                    category = "SERIF" if "serif" in str(ttf_path).lower() else "SANS_SERIF"
                    entries.append(
                        FontEntry(family=ttf_path.stem, path=ttf_path, category=category)
                    )

        if not entries:
            for fallback in SYSTEM_FALLBACKS:
                fb_path = Path(fallback)
                if fb_path.exists() and _supports_kabyle_glyphs(fb_path):
                    entries.append(
                        FontEntry(family=fb_path.stem, path=fb_path, category="SYSTEM_FALLBACK")
                    )

        if not entries:
            message = (
                f"no face in {pool_dir} or on this machine has all of "
                f"{''.join(KABYLE_PROBE_CHARS)}; rendering would supervise .notdef boxes"
            )
            raise FontPoolError(message)

        return cls(fonts=entries)

    def get_latin_font(self, index: int, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Retrieve a Latin font by index and point size."""
        if not self.fonts:
            return ImageFont.load_default()
        entry = self.fonts[index % len(self.fonts)]
        key = (str(entry.path), size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(str(entry.path), size=size)
            except Exception:
                return ImageFont.load_default()
        return self._font_cache[key]

    def get_tifinagh_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Retrieve the canonical Noto Sans Tifinagh font."""
        if not self.tifinagh_path.exists():
            return ImageFont.load_default()
        key = (str(self.tifinagh_path), size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(str(self.tifinagh_path), size=size)
            except Exception:
                return ImageFont.load_default()
        return self._font_cache[key]

    def count(self) -> int:
        return len(self.fonts)
