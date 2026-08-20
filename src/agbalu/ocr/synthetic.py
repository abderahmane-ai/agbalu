"""Synthetic text line image renderer for Kabyle document OCR.

Renders authentic normalized Kabyle text lines from `AƔBALU-Text v1` into high-resolution,
degraded document line images with exact ground-truth character alignment.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from agbalu.ocr.augment import augment_line_image

# Model input dimensions (fixed height 64, fixed width 512)
TARGET_HEIGHT: Final[int] = 64
TARGET_WIDTH: Final[int] = 512

# Common system font candidates across Linux, macOS, and container environments
FONT_CANDIDATES: Final[tuple[str, ...]] = (
    # Linux (Debian / Ubuntu / Modal containers)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTifinagh-Regular.ttf",
    # macOS Fonts
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Baskerville.ttc",
    "/System/Library/Fonts/Supplemental/Palatino.ttc",
    "/System/Library/Fonts/Supplemental/Didot.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Courier.dfont",
    "/Library/Fonts/Times New Roman.ttf",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Georgia.ttf",
)


TIFINAGH_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    str(
        Path(__file__).resolve().parents[3] / "resources" / "fonts" / "NotoSansTifinagh-Regular.ttf"
    ),
    "resources/fonts/NotoSansTifinagh-Regular.ttf",
    "/root/resources/fonts/NotoSansTifinagh-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTifinagh-Regular.ttf",
    "/System/Library/Fonts/Supplemental/NotoSansTifinagh-Regular.ttf",
    "/Library/Fonts/NotoSansTifinagh-Regular.ttf",
)


def is_tifinagh_text(text: str) -> bool:
    """Return True if text contains characters in the Tifinagh Unicode block (U+2D30..U+2D7F)."""
    return any("\u2d30" <= ch <= "\u2d7f" for ch in text)


def get_available_fonts() -> list[str]:
    """Return the list of existing font paths on the host system."""
    return [p for p in FONT_CANDIDATES if Path(p).is_file()]


def get_available_tifinagh_fonts() -> list[str]:
    """Return the list of existing Tifinagh font paths."""
    return [p for p in TIFINAGH_FONT_CANDIDATES if Path(p).is_file()]


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a FreeType font if path is valid, else fallback to Pillow's default."""
    if font_path and Path(font_path).is_file():
        try:
            return ImageFont.truetype(font_path, size=size)
        except (OSError, ValueError):
            return ImageFont.load_default()
    return ImageFont.load_default()


def render_text_line(
    text: str,
    font_path: str | None = None,
    font_size: int = 32,
    augment: bool = True,
    rng: random.Random | None = None,
) -> Image.Image:
    """Render a single text line to a PIL image with optional scan degradation.

    Preserves sub-dot diacritics by rendering at high resolution, then scales proportionally
    and pads to the target (TARGET_HEIGHT=64, TARGET_WIDTH=512) canvas.

    `rng` fixes the font and the horizontal offset. Passing one is what makes a validation
    render the same image every time it is scored: on the module generator the same held-out
    line is drawn in a different face at a different offset at every evaluation, and the CER
    curve then moves with the renderer rather than with the model.
    """
    if not text.strip():
        text = " "
    draw_from = rng if rng is not None else random

    if is_tifinagh_text(text):
        tif_fonts = get_available_tifinagh_fonts()
        chosen_font = (
            font_path
            if (font_path and Path(font_path).is_file())
            else (draw_from.choice(tif_fonts) if tif_fonts else None)
        )
    else:
        lat_fonts = get_available_fonts()
        chosen_font = (
            font_path
            if (font_path and Path(font_path).is_file())
            else (draw_from.choice(lat_fonts) if lat_fonts else None)
        )

    font = load_font(chosen_font, font_size)

    # Measure text bounding box
    dummy_img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(int(bbox[2] - bbox[0] + 16), 20)
    text_h = max(int(bbox[3] - bbox[1] + 16), 20)

    # Render on clean white canvas
    line_img = Image.new("RGB", (text_w, text_h), color=(255, 255, 255))
    line_draw = ImageDraw.Draw(line_img)
    # Slight margin offset
    offset_x = float(8 - bbox[0])
    offset_y = float(8 - bbox[1])
    line_draw.text((offset_x, offset_y), text, font=font, fill=(0, 0, 0))

    # Apply physical scan degradation if enabled
    if augment:
        line_img = augment_line_image(line_img)

    # Scale proportionally to fit TARGET_HEIGHT=64 without distorting aspect ratio
    orig_w, orig_h = line_img.size
    target_usable_h = TARGET_HEIGHT - 12  # 52px
    target_usable_w = TARGET_WIDTH - 16  # 496px

    scale = target_usable_h / max(orig_h, 1)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    if new_w > target_usable_w:
        scale = target_usable_w / max(orig_w, 1)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

    new_w = max(1, min(new_w, target_usable_w))
    new_h = max(1, min(new_h, target_usable_h))

    resized_line = line_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Place onto final fixed (64, 512) canvas with background color matching edge pixel
    bg_color = resized_line.getpixel((0, 0))
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=bg_color)

    # Random slight horizontal/vertical offset within padding
    pad_x = draw_from.randint(4, max(TARGET_WIDTH - new_w - 4, 4))
    pad_y = (TARGET_HEIGHT - new_h) // 2

    canvas.paste(resized_line, (pad_x, pad_y))
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image (RGB) to a normalized PyTorch FloatTensor of shape (3, H, W).

    Normalized to mean=0.5, std=0.5 (range [-1.0, 1.0]), standard for vision transformer backbones.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3), range [0, 1]
    arr = (arr - 0.5) / 0.5  # range [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)
