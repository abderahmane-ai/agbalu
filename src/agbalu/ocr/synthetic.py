"""Rendering a Kabyle text line onto the canvas the encoder reads.

The label of an image is the string that was drawn, so the two cannot drift apart. What can
drift is the draw: typeface, size and horizontal offset come from an `rng` the caller
passes where a render has to reproduce, and from the module generator where it should
differ every epoch.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from agbalu.ocr.augment import augment_line_image

TARGET_HEIGHT: Final[int] = 64
TARGET_WIDTH: Final[int] = 512

FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTifinagh-Regular.ttf",
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
    """Render one line, degraded, onto the fixed canvas.

    Drawn large and scaled down, so the sub-dots on `ḍ ḥ ṛ ṣ ṭ ẓ` survive as distinct dark
    pixels rather than the one or two they occupy when drawn at the target height.

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

    dummy_img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(int(bbox[2] - bbox[0] + 16), 20)
    text_h = max(int(bbox[3] - bbox[1] + 16), 20)

    line_img = Image.new("RGB", (text_w, text_h), color=(255, 255, 255))
    line_draw = ImageDraw.Draw(line_img)
    offset_x = float(8 - bbox[0])
    offset_y = float(8 - bbox[1])
    line_draw.text((offset_x, offset_y), text, font=font, fill=(0, 0, 0))

    if augment:
        line_img = augment_line_image(line_img)

    orig_w, orig_h = line_img.size
    target_usable_h = TARGET_HEIGHT - 12
    target_usable_w = TARGET_WIDTH - 16

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

    # The corner pixel, so a tinted or shaded render pads into its own background rather
    # than into a white border the model would learn as a line boundary.
    bg_color = resized_line.getpixel((0, 0))
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=bg_color)

    pad_x = draw_from.randint(4, max(TARGET_WIDTH - new_w - 4, 4))
    pad_y = (TARGET_HEIGHT - new_h) // 2

    canvas.paste(resized_line, (pad_x, pad_y))
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """One image to `(3, H, W)` at mean 0.5 and std 0.5.

    `agbalu.hub.feraoun` repeats this arithmetic exactly, because it is half the contract:
    a caller who scales differently gets logits of the right shape from a model that never
    saw that input.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1)
