"""Variable-width synthetic line rendering engine for Feraoun-Pro."""

from __future__ import annotations

import random
from typing import Final

from PIL import Image, ImageDraw

from agbalu.ocr.augment_pro import DegradationPipeline
from agbalu.ocr.backgrounds_pro import BackgroundPool
from agbalu.ocr.fonts_pro import FontPool

CANVAS_HEIGHT: Final[int] = 64
MAX_CANVAS_WIDTH: Final[int] = 1024
MIN_CANVAS_WIDTH: Final[int] = 64
PADDING_HORIZONTAL: Final[int] = 12
MIN_CHAR_WIDTH_PX: Final[int] = 8


def is_tifinagh_text(text: str) -> bool:
    """Check if string contains Neo-Tifinagh characters."""
    return any(0x2D30 <= ord(char) <= 0x2D7F for char in text)


class LineRenderer:
    """Renders text lines with variable width, font scaling, and paper compositing."""

    def __init__(
        self,
        font_pool: FontPool | None = None,
        bg_pool: BackgroundPool | None = None,
        degradation: DegradationPipeline | None = None,
    ) -> None:
        self.font_pool = font_pool or FontPool.discover()
        self.bg_pool = bg_pool or BackgroundPool()
        self.degradation = degradation or DegradationPipeline()

    def render_line(
        self,
        text: str,
        rng: random.Random | None = None,
        apply_augmentation: bool = True,
    ) -> tuple[Image.Image, str]:
        """Render a single text line into a (64, W) image."""
        r = rng or random.Random()
        cleaned_text = text.strip()
        if not cleaned_text:
            cleaned_text = " "

        is_tfng = is_tifinagh_text(cleaned_text)
        font_size = r.randint(24, 38)

        if is_tfng:
            font = self.font_pool.get_tifinagh_font(size=font_size)
        else:
            font_idx = r.randint(0, max(0, self.font_pool.count() - 1))
            font = self.font_pool.get_latin_font(index=font_idx, size=font_size)

        # Measure text bounding box
        dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), cleaned_text, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])

        # Enforce minimum character width to avoid CTC gemination collisions
        num_chars = len(cleaned_text)
        if num_chars > 0 and (text_w / num_chars) < MIN_CHAR_WIDTH_PX:
            scale_factor = (num_chars * MIN_CHAR_WIDTH_PX) / text_w
            new_size = int(font_size * scale_factor)
            if is_tfng:
                font = self.font_pool.get_tifinagh_font(size=new_size)
            else:
                font = self.font_pool.get_latin_font(index=font_idx, size=new_size)
            bbox = draw.textbbox((0, 0), cleaned_text, font=font)
            text_w = max(1, bbox[2] - bbox[0])
            text_h = max(1, bbox[3] - bbox[1])

        # `textbbox` measures in float. The canvas is a pixel count and reaches `Image.new`
        # and the collator's `W // 4`, so it is rounded here rather than coerced there.
        canvas_w = int(
            min(MAX_CANVAS_WIDTH, max(MIN_CANVAS_WIDTH, text_w + 2 * PADDING_HORIZONTAL))
        )

        # Sample background: 30% real Adlis paper tile, 70% synthetic textured
        bg = self.bg_pool.sample_tile(canvas_w, CANVAS_HEIGHT, rng=r)

        # Render text on a transparent layer
        text_layer = Image.new("RGBA", (canvas_w, CANVAS_HEIGHT), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        # Ink color: dark charcoal/black with subtle intensity variation
        ink_v = r.randint(10, 45)
        ink_color = (ink_v, ink_v, ink_v, 255)

        # Center vertically
        offset_x = PADDING_HORIZONTAL - bbox[0]
        offset_y = (CANVAS_HEIGHT - text_h) // 2 - bbox[1]

        text_draw.text((offset_x, offset_y), cleaned_text, font=font, fill=ink_color)

        # Composite text layer onto background
        bg_rgba = bg.convert("RGBA")
        composited = Image.alpha_composite(bg_rgba, text_layer).convert("RGB")

        if apply_augmentation:
            composited = self.degradation.apply(composited, rng=r)

        return composited, cleaned_text
