"""Optical degradation for Feraoun-Pro's rendered lines.

Augraphy is not used: `augment_single_image` writes every input as a PNG into
`./augraphy_cache/` and `BleedThrough` reads a random one back, which is a read/write race
across dataloader workers, and a truncated PNG decodes to a partial array its guard passes.

Every draw comes from the caller's generator, so an un-augmented validation render is
byte-identical between two evaluations.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter

BLUR_PROBABILITY = 0.4
BRIGHTNESS_PROBABILITY = 0.5
CONTRAST_PROBABILITY = 0.4
JPEG_PROBABILITY = 0.4
ROTATION_PROBABILITY = 0.5


class DegradationPipeline:
    """Blur, exposure, JPEG artefacting and a slight skew, in that order.

    The order is the physical one — ink spread happens on the page, exposure at the scanner,
    compression after it — and each stage draws independently, so a line can come through
    clean.
    """

    def apply(self, image: Image.Image, rng: random.Random | None = None) -> Image.Image:
        """Degrade one rendered line."""
        r = rng or random.Random()
        img = image.copy().convert("RGB")

        # Ink spread, as a bleeding nib or an out-of-focus scan.
        if r.random() < BLUR_PROBABILITY:
            img = img.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.3, 0.8)))

        # Scanner exposure.
        if r.random() < BRIGHTNESS_PROBABILITY:
            img = ImageEnhance.Brightness(img).enhance(r.uniform(0.85, 1.15))
        if r.random() < CONTRAST_PROBABILITY:
            img = ImageEnhance.Contrast(img).enhance(r.uniform(0.8, 1.3))

        # Most scanned Kabyle books reach us as JPEG, so the ringing around a glyph's edge is
        # a real feature of the input rather than an abstract corruption.
        if r.random() < JPEG_PROBABILITY:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=r.randint(30, 75))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # A page is never quite square to the platen. Bounded well below the point where a
        # 64 px line's ascenders would leave the canvas.
        if r.random() < ROTATION_PROBABILITY:
            img = img.rotate(
                r.uniform(-1.5, 1.5),
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(255, 255, 255),
            )

        return img
