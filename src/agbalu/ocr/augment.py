"""Physical scan and document degradation transforms for Kabyle OCR training.

Simulates real-world artifacts found in historical book scans, photocopies, and printed
manuscripts:
1. Scanner lens defocus and optical blur.
2. Sensor speckle noise and paper grain.
3. Ink bleed (dilation) and faded/broken ribbon strokes (erosion).
4. Mechanical paper feed skew and slight affine rotation.
5. Aging paper texture and illumination gradients.
"""

from __future__ import annotations

import random
from typing import Final

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Default ranges tuned for historical document scans (150–300 DPI equivalent)
MAX_SKEW_ANGLE: Final[float] = 2.0
BLUR_RADIUS_MAX: Final[float] = 0.8
NOISE_STD_MAX: Final[float] = 15.0

# How often each degradation fires. Together they are the recipe: read down this list to
# see what fraction of rendered lines carries which artifact, without opening the
# functions. `P_INK_BLEED` and `P_INK_FADE` are cumulative over one draw, so a line takes
# at most one of the two.
P_BLUR: Final[float] = 0.5
P_NOISE: Final[float] = 0.5
P_INK_BLEED: Final[float] = 0.25
P_INK_FADE: Final[float] = 0.50
P_PHOTOMETRIC: Final[float] = 0.6
P_PAPER_TINT: Final[float] = 0.4
P_SHADOW: Final[float] = 0.35
P_BLEEDTHROUGH: Final[float] = 0.3


def apply_skew(image: Image.Image, max_angle: float = MAX_SKEW_ANGLE) -> Image.Image:
    """Apply a random slight rotation simulating scanner paper feed skew."""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def apply_blur(image: Image.Image, max_radius: float = BLUR_RADIUS_MAX) -> Image.Image:
    """Apply slight Gaussian blur simulating scanner defocus or low-DPI optics."""
    if random.random() < P_BLUR:
        radius = random.uniform(0.1, max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image


def apply_noise(image: Image.Image, max_std: float = NOISE_STD_MAX) -> Image.Image:
    """Add Gaussian paper grain and sensor noise."""
    if random.random() < P_NOISE:
        arr = np.array(image, dtype=np.float32)
        std = random.uniform(2.0, max_std)
        noise = np.random.normal(0, std, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    return image


def apply_ink_degradation(image: Image.Image) -> Image.Image:
    """Simulate ink bleed (dilation/darkening) or dry-ribbon fade (erosion)."""
    p = random.random()
    if p < P_INK_BLEED:
        return image.filter(ImageFilter.MinFilter(size=3))
    if p < P_INK_FADE:
        return image.filter(ImageFilter.MaxFilter(size=3))
    return image


def apply_photometric_jitter(image: Image.Image) -> Image.Image:
    """Adjust brightness and contrast to simulate photocopier exposure variations."""
    if random.random() < P_PHOTOMETRIC:
        contrast_factor = random.uniform(0.75, 1.35)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    if random.random() < P_PHOTOMETRIC:
        brightness_factor = random.uniform(0.85, 1.15)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    return image


def apply_paper_tint(image: Image.Image) -> Image.Image:
    """Simulate aged or yellowed book paper backgrounds."""
    if random.random() < P_PAPER_TINT:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.float32)
        tint_r = random.uniform(0.98, 1.0)
        tint_g = random.uniform(0.95, 0.99)
        tint_b = random.uniform(0.88, 0.96)
        arr[:, :, 0] *= tint_r
        arr[:, :, 1] *= tint_g
        arr[:, :, 2] *= tint_b
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return image


def apply_shadow_gradient(image: Image.Image) -> Image.Image:
    """Simulate uneven scanner lighting or book spine gutter shadow gradient."""
    if random.random() < P_SHADOW:
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.float32)
        h, w, _ = arr.shape
        direction = random.choice(["left_to_right", "right_to_left", "top_to_bottom"])
        if direction == "left_to_right":
            grad = np.linspace(random.uniform(0.65, 0.9), 1.0, w, dtype=np.float32)
            grad = np.tile(grad, (h, 1))
        elif direction == "right_to_left":
            grad = np.linspace(1.0, random.uniform(0.65, 0.9), w, dtype=np.float32)
            grad = np.tile(grad, (h, 1))
        else:
            grad = np.linspace(random.uniform(0.7, 0.95), 1.0, h, dtype=np.float32)
            grad = np.tile(grad[:, None], (1, w))

        arr *= grad[:, :, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return image


def apply_bleedthrough_noise(image: Image.Image) -> Image.Image:
    """Simulate ghost ink bleed-through from the back side of thin printed paper."""
    if random.random() < P_BLEEDTHROUGH:
        arr = np.array(image.convert("L"), dtype=np.float32)
        h, w = arr.shape
        ghost_raw = np.random.uniform(0.88, 1.0, (h, w)).astype(np.float32)
        ghost_img = Image.fromarray((ghost_raw * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=2.0)
        )
        ghost_arr = np.array(ghost_img, dtype=np.float32) / 255.0

        arr = np.clip(arr * ghost_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return image


def augment_line_image(image: Image.Image) -> Image.Image:
    """Apply the full stochastic scan degradation pipeline to a single text line image."""
    img = image.convert("RGB")
    img = apply_skew(img)
    img = apply_shadow_gradient(img)
    img = apply_bleedthrough_noise(img)
    img = apply_ink_degradation(img)
    img = apply_blur(img)
    img = apply_noise(img)
    img = apply_photometric_jitter(img)
    return apply_paper_tint(img)
