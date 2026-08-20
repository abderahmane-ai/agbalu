"""Inference and page transcription engine for document OCR.

Provides single-line transcription, batched line decoding, and horizontal projection
page segmentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image

from agbalu.ocr.config import ModelConfig
from agbalu.ocr.models import VisionEncoderDecoder
from agbalu.ocr.synthetic import TARGET_HEIGHT, TARGET_WIDTH, image_to_tensor

LINE_INK_THRESHOLD: Final[int] = 200
"""Grey level below which a pixel counts as ink when splitting a page into lines. Stricter
than the crop threshold on purpose: page segmentation must not open a line on paper
texture, where a crop that clips a sub-dot is the worse error."""

CROP_INK_THRESHOLD: Final[int] = 225
"""Grey level below which a pixel counts as ink when tightening a line's bounding box. A
sub-dot on a photocopy sits well above the segmentation threshold and is what a stricter
value would cut off."""


def crop_text_bbox(img: Image.Image, padding: int = 4) -> Image.Image:
    """Crop image tightly to the bounding box of ink/text pixels."""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    binary = arr < CROP_INK_THRESHOLD
    if not binary.any():
        return img

    y_indices, x_indices = np.where(binary)
    x1 = max(0, int(x_indices.min()) - padding)
    x2 = min(img.width, int(x_indices.max()) + padding)
    y1 = max(0, int(y_indices.min()) - padding)
    y2 = min(img.height, int(y_indices.max()) + padding)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def prepare_line_image(image: Image.Image) -> torch.Tensor:
    """Resize an arbitrary line crop to fixed (64, 512) strictly preserving aspect ratio."""
    cropped = crop_text_bbox(image, padding=4)
    img = cropped.convert("RGB")
    orig_w, orig_h = img.size

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

    resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=(255, 255, 255))
    pad_x = 8
    pad_y = (TARGET_HEIGHT - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return image_to_tensor(canvas)


def segment_page_into_lines(
    page_image: Image.Image,
    min_line_height: int = 15,
) -> list[Image.Image]:
    """Segment a scanned page image into individual text line crops with safe vertical margins."""
    gray = page_image.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    binary = (arr < LINE_INK_THRESHOLD).astype(np.float32)
    proj = binary.sum(axis=1)

    threshold = 0.015 * binary.shape[1]
    is_text = proj > threshold

    lines: list[Image.Image] = []
    in_line = False
    start_y = 0

    for y, active in enumerate(is_text):
        if active and not in_line:
            in_line = True
            start_y = y
        elif not active and in_line:
            in_line = False
            end_y = y
            if end_y - start_y >= min_line_height:
                # Add safe 4px vertical margin to capture ascenders and sub-dots
                crop_y1 = max(0, start_y - 4)
                crop_y2 = min(page_image.height, end_y + 4)
                line_crop = page_image.crop((0, crop_y1, page_image.width, crop_y2))
                lines.append(line_crop)

    if in_line and (page_image.height - start_y >= min_line_height):
        crop_y1 = max(0, start_y - 4)
        line_crop = page_image.crop((0, crop_y1, page_image.width, page_image.height))
        lines.append(line_crop)

    return lines or [page_image]


class Recognizer:
    """Inference interface for document OCR."""

    def __init__(self, model: VisionEncoderDecoder, device: torch.device) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path | None = None,
        device: torch.device | None = None,
    ) -> Recognizer:
        """Load a trained OCR model from disk."""
        target_device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        config = ModelConfig()
        model = VisionEncoderDecoder(config=config, use_pretrained_encoder=False)

        if checkpoint_path and Path(checkpoint_path).is_file():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            # `strict=True`: a checkpoint whose vocabulary or depth differs from this config
            # must fail here. Loaded loosely it transcribes with random weights, which reads
            # as a very bad scan rather than as a wrong load.
            model.load_state_dict(state_dict, strict=True)

        return cls(model=model, device=target_device)

    def recognize_line(self, line_image: Image.Image) -> str:
        """Transcribe a single cropped text line image."""
        tensor = prepare_line_image(line_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model.generate(pixel_values=tensor)
        return preds[0] if preds else ""

    def recognize_lines(
        self,
        line_images: Sequence[Image.Image],
        batch_size: int = 16,
    ) -> list[str]:
        """Transcribe a sequence of cropped text line images in batches."""
        results: list[str] = []
        for i in range(0, len(line_images), batch_size):
            batch_imgs = line_images[i : i + batch_size]
            tensors = torch.stack(
                [prepare_line_image(img) for img in batch_imgs],
                dim=0,
            ).to(self.device)
            with torch.no_grad():
                preds = self.model.generate(pixel_values=tensors)
            results.extend(preds)
        return results

    def recognize_page(self, page_image: Image.Image) -> str:
        """Segment a scanned page into lines and transcribe into full text."""
        lines = segment_page_into_lines(page_image)
        transcriptions = self.recognize_lines(lines)
        return "\n".join(transcriptions)
