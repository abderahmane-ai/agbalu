"""Feraoun-Pro: Hybrid CNN-Transformer CTC Document OCR Model for Kabyle."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from .configuration_feraoun_pro import FeraounProConfig

if TYPE_CHECKING:
    from PIL import Image

TARGET_HEIGHT: Final[int] = 64
MAX_WIDTH: Final[int] = 1024
CROP_INK_THRESHOLD: Final[int] = 225
LINE_INK_THRESHOLD: Final[int] = 200
LINE_MARGIN: Final[int] = 4


def crop_text_bbox(image: Image.Image, padding: int = LINE_MARGIN) -> Image.Image:
    """Crop to the bounding box of the ink."""
    import numpy as np

    array = np.array(image.convert("L"), dtype=np.uint8)
    ink = array < CROP_INK_THRESHOLD
    if not ink.any():
        return image

    rows, columns = np.where(ink)
    left = max(0, int(columns.min()) - padding)
    right = min(image.width, int(columns.max()) + padding)
    top = max(0, int(rows.min()) - padding)
    bottom = min(image.height, int(rows.max()) + padding)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def prepare_line_image(image: Image.Image) -> Tensor:
    """Crop to the ink, scale the height to 64 px, normalise, and clamp the width."""
    import numpy as np
    from PIL import Image as PILImage

    cropped = crop_text_bbox(image).convert("RGB")
    width, height = cropped.size

    scale = float(TARGET_HEIGHT) / max(height, 1)
    new_width = max(16, min(MAX_WIDTH, int(width * scale)))
    # Ensure divisible by 4 for CNN stem
    new_width = int(math.ceil(new_width / 4.0) * 4)

    resized = cropped.resize((new_width, TARGET_HEIGHT), resample=PILImage.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 64, W)


def segment_page_into_lines(page: Image.Image, min_line_height: int = 15) -> list[Image.Image]:
    """Segment page into line strips on horizontal ink profile."""
    import numpy as np

    array = np.array(page.convert("L"), dtype=np.uint8)
    ink = (array < LINE_INK_THRESHOLD).astype(np.float32).sum(axis=1)
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
            if idx - start >= min_line_height:
                lines.append(
                    page.crop(
                        (0, max(0, start - 2), page.width, min(page.height, idx + 2))
                    ).convert("RGB")
                )

    if inside and len(active) - start >= min_line_height:
        lines.append(page.crop((0, start, page.width, len(active))).convert("RGB"))

    return lines


class CNNStem(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.proj = nn.Linear(256 * 4, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        out: Tensor = self.proj(x)
        return out


class FeraounProForCTC(PreTrainedModel):
    config_class = FeraounProConfig
    base_model_prefix = "feraoun_pro"

    def __init__(self, config: FeraounProConfig) -> None:
        super().__init__(config)
        self.config = config
        self.stem = CNNStem(d_model=config.d_model)

        # Build sinusoidal positional encodings
        pe = torch.zeros(config.max_width_frames, config.d_model)
        position = torch.arange(config.max_width_frames).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2) * (-math.log(10000.0) / config.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The nested-tensor fast path does not apply to a pre-norm layer; torch infers that
        # and warns. Declaring it keeps a downloader's console clean.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers, enable_nested_tensor=False
        )
        self.ctc_head = nn.Linear(config.d_model, config.vocab_size)
        self.post_init()

    def forward(self, pixel_values: Tensor) -> Tensor:
        features = self.stem(pixel_values)
        seq_len = features.size(1)
        # `get_buffer` rather than the attribute: `nn.Module.__getattr__` is typed as
        # `Tensor | Module`, and a registered buffer is always the former.
        pe = self.get_buffer("pos_encoding")[:seq_len, :].unsqueeze(0).to(features.device)
        encoded = self.encoder(features + pe)
        logits: Tensor = self.ctc_head(encoded)
        return logits

    def decode_ctc(self, token_ids: list[int]) -> str:
        """Collapse repeats, then drop blanks — the CTC rule, in that order.

        Dropping blanks first merges the two characters a blank was separating, so a
        geminate consonant comes back single.
        """
        vocabulary = self.config.vocabulary
        blank = self.config.pad_token_id
        special = {blank, self.config.bos_token_id, self.config.eos_token_id}
        collapsed: list[int] = []
        previous = -1
        for token_id in token_ids:
            if token_id != previous:
                if token_id not in special:
                    collapsed.append(token_id)
                previous = token_id
        return "".join(vocabulary[token_id] for token_id in collapsed if token_id < len(vocabulary))

    @torch.no_grad()
    def transcribe(self, image: Image.Image) -> str:
        """Read one line crop. Preprocessing included, because it is half of this model."""
        self.eval()
        pixel_values = prepare_line_image(image).to(self.device)
        logits = self(pixel_values=pixel_values)
        return self.decode_ctc(torch.argmax(logits, dim=-1)[0].tolist())

    @torch.no_grad()
    def transcribe_page(self, page: Image.Image) -> list[str]:
        """Segment a page into lines and read each one."""
        return [self.transcribe(line) for line in segment_page_into_lines(page)]


__all__ = ["FeraounProForCTC", "prepare_line_image", "segment_page_into_lines"]
