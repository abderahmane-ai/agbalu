"""Feraoun-36M: a Vision-Encoder-Decoder that reads a line of printed Kabyle.

Module attribute names are the published checkpoint's `state_dict` keys. Renaming one
breaks `from_pretrained` for everybody who downloaded the release.

The line geometry lives here too, in `prepare_line_image`, because it is not recoverable
from the weights: the model was fitted on lines scaled to 52 px of usable height on a
64 x 512 canvas, and a caller who feeds it anything else is measuring a different model.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .configuration_feraoun import FeraounConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

TARGET_HEIGHT: Final[int] = 64
TARGET_WIDTH: Final[int] = 512
USABLE_HEIGHT: Final[int] = TARGET_HEIGHT - 12
USABLE_WIDTH: Final[int] = TARGET_WIDTH - 16
LEFT_PADDING: Final[int] = 8
CROP_INK_THRESHOLD: Final[int] = 225
LINE_INK_THRESHOLD: Final[int] = 200
LINE_MARGIN: Final[int] = 4
LABEL_IGNORE_ID: Final[int] = -100


def crop_text_bbox(image: Image.Image, padding: int = LINE_MARGIN) -> Image.Image:
    """Crop to the bounding box of the ink, so page margin does not consume the canvas."""
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
    """One line crop as the (3, 64, 512) tensor the encoder was fitted on.

    Scaled by height and padded, never stretched: the aspect ratio carries the letterforms,
    and a line squashed to fill the canvas is not what the weights saw.
    """
    import numpy as np
    from PIL import Image as PILImage

    cropped = crop_text_bbox(image).convert("RGB")
    width, height = cropped.size

    scale = USABLE_HEIGHT / max(height, 1)
    new_width, new_height = int(width * scale), int(height * scale)
    if new_width > USABLE_WIDTH:
        scale = USABLE_WIDTH / max(width, 1)
        new_width, new_height = int(width * scale), int(height * scale)
    new_width = max(1, min(new_width, USABLE_WIDTH))
    new_height = max(1, min(new_height, USABLE_HEIGHT))

    resized = cropped.resize((new_width, new_height), resample=PILImage.Resampling.LANCZOS)
    canvas = PILImage.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=(255, 255, 255))
    canvas.paste(resized, (LEFT_PADDING, (TARGET_HEIGHT - new_height) // 2))

    array = np.array(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) / 0.5).permute(2, 0, 1)


def segment_page_into_lines(page: Image.Image, min_line_height: int = 15) -> list[Image.Image]:
    """Split a page on its horizontal ink profile.

    There is no column detection and no reading-order model: a multi-column page is read
    straight across, which the card states.
    """
    import numpy as np

    array = np.array(page.convert("L"), dtype=np.uint8)
    ink = (array < LINE_INK_THRESHOLD).astype(np.float32).sum(axis=1)
    active = ink > 0.015 * array.shape[1]

    lines: list[Image.Image] = []
    start = 0
    inside = False
    for row, is_text in enumerate(active):
        if is_text and not inside:
            inside, start = True, row
        elif not is_text and inside:
            inside = False
            if row - start >= min_line_height:
                top = max(0, start - LINE_MARGIN)
                bottom = min(page.height, row + LINE_MARGIN)
                lines.append(page.crop((0, top, page.width, bottom)))
    if inside and page.height - start >= min_line_height:
        lines.append(page.crop((0, max(0, start - LINE_MARGIN), page.width, page.height)))
    return lines or [page]


class PositionalEncoding(nn.Module):
    """Sinusoidal positions over character indices."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        table = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        table[:, 0::2] = torch.sin(position * divisor)
        table[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", table.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        # `get_buffer` rather than the attribute: `nn.Module.__getattr__` is typed as
        # `Tensor | Module`, and a registered buffer is always the former.
        table = self.get_buffer("pe")
        encoded: Tensor = self.dropout(x + table[:, : x.size(1), :])
        return encoded


class FeraounForVision2Seq(PreTrainedModel):
    """The published architecture, and the two calls a downloader needs.

    `transcribe` takes images and returns text. `forward` takes the tensors, so a caller
    scoring the model can reach the logits and the loss directly.
    """

    config_class = FeraounConfig
    base_model_prefix = "feraoun"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: FeraounConfig) -> None:
        super().__init__(config)
        from transformers import DeiTConfig, DeiTModel

        width = config.hidden_size
        self.encoder = DeiTModel(
            DeiTConfig(
                hidden_size=width,
                num_hidden_layers=config.encoder_layers,
                num_attention_heads=config.num_decoder_heads,
                intermediate_size=config.intermediate_size,
                image_size=config.encoder_image_size,
                patch_size=config.encoder_patch_size,
            )
        )
        self.encoder_proj = nn.Identity()
        # Annotated as `nn.Module`, which is what `set_input_embeddings` is declared to
        # accept: `resize_token_embeddings` substitutes a wider table through it.
        self.char_embeddings: nn.Module = nn.Embedding(
            config.vocab_size, width, padding_idx=config.pad_token_id
        )
        self.pos_encoder = PositionalEncoding(
            width, max_len=config.max_length, dropout=config.dropout
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=width,
                nhead=config.num_decoder_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.num_decoder_layers,
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.char_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.char_embeddings = value

    def extract_features(self, pixel_values: Tensor) -> Tensor:
        """Patch features for a batch of line canvases, resampled to the encoder's field."""
        target = (self.config.encoder_image_size, self.config.encoder_image_size)
        if pixel_values.shape[-2:] != target:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=target, mode="bilinear", align_corners=False
            )
        hidden = self.encoder(pixel_values).last_hidden_state
        projected: Tensor = self.encoder_proj(hidden)
        return projected

    def forward(
        self,
        pixel_values: Tensor,
        labels: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        memory = self.extract_features(pixel_values)

        if decoder_input_ids is None and labels is not None:
            start = torch.full(
                (labels.size(0), 1),
                self.config.decoder_start_token_id,
                dtype=labels.dtype,
                device=labels.device,
            )
            decoder_input_ids = torch.cat([start, labels[:, :-1]], dim=1).masked_fill(
                torch.cat([start, labels[:, :-1]], dim=1) == LABEL_IGNORE_ID,
                self.config.pad_token_id,
            )
        if decoder_input_ids is None:
            message = "either labels or decoder_input_ids must be given"
            raise ValueError(message)

        length = decoder_input_ids.size(1)
        logits = self.lm_head(
            self.final_norm(
                self.decoder(
                    tgt=self.pos_encoder(self.char_embeddings(decoder_input_ids)),
                    memory=memory,
                    tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                        length, device=decoder_input_ids.device
                    ).bool(),
                    tgt_key_padding_mask=(decoder_input_ids == self.config.pad_token_id).bool(),
                )
            )
        )

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE_ID)(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        return Seq2SeqLMOutput(loss=loss, logits=logits, encoder_last_hidden_state=memory)

    @torch.no_grad()
    def generate_ids(self, pixel_values: Tensor, max_length: int | None = None) -> Tensor:
        """Greedy character decoding. Returns ids including the start and stop symbols."""
        self.eval()
        limit = max_length or self.config.max_length
        device = pixel_values.device
        rows = pixel_values.size(0)

        memory = self.extract_features(pixel_values)
        ids = torch.full(
            (rows, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device
        )
        running = torch.ones(rows, dtype=torch.bool, device=device)

        for _ in range(limit):
            hidden = self.decoder(
                tgt=self.pos_encoder(self.char_embeddings(ids)),
                memory=memory,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    ids.size(1), device=device
                ).bool(),
            )
            step = self.lm_head(self.final_norm(hidden[:, -1:, :]))
            nxt = torch.argmax(step[:, -1, :], dim=-1, keepdim=True)
            nxt = torch.where(
                running.unsqueeze(1), nxt, torch.full_like(nxt, self.config.pad_token_id)
            )
            ids = torch.cat([ids, nxt], dim=1)
            running = running & (nxt.squeeze(1) != self.config.eos_token_id)
            if not running.any():
                break
        return ids

    @torch.no_grad()
    def transcribe(
        self,
        images: Sequence[Image.Image],
        tokenizer: Any,
        batch_size: int = 16,
    ) -> list[str]:
        """Line images to text, preprocessing included.

        Feed it a line or a page crop, not a single word: every input is scaled to 52 px of
        usable height, and one word blown up to that height is nothing the model has seen.
        """
        out: list[str] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([prepare_line_image(image) for image in batch]).to(self.device)
            out.extend(
                tokenizer.decode(row.tolist(), skip_special_tokens=True)
                for row in self.generate_ids(tensors)
            )
        return out

    @torch.no_grad()
    def transcribe_page(self, page: Image.Image, tokenizer: Any) -> str:
        """A scanned page, one transcribed line per output line."""
        return "\n".join(self.transcribe(segment_page_into_lines(page), tokenizer))


__all__ = [
    "FeraounForVision2Seq",
    "PositionalEncoding",
    "crop_text_bbox",
    "prepare_line_image",
    "segment_page_into_lines",
]
