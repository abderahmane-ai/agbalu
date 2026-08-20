"""Vision-Encoder-Decoder for document OCR.

Published as `agbalu/Feraoun-36M`. The release name is not a code identifier, so
the classes here are named for what they configure and execute.

Two decisions shape the model:

1. **Character-level autoregressive decoding.** A subword tokenizer hallucinates vocabulary
   words where ink is degraded or fragmented. The 171-symbol table in `agbalu.ocr.vocabulary`
   operates strictly on perceived glyphs and puts the logit projection at 171 x 384, which
   is 0.25 MB in fp32.

2. **A 64 x 512 line canvas, resampled to the encoder's square input.** The renderer scales
   each line to 52 px of usable height so the sub-dots on `ḍ ḥ ṛ ṣ ṭ ẓ` survive as distinct
   dark pixels rather than the 1-2 they occupy at 32 px. `extract_features` interpolates that
   canvas to the DeiT backbone's 384 x 384, giving 576 patches plus the class and
   distillation tokens — **578 memory positions**. The 8:1 line aspect ratio is lost at that
   interpolation, which is the first thing to change if the decoder is ever memory-bound.
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

import torch
from torch import Tensor, nn

from agbalu.ocr.config import ModelConfig
from agbalu.ocr.vocabulary import BOS_ID, EOS_ID, PAD_ID, decode

LABEL_IGNORE_ID: Final[int] = -100


class LossOutput(NamedTuple):
    """Loss, logits, and encoder memory returned by a forward step."""

    loss: Tensor | None
    logits: Tensor
    encoder_hidden_states: Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over character positions."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # `get_buffer` rather than the attribute: `nn.Module.__getattr__` is typed as
        # `Tensor | Module`, and a registered buffer is always the former.
        pe = self.get_buffer("pe")
        x = x + pe[:, : x.size(1), :]
        out: Tensor = self.dropout(x)
        return out


ENCODER_LAYERS: Final[int] = 12
ENCODER_IMAGE_SIZE: Final[int] = 384
ENCODER_PATCH_SIZE: Final[int] = 16
"""The DeiT backbone's own shapes, which the published checkpoint's tensors are cut to —
`encoder.embeddings.position_embeddings` is (1, 578, 384) and 578 is
`(384 / 16) ** 2 + 2`. They are constants rather than `ModelConfig` fields because the
config is pickled into every checkpoint, so adding a field leaves the ones already written
without it."""


class VisionEncoderDecoder(nn.Module):
    """Vision-Encoder-Decoder document OCR model."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        use_pretrained_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        d_model = self.config.hidden_size

        # Both branches produce the same tensor shapes, because a run resumes across them:
        # the release trains from the pretrained backbone and every local load rebuilds it
        # from the random one. Neither import is guarded — a substituted encoder is a
        # different model with a healthy-looking loss curve.
        if use_pretrained_encoder:
            from transformers import VisionEncoderDecoderModel

            pretrained = VisionEncoderDecoderModel.from_pretrained(self.config.encoder_backbone)
            self.encoder: nn.Module = pretrained.encoder
            enc_dim = getattr(self.encoder.config, "hidden_size", d_model)
            self.encoder_proj: nn.Module = (
                nn.Linear(enc_dim, d_model) if enc_dim != d_model else nn.Identity()
            )
        else:
            from transformers import DeiTConfig, DeiTModel

            self.encoder = DeiTModel(
                DeiTConfig(
                    hidden_size=d_model,
                    num_hidden_layers=ENCODER_LAYERS,
                    num_attention_heads=self.config.num_decoder_heads,
                    intermediate_size=self.config.intermediate_size,
                    image_size=ENCODER_IMAGE_SIZE,
                    patch_size=ENCODER_PATCH_SIZE,
                )
            )
            self.encoder_proj = nn.Identity()

        self.char_embeddings = nn.Embedding(self.config.vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(
            d_model,
            max_len=self.config.max_length,
            dropout=self.config.dropout,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=self.config.num_decoder_heads,
            dim_feedforward=self.config.intermediate_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.config.num_decoder_layers,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.char_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def extract_features(self, pixel_values: Tensor) -> Tensor:
        """Extract spatial patch features from line images."""
        if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "image_size"):
            expected = self.encoder.config.image_size
            if isinstance(expected, int):
                target_size = (expected, expected)
            elif isinstance(expected, (list, tuple)):
                target_size = (int(expected[0]), int(expected[-1]))
            else:
                target_size = (384, 384)
            if pixel_values.shape[-2:] != target_size:
                pixel_values = torch.nn.functional.interpolate(
                    pixel_values,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

        encoder_out = self.encoder(pixel_values)
        if hasattr(encoder_out, "last_hidden_state"):
            hidden = encoder_out.last_hidden_state
        elif isinstance(encoder_out, Tensor):
            hidden = encoder_out
        else:
            hidden = encoder_out[0]
        projected: Tensor = self.encoder_proj(hidden)
        return projected

    def forward(
        self,
        pixel_values: Tensor,
        labels: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
    ) -> LossOutput:
        """Compute character logits and optional cross-entropy loss."""
        memory = self.extract_features(pixel_values)

        if decoder_input_ids is None and labels is not None:
            bos = torch.full((labels.size(0), 1), BOS_ID, dtype=labels.dtype, device=labels.device)
            decoder_input_ids = torch.cat([bos, labels[:, :-1]], dim=1)
            decoder_input_ids = decoder_input_ids.masked_fill(
                decoder_input_ids == LABEL_IGNORE_ID,
                PAD_ID,
            )

        if decoder_input_ids is None:
            msg = "Either labels or decoder_input_ids must be provided"
            raise ValueError(msg)

        seq_len = decoder_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=decoder_input_ids.device,
        ).bool()
        tgt_key_padding_mask = (decoder_input_ids == PAD_ID).bool()

        tgt_emb = self.char_embeddings(decoder_input_ids)
        tgt_emb = self.pos_encoder(tgt_emb)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        out = self.final_norm(out)
        logits: Tensor = self.lm_head(out)

        loss: Tensor | None = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE_ID)
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return LossOutput(loss=loss, logits=logits, encoder_hidden_states=memory)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Tensor,
        max_length: int | None = None,
        temperature: float = 0.0,
    ) -> list[str]:
        """Greedy autoregressive character decoding from visual features.

        `max_length` defaults to the config's, not to a shorter constant: the positional
        encoding is sized for `config.max_length`, and a decode capped below it silently
        truncates every longer line into a character error the model did not make.
        """
        self.eval()
        max_length = self.config.max_length if max_length is None else max_length
        device = pixel_values.device
        batch_size = pixel_values.size(0)

        memory = self.extract_features(pixel_values)
        current_ids = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            seq_len = current_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device).bool()
            tgt_emb = self.pos_encoder(self.char_embeddings(current_ids))

            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.lm_head(self.final_norm(out[:, -1:, :]))

            if temperature > 0.0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            next_token = torch.where(
                unfinished.unsqueeze(1),
                next_token,
                torch.full_like(next_token, PAD_ID),
            )
            current_ids = torch.cat([current_ids, next_token], dim=1)

            is_eos = next_token.squeeze(1) == EOS_ID
            unfinished = unfinished & (~is_eos)
            if not unfinished.any():
                break

        transcriptions: list[str] = []
        for row in current_ids:
            clean_ids = [tok.item() for tok in row if tok.item() not in (BOS_ID, EOS_ID, PAD_ID)]
            transcriptions.append(decode(clean_ids))

        return transcriptions
