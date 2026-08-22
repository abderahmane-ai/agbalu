"""Hybrid CNN-Transformer CTC Recognizer with Auxiliary GTC Training Branch for Feraoun-Pro."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.ocr.vocabulary_pro import PAD_ID, VOCAB_SIZE

DEFAULT_D_MODEL: Final[int] = 512
DEFAULT_NUM_LAYERS: Final[int] = 16
DEFAULT_NUM_HEADS: Final[int] = 8
DEFAULT_FFN_DIM: Final[int] = 2048
DEFAULT_MAX_WIDTH_FRAMES: Final[int] = 256  # 1024 px // 4


def build_sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    """Build 1D sinusoidal positional encoding table."""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CNNStem(nn.Module):
    """4-stage convolutional downsampling stem reducing (64, W) to (4, W/4)."""

    def __init__(self, d_model: int = DEFAULT_D_MODEL) -> None:
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
        # Height is reduced: 64 -> 32 -> 16 -> 8 -> 4. Feature map is (256, 4, W/4)
        self.proj = nn.Linear(256 * 4, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 3, 64, W)
        x = self.block1(x)  # (B, 32, 32, W)
        x = self.block2(x)  # (B, 64, 16, W)
        x = self.block3(x)  # (B, 128, 8, W/2)
        x = self.block4(x)  # (B, 256, 4, W/4)

        b, c, h, w_out = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w_out, c * h)  # (B, W/4, 1024)
        out: Tensor = self.proj(x)  # (B, W/4, d_model)
        return out


class AuxiliaryDecoder(nn.Module):
    """2-layer autoregressive Transformer decoder used purely for GTC training gradients."""

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        num_heads: int = DEFAULT_NUM_HEADS,
        dim_feedforward: int = DEFAULT_FFN_DIM,
        vocab_size: int = VOCAB_SIZE,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: Tensor, memory: Tensor) -> Tensor:
        # tgt_ids: (B, L), memory: (B, T, d_model)
        tgt_emb = self.embedding(tgt_ids)
        seq_len = tgt_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt_ids.device)
        decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
        logits: Tensor = self.lm_head(decoded)
        return logits


@dataclass(frozen=True, slots=True)
class ModelOutputOCRPro:
    ctc_logits: Tensor
    loss: Tensor | None = None
    ctc_loss: Tensor | None = None
    aux_loss: Tensor | None = None


class HybridCTCRecognizer(nn.Module):
    """Feraoun-Pro recognizer: CNN stem + Transformer encoder + CTC head + auxiliary GTC branch."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = DEFAULT_D_MODEL,
        num_encoder_layers: int = DEFAULT_NUM_LAYERS,
        num_heads: int = DEFAULT_NUM_HEADS,
        dim_feedforward: int = DEFAULT_FFN_DIM,
        include_aux_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.stem = CNNStem(d_model=d_model)
        self.register_buffer(
            "pos_encoding", build_sinusoidal_positions(DEFAULT_MAX_WIDTH_FRAMES, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Declared rather than left to be inferred: the nested-tensor fast path does not
        # apply to a pre-norm layer, and torch decides that itself and warns about it. The
        # suite runs with `filterwarnings = ["error"]`, so an inferred `True` here is a
        # failed test at every construction.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)

        self.aux_decoder = (
            AuxiliaryDecoder(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                vocab_size=vocab_size,
            )
            if include_aux_decoder
            else None
        )

        self.ctc_criterion = nn.CTCLoss(blank=PAD_ID, zero_infinity=True)

    def encode(self, pixel_values: Tensor) -> Tensor:
        """Extract encoder sequence features from input images."""
        features = self.stem(pixel_values)  # (B, T, d_model)
        seq_len = features.size(1)

        # `get_buffer` rather than the attribute: `nn.Module.__getattr__` is typed as
        # `Tensor | Module`, and a registered buffer is always the former.
        pe = self.get_buffer("pos_encoding")[:seq_len, :].unsqueeze(0).to(features.device)
        features = features + pe
        encoded: Tensor = self.encoder(features)
        return encoded

    def forward(
        self,
        pixel_values: Tensor,
        input_lengths: Tensor | None = None,
        ctc_targets: Tensor | None = None,
        target_lengths: Tensor | None = None,
        aux_input_ids: Tensor | None = None,
        aux_labels: Tensor | None = None,
    ) -> ModelOutputOCRPro:
        memory = self.encode(pixel_values)
        ctc_logits = self.ctc_head(memory)  # (B, T, vocab_size)

        loss: Tensor | None = None
        ctc_loss: Tensor | None = None
        aux_loss: Tensor | None = None

        if ctc_targets is not None and target_lengths is not None and input_lengths is not None:
            # CTC loss expects log_probs of shape (T, B, C)
            log_probs = functional.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
            ctc_loss = self.ctc_criterion(log_probs, ctc_targets, input_lengths, target_lengths)
            loss = ctc_loss

            if (
                self.aux_decoder is not None
                and aux_input_ids is not None
                and aux_labels is not None
            ):
                # Teacher forcing: the decoder reads up to position i and is scored on i+1.
                aux_inp = aux_input_ids[:, :-1]
                aux_tgt = aux_labels[:, 1:]
                aux_logits = self.aux_decoder(aux_inp, memory)  # (B, L-1, vocab_size)
                aux_loss = functional.cross_entropy(
                    aux_logits.reshape(-1, self.vocab_size),
                    aux_tgt.reshape(-1),
                    ignore_index=-100,
                )
                loss = ctc_loss + 0.5 * aux_loss

        return ModelOutputOCRPro(
            ctc_logits=ctc_logits,
            loss=loss,
            ctc_loss=ctc_loss,
            aux_loss=aux_loss,
        )
