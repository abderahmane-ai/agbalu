"""Matoub-82M: Kabyle text to a 24 kHz waveform.

A StyleTTS2 model fine-tuned from Kokoro-82M. The text and prosody modules are adapted
from `hexgrad/Kokoro-82M`'s `modules.py` (Apache-2.0), itself adapted from StyleTTS2's
`models.py` (MIT). Module attribute names are the published checkpoint's `state_dict`
keys; renaming one breaks `from_pretrained` for everybody who downloaded the release.

The speaker style is a 256-dim vector carried in the weights, so synthesis needs no
reference clip: the first 128 dimensions condition the waveform decoder and the second
128 condition duration and pitch. Style diffusion is not part of this checkpoint —
`lambda_diff` was 0.0 for every epoch — so there is no sampler to blend against and no
`alpha`/`beta` to set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from transformers import AlbertConfig, AlbertModel, PreTrainedModel
from transformers.utils.generic import ModelOutput

from .configuration_matoub import MatoubConfig
from .istftnet import AdainResBlk1d, Decoder

BATCHED_SEQUENCE_RANK: Final = 2


@dataclass
class MatoubOutput(ModelOutput):
    """Synthesised audio, and the frame count each input token was given.

    `waveform` is right-padded to the longest item in the batch; `waveform_lengths` says
    where each one ends.
    """

    waveform: Tensor | None = None
    waveform_lengths: Tensor | None = None
    durations: Tensor | None = None


class LayerNorm(nn.Module):
    """Channel-last layer norm over a (batch, channels, time) tensor."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, -1)
        x = nn.functional.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class LinearNorm(nn.Module):
    """A linear layer under the attribute name the checkpoint stores it by."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        projected: Tensor = self.linear_layer(x)
        return projected


class TextEncoder(nn.Module):
    """Phoneme ids to the acoustic features the decoder reads."""

    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                    LayerNorm(channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                )
                for _ in range(depth)
            ]
        )
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        for block in self.cnn:
            x = block(x)
        encoded: Tensor
        encoded, _ = self.lstm(x.transpose(1, 2))
        return encoded.transpose(-1, -2)


class AdaLayerNorm(nn.Module):
    """Layer norm whose scale and shift are read off the style vector."""

    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        x = x.transpose(-1, -2).transpose(1, -1)
        h = self.fc(s).view(s.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = nn.functional.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    """Style-conditioned recurrent stack the duration head reads."""

    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float) -> None:
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1).transpose(0, 1).transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
            else:
                x, _ = block(x.transpose(-1, -2))
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    """Per-token duration, and the pitch and energy contours over the expanded frames."""

    def __init__(
        self, style_dim: int, d_hid: int, nlayers: int, max_dur: int, dropout: float
    ) -> None:
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(
            d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True
        )
        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
            ]
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def contours(self, aligned: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
        x, _ = self.shared(aligned.transpose(-1, -2))
        pitch = x.transpose(-1, -2)
        for block in self.F0:
            pitch = block(pitch, s)
        energy = x.transpose(-1, -2)
        for block in self.N:
            energy = block(energy, s)
        return self.F0_proj(pitch).squeeze(1), self.N_proj(energy).squeeze(1)


class MatoubPreTrainedModel(PreTrainedModel):
    config_class = MatoubConfig
    base_model_prefix = "matoub"
    main_input_name = "input_ids"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Conv1d | nn.ConvTranspose1d):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)


class MatoubForTextToWaveform(MatoubPreTrainedModel):
    """`model(**tokenizer(text, return_tensors="pt")).waveform` — 24 kHz mono float32."""

    voice: Tensor

    def __init__(self, config: MatoubConfig) -> None:
        super().__init__(config)
        self.bert = AlbertModel(
            AlbertConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.plbert_hidden_size,
                num_attention_heads=config.plbert_num_attention_heads,
                intermediate_size=config.plbert_intermediate_size,
                num_hidden_layers=config.plbert_num_hidden_layers,
                max_position_embeddings=config.plbert_max_position_embeddings,
            )
        )
        self.bert_encoder = nn.Linear(config.plbert_hidden_size, config.hidden_size)
        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_size,
            nlayers=config.num_layers,
            max_dur=config.max_duration,
            dropout=config.dropout,
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_size,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.num_layers,
            n_symbols=config.vocab_size,
        )
        self.decoder = Decoder(
            dim_in=config.hidden_size,
            style_dim=config.style_dim,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            gen_istft_n_fft=config.gen_istft_n_fft,
            gen_istft_hop_size=config.gen_istft_hop_size,
            sampling_rate=config.sampling_rate,
        )
        self.register_buffer("voice", torch.zeros(1, config.style_dim * 2))
        self.post_init()

    @property
    def sampling_rate(self) -> int:
        return int(self.config.sampling_rate)

    def _synthesise(self, input_ids: Tensor, style: Tensor, speed: float) -> tuple[Tensor, Tensor]:
        attention = torch.ones_like(input_ids)
        bert_dur = self.bert(input_ids, attention_mask=attention).last_hidden_state
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        prosody_style = style[:, self.config.style_dim :]
        acoustic_style = style[:, : self.config.style_dim]

        d = self.predictor.text_encoder(d_en, prosody_style)
        x, _ = self.predictor.lstm(d)
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(dim=-1) / speed
        frames = torch.round(duration).clamp(min=1).long().squeeze(0)

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=input_ids.device), frames
        )
        alignment = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=input_ids.device, dtype=d.dtype
        )
        alignment[indices, torch.arange(indices.shape[0], device=input_ids.device)] = 1
        alignment = alignment.unsqueeze(0)

        pitch, energy = self.predictor.contours(d.transpose(-1, -2) @ alignment, prosody_style)
        asr = self.text_encoder(input_ids) @ alignment
        waveform = self.decoder(asr, pitch, energy, acoustic_style).squeeze(1).squeeze(0)
        return waveform, frames

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        speed: float = 1.0,
        voice: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> MatoubOutput | tuple[Tensor, Tensor, Tensor]:
        if speed <= 0:
            message = f"speed must be positive, got {speed}"
            raise ValueError(message)
        if input_ids.dim() != BATCHED_SEQUENCE_RANK:
            message = f"input_ids must be (batch, tokens), got shape {tuple(input_ids.shape)}"
            raise ValueError(message)
        limit = self.config.max_token_length
        if input_ids.shape[1] > limit:
            message = (
                f"{input_ids.shape[1]} tokens exceeds the {limit} PL-BERT can position; "
                f"synthesise one sentence at a time"
            )
            raise ValueError(message)

        style = self.voice if voice is None else voice.to(self.voice.dtype)
        if style.shape[-1] != self.config.style_dim * 2:
            message = (
                f"voice must be a {self.config.style_dim * 2}-dim style vector, "
                f"got shape {tuple(style.shape)}"
            )
            raise ValueError(message)
        style = style.reshape(1, -1).to(input_ids.device)

        # Each item is synthesised on its own: the alignment matrix that expands tokens to
        # frames is built from that item's own durations, so a padded row would be given
        # frames of its own padding.
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        waveforms: list[Tensor] = []
        durations: list[Tensor] = []
        for row, keep in zip(input_ids, mask, strict=True):
            tokens = row[keep.bool()].unsqueeze(0)
            waveform, frames = self._synthesise(tokens, style, speed)
            waveforms.append(waveform)
            durations.append(
                nn.functional.pad(frames, (0, int(input_ids.shape[1] - frames.shape[0])))
            )

        lengths = torch.tensor([w.shape[0] for w in waveforms], device=input_ids.device)
        longest = int(lengths.max())
        audio = torch.stack([nn.functional.pad(w, (0, longest - w.shape[0])) for w in waveforms])
        stacked_durations = torch.stack(durations)

        if return_dict is False:
            return audio, lengths, stacked_durations
        return MatoubOutput(waveform=audio, waveform_lengths=lengths, durations=stacked_durations)


__all__ = ["MatoubForTextToWaveform", "MatoubOutput", "MatoubPreTrainedModel"]
