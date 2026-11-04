"""Encode and reconstruct native eight-stream speech codes."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

from agbalu.codec_tts.layout import CODEBOOKS, validate_codes
from agbalu.codec_tts.waveforms import Waveform, validate_waveform

PREFIX = "tied.embedding.modality_embeddings.0.model."
UNUSED_PRETRAINING_KEY = "semantic_model.masked_spec_embed"
AUDIO_AXES = 3


class Codec:
    def __init__(self, model: HiggsAudioV2TokenizerModel, device: torch.device) -> None:
        nn.Module.to(model, device=device, dtype=torch.float32)
        self.model = model
        self.model.train(False)
        self.model.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def encode(self, waveform: Waveform) -> Tensor:
        validate_waveform(waveform, 24000)
        audio = torch.from_numpy(waveform.copy()).to(self.device).reshape(1, 1, -1)
        audio = functional.pad(audio, (0, max(0, 24000 - audio.shape[-1])))
        output = self.model.encode(audio)
        codes = getattr(output, "audio_codes", None)
        if (
            not isinstance(codes, Tensor)
            or codes.ndim != AUDIO_AXES
            or codes.shape[:2] != (1, CODEBOOKS)
        ):
            message = f"unexpected codec encoder output: {type(codes).__name__}"
            raise ValueError(message)
        result = codes[0].transpose(0, 1).long().cpu()
        validate_codes(result)
        return result

    @torch.inference_mode()
    def decode(self, codes: Tensor) -> Tensor:
        validate_codes(codes)
        output = self.model.decode(codes.transpose(0, 1).unsqueeze(0).to(self.device))
        values = getattr(output, "audio_values", None)
        if (
            not isinstance(values, Tensor)
            or values.ndim != AUDIO_AXES
            or values.shape[:2] != (1, 1)
        ):
            message = f"unexpected codec decoder output: {type(values).__name__}"
            raise ValueError(message)
        waveform = values[0, 0].float().cpu()
        if not waveform.numel() or not bool(torch.isfinite(waveform).all().item()):
            message = "codec decoder returned empty/non-finite waveform"
            raise ValueError(message)
        return waveform


def load_codec(
    checkpoint_path: Path, config_path: Path, device: torch.device
) -> tuple[Codec, dict[str, object]]:
    config = HiggsAudioV2TokenizerConfig.from_json_file(config_path)
    with torch.device("meta"):
        model = HiggsAudioV2TokenizerModel(config)
    expected = model.state_dict()
    tensors: dict[str, Tensor] = {}
    dropped: list[str] = []
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        names = checkpoint.keys()
        for name in names:
            if not name.startswith(PREFIX):
                continue
            key = name.removeprefix(PREFIX)
            if key == UNUSED_PRETRAINING_KEY:
                dropped.append(name)
                continue
            tensor = checkpoint.get_tensor(name)
            if key not in expected or tensor.shape != expected[key].shape:
                message = f"unexpected codec tensor {checkpoint_path}:{name}, shape={tensor.shape}"
                raise ValueError(message)
            tensors[key] = tensor.float()
    model.load_state_dict(tensors, strict=True, assign=True)
    if any(value.is_meta for value in model.buffers()):
        # Materialize derived semantic-model position buffers through a normal construction.
        materialized = HiggsAudioV2TokenizerModel(config)
        materialized.load_state_dict(tensors, strict=True)
        model = materialized
    return Codec(model, device), {"loaded": sorted(tensors), "dropped": dropped}
