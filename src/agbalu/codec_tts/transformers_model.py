"""Expose verified speech generation through Transformers AutoClasses."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.utils.generic import ModelOutput

from agbalu.codec_tts.adaptation import inject_adapter
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.codec import Codec
from agbalu.codec_tts.configuration import Geometry
from agbalu.codec_tts.synthesis import (
    InferenceSettings,
    SpeechRequest,
    read_waveform,
    render_speech,
)


def compute_weight_identity(checkpoint: str, quantization: object, has_adapter: bool) -> str:
    payload = {"checkpoint": checkpoint, "quantization": quantization, "has_adapter": has_adapter}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SpeechConfig(PreTrainedConfig):
    model_type = "agbalu_speech"

    def __init__(
        self,
        geometry: dict[str, object] | None = None,
        audio_configuration: dict[str, object] | None = None,
        inference: dict[str, object] | None = None,
        checkpoint_sha256: str = "0" * 64,
        source_stamp: dict[str, object] | None = None,
        has_adapter: bool = True,
        **kwargs: object,
    ) -> None:
        self.geometry = Geometry.model_validate(geometry or {}).model_dump()
        self.audio_configuration = audio_configuration or {}
        self.inference = InferenceSettings.model_validate(inference or {}).model_dump()
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_stamp = source_stamp or {}
        self.has_adapter = has_adapter
        self.sampling_rate = 24000
        super().__init__(**kwargs)


@dataclass
class SpeechOutput(ModelOutput):
    waveform: Tensor | None = None
    sampling_rate: int | None = None
    report: dict[str, object] | None = None


class SpeechModel(PreTrainedModel):
    config: SpeechConfig
    config_class = SpeechConfig
    main_input_name = "text"
    _keep_in_fp32_modules = ["audio_tokenizer"]
    _keep_in_fp32_modules_strict = ["audio_tokenizer", "factor_a", "factor_b"]
    _no_split_modules = ["Qwen3DecoderLayer", "AdaptedLinear"]

    def __init__(self, config: SpeechConfig) -> None:
        super().__init__(config)
        self.generator = Generator(Geometry.model_validate(config.geometry))
        if config.has_adapter:
            inject_adapter(self.generator)
        self.audio_tokenizer = HiggsAudioV2TokenizerModel(
            HiggsAudioV2TokenizerConfig(**config.audio_configuration)
        )
        self.post_init()

    def save_pretrained(self, save_directory: str | Path, **kwargs: object) -> None:
        commit: object = getattr(self.config, "_commit_hash", None)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.name_or_path,
            revision=commit if isinstance(commit, str) else "main",
            trust_remote_code=True,
        )
        super().save_pretrained(save_directory, **kwargs)
        tokenizer.save_pretrained(save_directory)

    @torch.inference_mode()
    def forward(
        self,
        text: str,
        reference_audio: str | Path,
        reference_text: str,
        output_path: str | Path | None = None,
    ) -> SpeechOutput:
        return self.generate(text, reference_audio, reference_text, output_path)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        reference_audio: str | Path,
        reference_text: str,
        output_path: str | Path | None = None,
    ) -> SpeechOutput:
        request = SpeechRequest(
            text=text,
            reference=Path(reference_audio),
            reference_text=reference_text,
            checkpoint_sha256=compute_weight_identity(
                self.config.checkpoint_sha256,
                self.config.to_dict().get("quantization_config"),
                self.config.has_adapter,
            ),
        )
        if not request.reference.is_file():
            message = f"reference audio missing: {request.reference}"
            raise FileNotFoundError(message)
        settings = InferenceSettings.model_validate(self.config.inference)
        device = self.generator.audio_embedding.weight.device
        commit: object = getattr(self.config, "_commit_hash", None)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.name_or_path,
            revision=commit if isinstance(commit, str) else "main",
            trust_remote_code=True,
        )
        with tempfile.TemporaryDirectory(prefix="speech-") as temporary:
            destination = (
                Path(output_path) if output_path is not None else Path(temporary) / "speech.wav"
            )
            report = render_speech(
                self.generator,
                Codec(self.audio_tokenizer, device),
                tokenizer,
                request,
                settings,
                destination,
                provenance_stamp=self.config.source_stamp,
            )
            waveform = torch.from_numpy(read_waveform(destination)).unsqueeze(0)
        return SpeechOutput(waveform=waveform, sampling_rate=24000, report=report)
