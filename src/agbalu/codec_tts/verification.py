"""Measure generated Kabyle content, speaker identity and predicted speech quality."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from pydantic import Field
from torch import Tensor, nn
from torch.nn import functional
from torchaudio.compliance.kaldi import fbank
from torchaudio.models import squim_subjective_base
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from agbalu.codec_tts.artifacts import atomic_binary, digest_file, run_lock
from agbalu.codec_tts.configuration import Specification
from agbalu.codec_tts.evaluation import ASR_ID, ASR_REVISION, edit_count, extract_input
from agbalu.codec_tts.waveforms import Waveform, measure_signal, resample_waveform

VOICE_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
VOICE_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
VOICE_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
MINIMUM_SPEAKER_SAMPLES = 400
QUALITY_SHA256 = "dd36e1e4a5e3b36a5e7c32c936097b57b3221663a0529704171a3bb792699718"
INSTRUMENTS = {
    "asr": {"id": ASR_ID, "revision": ASR_REVISION},
    "speaker": {"id": VOICE_ID, "revision": VOICE_REVISION, "sha256": VOICE_SHA256},
    "speech_quality": {"id": "torchaudio/SQUIM_SUBJECTIVE", "sha256": QUALITY_SHA256},
}


def quality_checkpoint() -> Path:
    path = Path(torch.hub.get_dir()) / "torchaudio/models/squim_subjective_bvcc_daps.pth"
    with run_lock(path.parent):
        if not path.exists():
            url = "https://download.pytorch.org/torchaudio/models/squim_subjective_bvcc_daps.pth"
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with atomic_binary(path) as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(block)
        if digest_file(path) != QUALITY_SHA256:
            message = f"speech-quality checkpoint checksum mismatch at {path}"
            raise ValueError(message)
    return path


class Quality(Specification):
    hypothesis: str
    character_errors: int = Field(ge=0)
    characters: int = Field(ge=1)
    voice_cosine: float = Field(ge=-1, le=1)
    predicted_mos: float = Field(ge=1, le=5)
    duration_s: float = Field(gt=0)
    peak: float = Field(ge=0)
    rms_dbfs: float

    @property
    def cer(self) -> float:
        return self.character_errors / self.characters


def spoken_text(text: str) -> str:
    plain = unicodedata.normalize("NFC", re.sub(r"<\|[^<>]*\|>", "", text)).lower()
    result = " ".join("".join(char if char.isalnum() else " " for char in plain).split())
    if not result:
        message = f"no spoken text in {text!r}"
        raise ValueError(message)
    return result


def speaker_features(waveform: Waveform) -> NDArray[np.float32]:
    audio = resample_waveform(waveform, 24000, 16000)
    if audio.size < MINIMUM_SPEAKER_SAMPLES:
        message = f"speaker verification needs 400 samples at 16 kHz, got {audio.size}"
        raise ValueError(message)
    features: Tensor = fbank(
        torch.from_numpy(audio).reshape(1, -1) * (1 << 15),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    return np.asarray((features - features.mean(0)).unsqueeze(0).numpy(), dtype=np.float32)


class SpeechVerifier:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        path = Path(
            hf_hub_download(
                VOICE_ID, "voxceleb_resnet34_LM.onnx", revision=VOICE_REVISION, etag_timeout=30
            )
        )
        if digest_file(path) != VOICE_SHA256:
            message = f"speaker-model checksum mismatch at {path}"
            raise ValueError(message)
        options = ort.SessionOptions()
        options.inter_op_num_threads, options.intra_op_num_threads = 1, 4
        self.speaker = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.recognizer: nn.Module = Wav2Vec2ForCTC.from_pretrained(ASR_ID, revision=ASR_REVISION)
        self.quality: nn.Module = squim_subjective_base()
        self.quality.load_state_dict(
            torch.load(quality_checkpoint(), weights_only=True, map_location="cpu")
        )
        for model in (self.recognizer, self.quality):
            nn.Module.to(model, device=device)
            nn.Module.train(model, False)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(ASR_ID, revision=ASR_REVISION)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(ASR_ID, revision=ASR_REVISION)

    def embed(self, waveform: Waveform) -> Tensor:
        output: object = self.speaker.run(["embs"], {"feats": speaker_features(waveform)})[0]
        if (
            not isinstance(output, np.ndarray)
            or output.shape != (1, 256)
            or not np.isfinite(output).all()
        ):
            message = f"invalid speaker embedding from {VOICE_ID}"
            raise ValueError(message)
        tensor = torch.from_numpy(np.asarray(output, dtype=np.float32))
        if float(tensor.norm()) == 0:
            message = "speaker encoder returned a zero vector"
            raise ValueError(message)
        return functional.normalize(tensor, dim=-1)

    @torch.inference_mode()
    def assess(self, waveform: Waveform, text: str, reference: Waveform) -> Quality:
        signal = measure_signal(waveform, 24000)
        if signal.rms_dbfs is None:
            message = "generated speech is silent"
            raise ValueError(message)
        audio = resample_waveform(waveform, 24000, 16000)
        output = self.recognizer(input_values=extract_input(self.extractor, audio, self.device))
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            message = "ASR returned no logits"
            raise TypeError(message)
        raw = str(self.tokenizer.decode(logits[0].argmax(-1).tolist()))
        hypothesis = spoken_text(raw) if raw.strip() else ""
        expected = spoken_text(text)
        score = self.quality(
            torch.from_numpy(audio).reshape(1, -1).to(self.device),
            torch.from_numpy(resample_waveform(reference, 24000, 16000))
            .reshape(1, -1)
            .to(self.device),
        )
        if not isinstance(score, Tensor):
            message = "speech-quality model returned no MOS tensor"
            raise TypeError(message)
        return Quality(
            hypothesis=hypothesis,
            character_errors=edit_count(expected, hypothesis),
            characters=len(expected),
            voice_cosine=float((self.embed(waveform) * self.embed(reference)).sum().clamp(-1, 1)),
            predicted_mos=float(score[0]),
            duration_s=signal.samples / signal.rate_hz,
            peak=signal.peak,
            rms_dbfs=signal.rms_dbfs,
        )
