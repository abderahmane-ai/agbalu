"""Verify one-pass speech keeps the text, the peak, and a repeated request."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from agbalu.codec_tts import synthesis
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.codec import Codec
from agbalu.codec_tts.configuration import Geometry
from agbalu.codec_tts.inference import Decoding, Sampling
from agbalu.codec_tts.layout import SPECIALS, SequenceLayout
from agbalu.codec_tts.synthesis import (
    InferenceSettings,
    SpeechRequest,
    SpeechSynthesis,
    save_waveform,
)
from agbalu.codec_tts.verification import Quality
from agbalu.codec_tts.waveforms import Waveform

pytestmark = pytest.mark.integration


class TestTokenizer:
    __test__ = False

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [8] * len(text.split())

    def get_added_vocab(self) -> dict[str, int]:
        return dict(zip(SPECIALS, range(5), strict=True))


class TestCodec(Codec):
    __test__ = False

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def encode(self, waveform: Waveform) -> Tensor:
        assert waveform.size > 0
        return torch.zeros(25, 8, dtype=torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        assert codes.shape == (4, 8)
        return torch.linspace(-1.2, 1.2, 96)


class TestVerifier:
    __test__ = False

    def assess(self, waveform: Waveform, text: str, reference: Waveform) -> Quality:
        assert reference.size == 24000
        return Quality(
            hypothesis=text.strip(),
            character_errors=0,
            characters=len(text.strip()),
            voice_cosine=0.9,
            predicted_mos=4.4,
            duration_s=waveform.size / 24000,
            peak=float(np.abs(waveform).max()),
            rms_dbfs=-10,
        )


@pytest.fixture
def engine() -> SpeechSynthesis:
    return SpeechSynthesis(
        Generator(
            Geometry(
                hidden_size=8,
                intermediate_size=12,
                layers=1,
                heads=1,
                kv_heads=1,
                head_dim=8,
                text_vocabulary=16,
            )
        ),
        TestCodec(),
        TestTokenizer(),
        TestVerifier(),
    )


@pytest.fixture
def speech_request(tmp_path: Path) -> SpeechRequest:
    path = tmp_path / "reference.wav"
    save_waveform(path, np.full(24000, 0.1, dtype=np.float32))
    return SpeechRequest(
        text="Azul fell-awen. Aseggas ameggaz fell-awen.",
        reference=path,
        reference_text="Ɣef waman.",
        checkpoint_sha256="a" * 64,
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    result: list[int] = []

    def generate(
        model: Generator,
        prompt: SequenceLayout,
        settings: Sampling,
        decoding: Decoding,
        contrast: SequenceLayout | None = None,
        *,
        deadline_monotonic: float | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        assert isinstance(model, Generator)
        assert prompt.positions > 0
        assert isinstance(decoding, Decoding)
        assert contrast is None
        assert deadline_monotonic is None
        assert settings.max_new_positions == 1500
        result.append(settings.seed)
        return torch.ones(4, 8, dtype=torch.long), {"terminated": True, "frames": 4}

    monkeypatch.setattr(synthesis, "generate_codes", generate)
    return result


def test_one_sample_is_cached_and_peak_normalized(
    tmp_path: Path, engine: SpeechSynthesis, speech_request: SpeechRequest, calls: list[int]
) -> None:
    settings = InferenceSettings()
    output = tmp_path / "speech.wav"
    report = engine.synthesize(speech_request, settings, output)
    assert report["complete"] is True
    assert report["samples"] == 96
    assert len(report["segments"]) == 1
    assert calls == [settings.sampling.seed]
    wave = synthesis.read_waveform(output)
    assert wave.size == 96
    assert float(np.abs(wave).max()) == pytest.approx(0.99)
    again = engine.synthesize(speech_request, settings, output)
    assert again["output_sha256"] == report["output_sha256"]
    assert calls == [settings.sampling.seed]


def test_changed_request_and_corrupt_output_are_detected(
    tmp_path: Path, engine: SpeechSynthesis, speech_request: SpeechRequest, calls: list[int]
) -> None:
    output = tmp_path / "speech.wav"
    settings = InferenceSettings()
    engine.synthesize(speech_request, settings, output)
    with pytest.raises(FileExistsError, match="different speech request"):
        engine.synthesize(
            speech_request.model_copy(update={"text": "Different."}), settings, output
        )
    output.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        engine.synthesize(speech_request, settings, output)
    assert len(calls) == 1


def test_silent_reference_is_rejected_before_generation(
    tmp_path: Path, engine: SpeechSynthesis, speech_request: SpeechRequest, calls: list[int]
) -> None:
    save_waveform(speech_request.reference, np.zeros(24000, dtype=np.float32))
    with pytest.raises(ValueError, match="reference audio is silent"):
        engine.synthesize(speech_request, InferenceSettings(), tmp_path / "speech.wav")
    assert calls == []
