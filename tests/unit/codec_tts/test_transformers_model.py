"""Verify the public speech configuration and waveform return contract."""

import pytest
import torch
from pydantic import ValidationError

from agbalu.codec_tts.transformers_model import (
    SpeechConfig,
    SpeechModel,
    SpeechOutput,
    compute_weight_identity,
)


def test_speech_configuration_round_trip_preserves_settings() -> None:
    original = SpeechConfig(
        geometry={"layers": 2},
        inference={"candidates": 3},
        checkpoint_sha256="a" * 64,
        source_stamp={"git_sha": "test", "dirty": False},
    )
    restored = SpeechConfig.from_dict(original.to_dict())
    assert restored.geometry["layers"] == 2
    assert restored.inference["candidates"] == 3
    assert restored.checkpoint_sha256 == "a" * 64
    assert restored.source_stamp == {"git_sha": "test", "dirty": False}


def test_invalid_geometry_reports_the_field() -> None:
    with pytest.raises(ValidationError, match="layers"):
        SpeechConfig(geometry={"layers": 0})


def test_unknown_inference_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="candidate_typo"):
        SpeechConfig(inference={"candidate_typo": 6})


def test_merged_configuration_disables_adapter_construction() -> None:
    assert SpeechConfig().has_adapter is True
    restored = SpeechConfig.from_dict(SpeechConfig(has_adapter=False).to_dict())
    assert restored.has_adapter is False


def test_speech_output_exposes_waveform_and_sampling_rate() -> None:
    wave = torch.tensor([[0.0, 0.25, -0.5]])
    result = SpeechOutput(waveform=wave, sampling_rate=24000, report={"complete": True})
    assert result.waveform is not None
    assert torch.equal(result.waveform, wave)
    assert result.sampling_rate == 24000
    assert result.report == {"complete": True}


def test_speech_model_quantization_declarations() -> None:
    assert SpeechModel._keep_in_fp32_modules == ["audio_tokenizer"]
    assert SpeechModel._keep_in_fp32_modules_strict == ["audio_tokenizer", "factor_a", "factor_b"]
    assert SpeechModel._no_split_modules == ["Qwen3DecoderLayer", "AdaptedLinear"]


def test_quantization_and_merge_changes_invalidate_audio_cache() -> None:
    plain = compute_weight_identity("a" * 64, None, True)
    assert plain == compute_weight_identity("a" * 64, None, True)
    assert plain != compute_weight_identity("a" * 64, {"load_in_4bit": True}, True)
    assert plain != compute_weight_identity("a" * 64, None, False)
