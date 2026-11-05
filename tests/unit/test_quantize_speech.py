"""Verify quantization accounting and benchmark metric definitions."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from tools.quantize_speech import (
    Trial,
    metric_seconds,
    summarize,
    validate_codec_types,
    validate_quantized_modules,
)

from agbalu.codec_tts.verification import Quality


def test_aggregate_metrics_weight_characters_and_audio_duration() -> None:
    first = Quality(
        hypothesis="a",
        character_errors=1,
        characters=1,
        voice_cosine=0.8,
        predicted_mos=4,
        duration_s=1,
        peak=0.8,
        rms_dbfs=-20,
    )
    second = first.model_copy(update={"character_errors": 0, "characters": 9, "voice_cosine": 0.6})
    result = summarize(
        [
            Trial(case="a", seed=1, output=Path("a.wav"), seconds=1, duration_s=1, quality=first),
            Trial(case="b", seed=1, output=Path("b.wav"), seconds=9, duration_s=3, quality=second),
            Trial(case="c", seed=1, output=Path("c.wav"), error="cap"),
        ]
    )
    assert result["cer"] == 0.1
    assert result["rtf_audio_weighted"] == 2.5
    assert result["voice_cosine_mean"] == 0.7
    assert result["completed"] == 2
    assert result["failed"] == 1
    assert result["characters"] == 10


def test_unmeasured_metrics_are_not_zero() -> None:
    result = summarize([])
    assert result["cer"] is None
    assert result["rtf_audio_weighted"] is None
    assert result["predicted_mos_mean"] is None


def test_trial_updates_validate_measured_values() -> None:
    trial = Trial(case="a", seed=1, output=Path("a.wav"))
    trial.seconds = 1.25
    assert trial.seconds == 1.25
    with pytest.raises(ValidationError, match="seconds"):
        trial.seconds = 0


def test_codec_quantization_positive_control_is_rejected() -> None:
    with pytest.raises(ValueError, match="audio_tokenizer"):
        validate_quantized_modules(["audio_tokenizer.decoder"], 1)


def test_missing_quantized_projections_are_rejected() -> None:
    with pytest.raises(ValueError, match="1/252"):
        validate_quantized_modules(["generator.body.layer"], 252)


def test_logical_codec_types_detect_quantization_under_physical_aliases() -> None:
    name = "tied.embedding.modality_embeddings.0.model.semantic_model.encoder.layer.weight"
    assert validate_codec_types({name: "F32"}) == 1
    with pytest.raises(ValueError, match="semantic_model"):
        validate_codec_types({name: "Q8_0"})
    with pytest.raises(ValueError, match="missing"):
        validate_codec_types({"_audiocpp.398": "F32"})


def test_native_metrics_use_milliseconds() -> None:
    assert metric_seconds("metrics[native-short-s1].wall_ms=1250\n", "native-short-s1") == 1.25
    assert metric_seconds("metrics[other].wall_ms=1250\n", "native-short-s1") is None


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "broken"])
def test_native_invalid_metrics_fail_with_request_id(value: str) -> None:
    with pytest.raises(ValueError, match="case"):
        metric_seconds(f"metrics[case].wall_ms={value}\n", "case")
