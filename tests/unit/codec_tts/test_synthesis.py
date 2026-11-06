"""Verify joint selection and lossless text boundaries."""

import numpy as np
import pytest

from agbalu.codec_tts.inference import Decoding
from agbalu.codec_tts.synthesis import (
    Candidate,
    GenerationStrategy,
    InferenceSettings,
    candidate_settings,
    choose_candidate,
    close_gaps,
    split_text,
)
from agbalu.codec_tts.verification import Quality


def candidate(errors: int = 0, voice: float = 0.8, mos: float = 4.3) -> Candidate:
    return Candidate(
        request_sha256="a" * 64,
        seed=1,
        waveform_sha256="b" * 64,
        quality=Quality(
            hypothesis="azul",
            character_errors=errors,
            characters=100,
            voice_cosine=voice,
            predicted_mos=mos,
            duration_s=2,
            peak=0.5,
            rms_dbfs=-10,
        ),
        trace={"terminated": True},
    )


def test_voice_similarity_cannot_override_content_or_quality_failures() -> None:
    settings = InferenceSettings()
    values = [
        candidate(40, 0.99),
        candidate(0, 0.99, 2),
        candidate(0, 0.85, 4.4),
        candidate(0, 0.95, 3.6),
    ]
    assert choose_candidate(values, settings) == 2
    assert choose_candidate(values[:2], settings) is None


def test_selection_prefers_voice_inside_content_and_quality_tolerances() -> None:
    assert (
        choose_candidate(
            [candidate(0, 0.7), candidate(4, 0.9), candidate(19, 0.95)], InferenceSettings()
        )
        == 1
    )
    assert choose_candidate([], InferenceSettings()) is None


@pytest.mark.parametrize(
    "text",
    [
        "Azul fell-awen. Aseggas ameggaz fell-awen.",
        "Ɣef\u00a0waman, aḍar d uḍar; ɛerḍ ameqqran. " * 5,
        "Aḍar.\n" + " " * 40,
    ],
)
def test_segmentation_accounts_for_every_original_character(text: str) -> None:
    parts = split_text(text, 32)
    assert "".join(parts) == text
    assert all(part.strip() for part in parts)


def test_long_unbroken_token_fails_with_its_offset() -> None:
    with pytest.raises(ValueError, match="character=0"):
        split_text("x" * 100, 32)


def test_strategy_pool_and_recovery_are_explicit_and_duration_bounded() -> None:
    settings = InferenceSettings(
        candidates=3,
        strategies=(
            GenerationStrategy(adapter_multiplier=1, decoding=Decoding()),
            GenerationStrategy(adapter_multiplier=0.65, decoding=Decoding()),
            GenerationStrategy(adapter_multiplier=1, decoding=Decoding(alignment_strength=1)),
        ),
    )
    variants = [candidate_settings(settings, "Azul fell-awen.", 3, 20, index) for index in range(4)]
    assert variants[0].adapter_multiplier == 1
    assert variants[1].adapter_multiplier == 0.65
    assert variants[2].decoding.alignment_strength == 1
    assert variants[3].decoding.recover_loops
    assert variants[0].sampling.max_new_positions == 257
    assert settings.sampling.max_new_positions == 1500
    with pytest.raises(ValueError, match="reference_characters"):
        candidate_settings(settings, "Azul.", 3, 0, 0)


def test_internal_silence_longer_than_a_breath_is_shortened() -> None:
    rate = 24000
    tone = np.full(rate, 0.2, dtype=np.float32)
    audio = np.concatenate((tone, np.zeros(rate, dtype=np.float32), tone))
    result = close_gaps(audio)
    assert rate * 2 < result.size < rate * 2 + int(0.25 * rate)
    brief = np.concatenate((tone, np.zeros(int(0.1 * rate), dtype=np.float32), tone))
    assert close_gaps(brief).size == brief.size
