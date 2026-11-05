"""Verify pronunciation normalization and pinned speaker features."""

import numpy as np
import pytest
from pydantic import ValidationError

from agbalu.codec_tts.verification import Quality, speaker_features, spoken_text


def test_scoring_ignores_formatting_and_preserves_kabyle_distinctions() -> None:
    assert spoken_text("<|style:whispering|>AḍAR, Ɣef ɛerḍ!") == "aḍar ɣef ɛerḍ"
    assert spoken_text("Ɣef\u00a0waman.") == "ɣef waman"
    assert spoken_text("Ḍ") != spoken_text("D")


@pytest.mark.parametrize("text", ["", "  ", "<|style:whispering|>", "...!"])
def test_nonspoken_inputs_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="no spoken text"):
        spoken_text(text)


def test_speaker_features_use_mean_centered_eighty_bin_frontend() -> None:
    wave = np.sin(np.arange(2400) * 0.1).astype(np.float32)
    features = speaker_features(wave)
    assert features.shape == (1, 8, 80)
    np.testing.assert_allclose(features.mean(1), 0, atol=2e-6)
    with pytest.raises(ValueError, match="400 samples"):
        speaker_features(wave[:30])


def test_nonfinite_quality_does_not_pass_validation() -> None:
    with pytest.raises(ValidationError):
        Quality(
            hypothesis="azul",
            character_errors=0,
            characters=4,
            voice_cosine=float("nan"),
            predicted_mos=4,
            duration_s=1,
            peak=0.5,
            rms_dbfs=-10,
        )
