"""Verify complete CTC alignment across repeated and invalid labels."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from tools.speech_showcase import Showcase, Voice, Word, align_characters, curate

from agbalu.codec_tts.artifacts import digest_file
from agbalu.codec_tts.synthesis import Candidate, InferenceSettings, read_waveform, save_waveform
from agbalu.codec_tts.verification import Quality


def test_ctc_alignment_preserves_each_character_frame() -> None:
    emissions = np.full((5, 3), -20, dtype=np.float32)
    emissions[np.arange(5), [0, 1, 0, 2, 0]] = 0
    assert align_characters(emissions, [1, 2], 0) == [(1, 2), (3, 4)]


def test_ctc_repeated_labels_require_a_blank() -> None:
    emissions = np.full((3, 2), -20, dtype=np.float32)
    emissions[np.arange(3), [1, 0, 1]] = 0
    assert align_characters(emissions, [1, 1], 0) == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="needs 3 frames"):
        align_characters(emissions[:2], [1, 1], 0)


@pytest.mark.parametrize("tokens", [[], [0], [-1], [3]])
def test_ctc_invalid_labels_fail_explicitly(tokens: list[int]) -> None:
    with pytest.raises(ValueError, match="invalid CTC"):
        align_characters(np.zeros((5, 3), dtype=np.float32), tokens, 0)


def test_ctc_nan_positive_control_is_rejected() -> None:
    emissions = np.zeros((3, 2), dtype=np.float32)
    emissions[1, 0] = np.nan
    with pytest.raises(ValueError, match="invalid CTC input"):
        align_characters(emissions, [1], 0)


def test_word_interval_cannot_run_backwards() -> None:
    with pytest.raises(ValidationError, match="invalid word interval"):
        Word(text="Azul", start_s=2, end_s=1)


@pytest.mark.integration
def test_curation_preserves_pronunciation_and_prefers_voice_within_quality_band(
    tmp_path: Path,
) -> None:
    voice = Voice(
        key="aa",
        name="Demo",
        route="Kabyle to Kabyle",
        reference=tmp_path / "ref.wav",
        reference_text="Azul.",
        text="Azul.",
        identity="pseudonym",
        source="test",
    )
    spec = Showcase(
        output=tmp_path,
        voices=tuple(voice.model_copy(update={"key": key}) for key in ["aa", "bb", "cc", "dd"]),
    )
    directory = tmp_path / "aa.candidates"
    directory.mkdir()
    candidates = []
    for index, (cosine, mos, level, errors) in enumerate(
        [
            (0.3938, 4.59, 0.2, 0),
            (0.8, 4.2, 0.4, 0),
            (0.9, 4.5, 0.6, 1),
            (0.71, 4.9, 0.7, 0),
        ]
    ):
        path = directory / f"segment-0000-candidate-{index:02d}.wav"
        save_waveform(path, np.full(1200, level, dtype=np.float32))
        quality = Quality(
            hypothesis="azul",
            character_errors=errors,
            characters=4,
            voice_cosine=cosine,
            predicted_mos=mos,
            duration_s=0.05,
            peak=level,
            rms_dbfs=-20,
        )
        candidate = Candidate(
            request_sha256="a" * 64,
            seed=index,
            waveform_sha256=digest_file(path),
            quality=quality,
            trace={"terminated": True},
        )
        candidates.append({**candidate.model_dump(mode="json"), "rejected": []})
    audit = tmp_path / "aa.stats.json"
    audit.write_text(
        json.dumps(
            {
                "complete": True,
                "settings": InferenceSettings(
                    candidates=4, min_voice_cosine=0.7, max_cer=0, quality_tolerance=4
                ).model_dump(),
                "segments": [{"selected": 0, "candidates": candidates}],
            }
        )
    )
    before = digest_file(audit)
    output = curate(spec, voice)
    np.testing.assert_allclose(read_waveform(output), 0.4, atol=1e-7)
    report = json.loads(output.with_suffix(".stats.json").read_text())
    assert report["selected"] == 1
    assert report["quality"]["character_errors"] == 0
    assert digest_file(audit) == before
    broken = directory / "segment-0000-candidate-01.wav"
    broken.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="waveform changed"):
        curate(spec, voice)
