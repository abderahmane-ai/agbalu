"""Verify consent gates and deterministic launch-video timing."""

import math

import pytest
from pydantic import ValidationError
from tools.produce_video import (
    Consent,
    Generation,
    Production,
    build_scenes,
    calculate_envelope,
    locate_scene,
    require_equal,
    subtitle_time,
)


def test_preview_accounts_for_every_frame() -> None:
    scenes = build_scenes(())
    counts = dict.fromkeys(scenes, 0)
    for frame in range(55 * 30):
        counts[locate_scene(scenes, frame / 30)] += 1
    assert sum(counts.values()) == 1650
    assert tuple(counts.values()) == (120, 240, 150, 510, 120, 210, 150, 150)
    assert all(scene.audio is None for scene in scenes)


@pytest.mark.parametrize("time_s", [-1, 55, math.inf, math.nan])
def test_scene_rejects_outside_time(time_s: float) -> None:
    with pytest.raises(ValueError, match="No scene"):
        locate_scene(build_scenes(()), time_s)


def test_conservation_check_detects_lost_frame() -> None:
    with pytest.raises(ValueError, match="expected 1650, received 1649"):
        require_equal(1649, 1650, "frame count")


def test_waveform_uses_actual_peaks() -> None:
    assert calculate_envelope((0.1, -0.8, 0.2, -0.4), 2) == (0.8, 0.4)
    assert calculate_envelope((0.0,), 1) == (0.0,)
    assert calculate_envelope((0.7,), 2) == (0.0, 0.7)


@pytest.mark.parametrize(
    ("values", "bins"), [((), 3), ((0.1,), 0), ((math.nan,), 1), ((math.inf,), 1)]
)
def test_waveform_rejects_invalid_values(values: tuple[float, ...], bins: int) -> None:
    with pytest.raises(ValueError, match="Invalid waveform"):
        calculate_envelope(values, bins)


def test_caption_timestamps_preserve_milliseconds() -> None:
    assert subtitle_time(0) == "00:00:00,000"
    assert subtitle_time(17.125) == "00:00:17,125"
    assert subtitle_time(3661.1) == "01:01:01,100"


def test_missing_consent_fails_closed() -> None:
    with pytest.raises(ValidationError, match="voice_cloning"):
        Consent.model_validate({"speaker_id": "speaker-a"})


def test_denied_consent_cannot_enter_production() -> None:
    with pytest.raises(ValidationError, match="voice_cloning"):
        Consent.model_validate(
            {
                "speaker_id": "speaker-a",
                "signed_by": "Speaker A",
                "signed_date": "2026-11-06",
                "reference_sha256": "a" * 64,
                "voice_cloning": False,
                "public_demonstration": True,
                "distribute_audio": True,
            }
        )


def test_local_generation_is_rejected() -> None:
    with pytest.raises(ValidationError, match="execution_target"):
        Generation.model_validate({"execution_target": "local"})


def test_production_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Production.model_validate(
            {
                "version": 1,
                "model_revision": "a" * 40,
                "samples": [],
                "publication_permission": None,
                "skip_consent": True,
            }
        )


def test_schema_instantiates_without_audio() -> None:
    production = Production(
        version=1, model_revision="a" * 40, samples=(), publication_permission=None
    )
    assert production.samples == ()
