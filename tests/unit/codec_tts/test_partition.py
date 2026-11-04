"""Prevent speaker and script leakage between codec speech splits."""

import pytest

from agbalu.codec_tts.artifacts import Recording
from agbalu.codec_tts.configuration import Recipe
from agbalu.codec_tts.partition import partition_recordings, speaker_split


def recording(index: int, speaker: str, text: str) -> Recording:
    return Recording(
        record_id=str(index),
        shard="a.parquet",
        row=index,
        audio_sha256=f"audio-{index}",
        speaker=speaker,
        text=text,
        text_key=text,
        stored_rate_hz=48000,
        samples=48000,
        prepared_samples=24000,
        duration_s=1,
        peak=0.5,
        rms_dbfs=-10,
        clipping_fraction=0,
        source_id="fixture",
        source_revision="a" * 40,
        license="CC0-1.0",
        retrieval_date="2026-11-03",
    )


def test_empty_partition() -> None:
    result = partition_recordings([], Recipe())
    assert result.assignments == result.excluded == {}


def test_shared_script_has_one_owner_and_speakers_stay_independent() -> None:
    recipe = Recipe()
    records = [recording(i, f"speaker-{i}", "same") for i in range(200)]
    result = partition_recordings(records, recipe)
    assert set(result.assignments.values()) == {"test"}
    assert len(result.assignments) + len(result.excluded) == len(records)
    assert result == partition_recordings(list(reversed(records)), recipe)
    assert all(
        speaker_split(item.speaker, recipe) == "test"
        for item in records
        if item.record_id in result.assignments
    )


def test_duplicate_id_or_audio_fails() -> None:
    record = recording(0, "s", "a")
    with pytest.raises(ValueError, match="duplicate"):
        partition_recordings([record, record], Recipe())
    duplicate = record.model_copy(update={"record_id": "1"})
    with pytest.raises(ValueError, match="duplicate"):
        partition_recordings([record, duplicate], Recipe())


def test_changed_seed_changes_speaker_assignment() -> None:
    first, second = Recipe(), Recipe(seed=77)
    assert any(
        speaker_split(str(index), first) != speaker_split(str(index), second)
        for index in range(100)
    )
