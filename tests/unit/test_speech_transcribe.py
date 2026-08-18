"""Clip discovery and audio loading for the local decode path.

The model itself is not built here — that needs the released weights and a torch install,
and `tests/integration/test_hub_release_directories.py` is what proves those load. What is
covered is everything that decides *which bytes reach the model* and in what shape, because
a resample or a channel fold that is silently wrong produces a plausible hypothesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from agbalu.speech.transcribe import (
    SAMPLE_RATE,
    TranscriptionError,
    audio_files,
    load_audio,
    read_vocabulary,
)


def write_wav(path: Path, *, rate: int, seconds: float = 0.25, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(rate * seconds)
    tone = np.sin(2 * np.pi * 440 * np.arange(samples) / rate).astype(np.float32)
    data = tone if channels == 1 else np.stack([tone] * channels, axis=1)
    soundfile.write(path, data, rate)
    return path


def test_a_single_file_resolves_to_itself(tmp_path: Path) -> None:
    clip = write_wav(tmp_path / "one.wav", rate=SAMPLE_RATE)
    assert audio_files(clip) == [clip]


def test_a_directory_yields_every_decodable_clip_sorted(tmp_path: Path) -> None:
    write_wav(tmp_path / "b.wav", rate=SAMPLE_RATE)
    write_wav(tmp_path / "a.wav", rate=SAMPLE_RATE)
    write_wav(tmp_path / "nested" / "c.WAV", rate=SAMPLE_RATE)
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")

    found = audio_files(tmp_path)
    assert [path.name for path in found] == ["a.wav", "b.wav", "c.WAV"]


def test_a_missing_path_is_reported_by_name(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="absent"):
        audio_files(tmp_path / "absent")


def test_a_directory_with_no_audio_names_the_suffixes(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")
    with pytest.raises(TranscriptionError, match=r"\.wav"):
        audio_files(tmp_path)


def test_audio_is_resampled_to_the_rate_the_model_was_trained_at(tmp_path: Path) -> None:
    clip = write_wav(tmp_path / "half.wav", rate=8_000, seconds=1.0)
    audio = load_audio(clip)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - SAMPLE_RATE) <= 1


def test_audio_already_at_the_rate_is_not_resampled(tmp_path: Path) -> None:
    clip = write_wav(tmp_path / "exact.wav", rate=SAMPLE_RATE, seconds=1.0)
    assert len(load_audio(clip)) == SAMPLE_RATE


def test_stereo_is_folded_to_one_channel(tmp_path: Path) -> None:
    clip = write_wav(tmp_path / "stereo.wav", rate=SAMPLE_RATE, seconds=0.5, channels=2)
    audio = load_audio(clip)

    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE // 2


def test_a_file_that_is_not_audio_is_reported_by_name(tmp_path: Path) -> None:
    impostor = tmp_path / "text.wav"
    impostor.write_text("this is not a RIFF header", encoding="utf-8")

    with pytest.raises(TranscriptionError, match=r"text\.wav"):
        load_audio(impostor)


def test_an_empty_clip_loads_as_an_empty_array(tmp_path: Path) -> None:
    clip = write_wav(tmp_path / "silent.wav", rate=SAMPLE_RATE, seconds=0.0)
    assert len(load_audio(clip)) == 0


def test_a_missing_vocabulary_names_the_command_that_writes_it(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="make release REPO=fadhma"):
        read_vocabulary(tmp_path)


def test_the_vocabulary_is_read_as_written(tmp_path: Path) -> None:
    mapping = {"[PAD]": 0, "|": 1, "a": 2, "ɣ": 3}
    (tmp_path / "vocab.json").write_text(json.dumps(mapping), encoding="utf-8")
    assert read_vocabulary(tmp_path) == mapping
