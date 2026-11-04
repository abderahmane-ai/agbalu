"""Decode real encoded audio through complete shard transactions."""

import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from agbalu.codec_tts.artifacts import read_recordings, write_json
from agbalu.codec_tts.audio import decode_audio, prepare_audio, source_files
from agbalu.codec_tts.configuration import Recipe

pytestmark = pytest.mark.integration


def test_complete_decode_reject_resume_and_corruption(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    (source / "data").mkdir(parents=True)
    buffer = io.BytesIO()
    sf.write(buffer, np.ones(4800, dtype=np.float32) * 0.2, 48000, format="WAV")
    raw = buffer.getvalue()
    shard = source / "data" / "one.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "audio": {"bytes": raw, "path": "a.wav"},
                    "transcription": "Azul",
                    "client_id": "a",
                },
                {
                    "audio": {"bytes": b"bad", "path": "b.mp3"},
                    "transcription": "Axxam",
                    "client_id": "b",
                },
            ]
        ),
        shard,
    )
    write_json(
        source / "acquisition.stats.json",
        {
            "revision": "a" * 40,
            "access_date": "2026-11-03",
            "files": [
                {
                    "path": "data/one.parquet",
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    recipe = Recipe(
        source=source, output=tmp_path / "prepared", source_revision="a" * 40, min_duration_s=0.01
    )
    first = prepare_audio(recipe)
    assert (first["input_rows"], first["accepted_rows"], first["rejected_rows"]) == (2, 1, 1)
    record = next(read_recordings(recipe.output / "recordings.jsonl"))
    assert (record.samples, record.prepared_samples, record.stored_rate_hz) == (4800, 2400, 48000)
    assert record.text == "Azul"
    assert prepare_audio(recipe)["recordings_sha256"] == first["recordings_sha256"]
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        source_files(recipe)


def test_stereo_is_not_silently_mixed() -> None:
    buffer = io.BytesIO()
    sf.write(buffer, np.ones((10, 2), dtype=np.float32), 24000, format="WAV")
    with pytest.raises(ValueError, match="channels=2"):
        decode_audio(buffer.getvalue())
