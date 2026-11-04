"""Audit embedded source audio through resumable shard transactions."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from pydantic import Field

from agbalu.codec_tts.artifacts import (
    Recording,
    Rejection,
    atomic_binary,
    digest_file,
    provenance,
    read_recordings,
    run_lock,
    verify_conservation,
    write_json,
)
from agbalu.codec_tts.configuration import Recipe, Specification
from agbalu.codec_tts.pairs import pair_recordings
from agbalu.codec_tts.partition import partition_recordings
from agbalu.codec_tts.waveforms import Waveform, measure_signal, resample_waveform, text_key
from agbalu.llm.progress import Tracker, stage

DECODE_LOCK = threading.Lock()


class SourceFile(Specification):
    path: str
    sha256: str


class Acquisition(Specification):
    # Acquisition contains additional source-specific observations, read separately below.
    revision: str
    files: tuple[SourceFile, ...]
    access_date: date


class ShardResult(Specification):
    config_hash: str
    source_sha256: str
    rows: int = Field(ge=0)
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    recordings_sha256: str
    rejects_sha256: str
    provenance: dict[str, object]


def read_acquisition(recipe: Recipe) -> Acquisition:
    path = recipe.source / "acquisition.stats.json"
    payload: object = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        message = f"invalid acquisition file list: {path}"
        raise TypeError(message)
    files: list[SourceFile] = []
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            message = f"invalid acquisition entry at {path}: {item!r}"
            raise TypeError(message)
        if item["path"].endswith(".parquet"):
            files.append(
                SourceFile.model_validate({"path": item["path"], "sha256": item.get("sha256")})
            )
    acquisition = Acquisition.model_validate(
        {
            "revision": payload.get("revision"),
            "files": tuple(files),
            "access_date": payload.get("access_date"),
        }
    )
    if acquisition.revision != recipe.source_revision or not files:
        message = f"source identity mismatch or no Parquet files at {path}: {acquisition.revision}"
        raise ValueError(message)
    return acquisition


def source_files(recipe: Recipe) -> tuple[Path, ...]:
    path = recipe.source / "acquisition.stats.json"
    acquisition = read_acquisition(recipe)
    expected: dict[str, str] = {}
    for item in acquisition.files:
        name = Path(item.path).name
        if name in expected:
            message = f"duplicate acquisition shard name {name!r} in {path}"
            raise ValueError(message)
        expected[name] = item.sha256
    actual = tuple(sorted((recipe.source / "data").glob("*.parquet")))
    if set(expected) != {item.name for item in actual}:
        message = (
            f"source shard set differs from {path}: expected={sorted(expected)} actual={actual}"
        )
        raise ValueError(message)
    for shard in actual:
        observed = digest_file(shard)
        if observed != expected[shard.name]:
            message = (
                f"checksum mismatch {shard}: observed={observed}, expected={expected[shard.name]}"
            )
            raise ValueError(message)
    return actual


def raw_rows(shard: Path) -> Iterator[tuple[int, object]]:
    with pq.ParquetFile(shard) as parquet:
        offset = 0
        for batch in parquet.iter_batches(
            batch_size=128, columns=["audio", "transcription", "client_id"], use_threads=False
        ):
            for value in batch.to_pylist():
                yield offset, value
                offset += 1
        verify_conservation(parquet.metadata.num_rows, offset, 0)


def embedded_bytes(value: object, max_size_bytes: int) -> tuple[bytes, str, str]:
    if not isinstance(value, dict):
        message = f"expected audio record, got {type(value).__name__}"
        raise TypeError(message)
    audio, text, speaker = value.get("audio"), value.get("transcription"), value.get("client_id")
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        message = "missing embedded audio.bytes"
        raise TypeError(message)
    raw = audio["bytes"]
    if len(raw) > max_size_bytes or not raw:
        message = f"embedded audio size_bytes={len(raw)}, limit={max_size_bytes}"
        raise ValueError(message)
    if not isinstance(text, str) or not text_key(text):
        message = f"missing/empty transcription: {text!r}"
        raise ValueError(message)
    if not isinstance(speaker, str) or not speaker.strip():
        message = f"missing/empty client_id: {speaker!r}"
        raise ValueError(message)
    return raw, text, speaker


def decode_audio(raw: bytes) -> tuple[Waveform, int]:
    with DECODE_LOCK:
        waveform, rate_hz = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        message = f"expected mono audio, got channels={waveform.shape[1]}"
        raise ValueError(message)
    mono = np.asarray(waveform[:, 0], dtype=np.float32)
    measure_signal(mono, rate_hz)
    return mono, int(rate_hz)


def recording_for(
    value: object, shard: Path, row: int, recipe: Recipe, retrieval_date: str
) -> Recording:
    raw, text, speaker = embedded_bytes(value, recipe.max_embedded_bytes)
    waveform, rate_hz = decode_audio(raw)
    signal = measure_signal(waveform, rate_hz)
    duration_s = signal.samples / rate_hz
    if not recipe.min_duration_s <= duration_s <= recipe.max_duration_s:
        message = (
            f"duration_s={duration_s} outside [{recipe.min_duration_s},{recipe.max_duration_s}]"
        )
        raise ValueError(message)
    converted = resample_waveform(waveform, rate_hz, recipe.rate_hz)
    measure_signal(converted, recipe.rate_hz)
    identity = f"{recipe.source_revision}:{shard.name}:{row}"
    return Recording(
        record_id=hashlib.sha256(identity.encode()).hexdigest(),
        shard=shard.name,
        row=row,
        audio_sha256=hashlib.sha256(raw).hexdigest(),
        speaker=speaker,
        text=text,
        text_key=text_key(text),
        stored_rate_hz=rate_hz,
        samples=signal.samples,
        prepared_samples=converted.size,
        duration_s=duration_s,
        peak=signal.peak,
        rms_dbfs=signal.rms_dbfs,
        clipping_fraction=signal.clipping_fraction,
        source_id=recipe.source_id,
        source_revision=recipe.source_revision,
        license=recipe.license,
        retrieval_date=retrieval_date,
    )


def write_line(stream: BinaryIO, value: Specification) -> None:
    stream.write((value.model_dump_json() + "\n").encode())


def prepare_shard(shard: Path, recipe: Recipe, date: str) -> ShardResult:
    root = recipe.output / "shards" / shard.stem
    stats_path = root / "audit.stats.json"
    accepted_path, rejects_path = root / "recordings.jsonl", root / "rejects.jsonl"
    source_digest = digest_file(shard)
    if stats_path.is_file():
        result = ShardResult.model_validate_json(stats_path.read_bytes())
        if (
            result.config_hash != recipe.digest
            or result.source_sha256 != source_digest
            or result.recordings_sha256 != digest_file(accepted_path)
            or result.rejects_sha256 != digest_file(rejects_path)
        ):
            message = f"prepared artifact identity/checksum mismatch: {stats_path}"
            raise ValueError(message)
        verify_conservation(result.rows, result.accepted, result.rejected)
        return result
    rows = accepted = rejected = 0
    with atomic_binary(accepted_path) as kept, atomic_binary(rejects_path) as rejects:
        for row, value in raw_rows(shard):
            rows += 1
            try:
                record = recording_for(value, shard, row, recipe, date)
            except (ValueError, TypeError, sf.LibsndfileError) as error:
                rejected += 1
                rejection = Rejection(
                    record_id=f"{shard.name}:{row}", shard=shard.name, row=row, reason=str(error)
                )
                write_line(rejects, rejection)
            else:
                accepted += 1
                write_line(kept, record)
    verify_conservation(rows, accepted, rejected)
    result = ShardResult(
        config_hash=recipe.digest,
        source_sha256=source_digest,
        rows=rows,
        accepted=accepted,
        rejected=rejected,
        recordings_sha256=digest_file(accepted_path),
        rejects_sha256=digest_file(rejects_path),
        provenance=provenance(recipe.digest),
    )
    write_json(stats_path, result.model_dump())
    return result


def prepare_audio(recipe: Recipe) -> dict[str, object]:
    with run_lock(recipe.output), stage("decode source audio"):
        shards = source_files(recipe)
        tracker = Tracker("decoded shards", len(shards), interval_s=5)
        counts: list[ShardResult] = []
        retrieval_date = read_acquisition(recipe).access_date.isoformat()
        with atomic_binary(recipe.output / "recordings.jsonl") as combined:
            for shard in shards:
                counts.append(prepare_shard(shard, recipe, retrieval_date))
                path = recipe.output / "shards" / shard.stem / "recordings.jsonl"
                with path.open("rb") as stream:
                    while block := stream.read(1 << 20):
                        combined.write(block)
                tracker.advance()
        report: dict[str, object] = {
            **provenance(recipe.digest),
            "input_rows": sum(item.rows for item in counts),
            "accepted_rows": sum(item.accepted for item in counts),
            "rejected_rows": sum(item.rejected for item in counts),
            "language_reviewed": False,
            "recordings_sha256": digest_file(recipe.output / "recordings.jsonl"),
            "shards": [item.model_dump() for item in counts],
        }
        write_json(recipe.output / "preparation.stats.json", report)
        return report


def prepare_pairs(recipe: Recipe) -> dict[str, object]:
    with run_lock(recipe.output), stage("partition reference pairs"):
        records = tuple(read_recordings(recipe.output / "recordings.jsonl"))
        partition = partition_recordings(records, recipe)
        pairing = pair_recordings(records, partition, recipe.seed)
        for split in ("train", "validation", "test"):
            if not any(pair.split == split for pair in pairing.pairs):
                message = f"no eligible pairs in {split}: inspect speaker/script coverage"
                raise ValueError(message)
        with atomic_binary(recipe.output / "pairs.jsonl") as stream:
            for pair in pairing.pairs:
                write_line(stream, pair)
        excluded = {**partition.excluded, **pairing.excluded}
        write_json(recipe.output / "pair-rejects.json", excluded)
        verify_conservation(len(records), len(pairing.pairs), len(excluded))
        report: dict[str, object] = {
            **provenance(recipe.digest),
            "input_rows": len(records),
            "paired_rows": len(pairing.pairs),
            "excluded_rows": len(excluded),
            "split_pairs": {
                split: sum(pair.split == split for pair in pairing.pairs)
                for split in ("train", "validation", "test")
            },
            "recordings_sha256": digest_file(recipe.output / "recordings.jsonl"),
            "pairs_sha256": digest_file(recipe.output / "pairs.jsonl"),
        }
        write_json(recipe.output / "pairs.stats.json", report)
        return report
