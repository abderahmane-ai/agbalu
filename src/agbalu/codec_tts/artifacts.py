"""Publish durable codec speech artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

from pydantic import Field

from agbalu.codec_tts.configuration import Specification
from agbalu.llm.provenance import code_stamp

Split = Literal["train", "validation", "test"]


class Recording(Specification):
    record_id: str
    shard: str
    row: int = Field(ge=0)
    audio_sha256: str
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    text_key: str = Field(min_length=1)
    stored_rate_hz: int = Field(gt=0)
    samples: int = Field(gt=0)
    prepared_samples: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    peak: float = Field(ge=0)
    rms_dbfs: float | None
    clipping_fraction: float = Field(ge=0, le=1)
    source_id: str
    source_revision: str
    license: str
    retrieval_date: str
    language_decision: Literal["review"] = "review"
    language_reason: str = "upstream Kabyle label; project acoustic/transcript review pending"


class Rejection(Specification):
    record_id: str
    shard: str
    row: int = Field(ge=0)
    reason: str


class Pair(Specification):
    target: str
    reference: str
    speaker: str
    split: Split


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_binary(path: Path) -> Iterator[BinaryIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        staged.replace(path)
        sync_directory(path.parent)
    finally:
        staged.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with atomic_binary(path) as stream:
        stream.write(encoded.encode())


@contextmanager
def run_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"another writer holds {root / '.lock'}"
            raise RuntimeError(message) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_recordings(path: Path) -> Iterator[Recording]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                yield Recording.model_validate_json(line)
            except ValueError as error:
                message = f"invalid recording at {path}:{number}: {error}"
                raise ValueError(message) from error


def provenance(config_hash: str) -> dict[str, object]:
    return {"version": 1, "version_tag": None, "config_hash": config_hash, **code_stamp()}


def verify_conservation(input_rows: int, output_rows: int, rejected_rows: int) -> None:
    if min(input_rows, output_rows, rejected_rows) < 0 or input_rows != output_rows + rejected_rows:
        message = f"unaccounted records: input={input_rows}, output={output_rows}"
        message += f", rejected={rejected_rows}"
        raise ValueError(message)


class EventLog:
    def __init__(self, stream: TextIO, root: Path, identity: str) -> None:
        self.stream = stream
        self.root = root
        self.identity = identity
        self.lock = threading.Lock()
        self.status: dict[str, object] = {}

    def emit(self, event: str, **values: object) -> None:
        payload = {
            "schema_version": 1,
            "family": "codec_tts",
            "config_hash": self.identity,
            "event": event,
            "time": time.time(),
            **values,
        }
        line = json.dumps(payload, allow_nan=False) + "\n"
        with self.lock:
            self.stream.write(line)
            self.stream.flush()
            sys.stderr.write(line)
            sys.stderr.flush()
            self.status.update(payload)
            write_json(self.root / "status.json", self.status)


@contextmanager
def event_log(root: Path, identity: str) -> Iterator[EventLog]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "log.jsonl").open("a", encoding="utf-8") as stream:
        yield EventLog(stream, root, identity)
