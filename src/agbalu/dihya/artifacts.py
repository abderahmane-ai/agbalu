"""Write Dihya artifacts atomically with durable directory entries."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    """Prevent two processes from writing the same training run."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def atomic_file(path: Path) -> Iterator[BinaryIO]:
    """Replace a file only after its complete contents reach storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: object) -> None:
    """Commit strict JSON without nonfinite values."""
    with atomic_file(path) as stream:
        stream.write(
            (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        )
