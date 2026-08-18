"""Filesystem primitives for acquisition: hashing, atomic landing, space checks.

Every byte that enters `data/raw/` does so through `atomic_write`, so an
interrupted fetch leaves a `.part` file and never a truncated artifact that a
later `exists()` check would mistake for a complete one.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Final

CHUNK_BYTES: Final = 1 << 20

PART_SUFFIX: Final = ".part"


class StorageError(Exception):
    """A write could not be completed safely."""


def sha256_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> tuple[str, int]:
    """Return `(hex_digest, byte_count)`, streaming so size is bounded by `chunk_bytes`."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def free_bytes(path: Path) -> int:
    """Bytes available on the filesystem holding `path`, or its nearest existing parent."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free


def require_space(path: Path, needed_bytes: int, *, headroom: float = 1.05) -> None:
    """Raise before starting a write that the filesystem cannot hold.

    Failing up front turns a disk-full ENOSPC halfway through a 19 GB download
    into an error message.
    """
    available = free_bytes(path)
    required = int(needed_bytes * headroom)
    if available < required:
        msg = (
            f"insufficient space at {path}: need ~{required:,} bytes "
            f"(incl. {headroom:.0%} headroom), have {available:,}"
        )
        raise StorageError(msg)


def resolve_within(root: Path, relative: str) -> Path:
    """Join `relative` under `root`, refusing anything that escapes it.

    Archive members and provider-supplied filenames are untrusted input; a member
    named `../../.ssh/authorized_keys` must not resolve outside the source root.
    """
    if Path(relative).is_absolute():
        msg = f"artifact path must be relative: {relative!r}"
        raise StorageError(msg)
    candidate = (root / relative).resolve()
    anchor = root.resolve()
    if candidate != anchor and anchor not in candidate.parents:
        msg = f"artifact path escapes its source root: {relative!r}"
        raise StorageError(msg)
    return candidate


def _require_written(part: Path) -> None:
    if not part.exists():
        msg = f"nothing was written to {part}"
        raise StorageError(msg)


def finalize(part: Path, final: Path) -> None:
    """Durably move a completed `.part` onto its final name.

    Split out of `atomic_write` because a resumable download must *keep* its
    partial file across a failure, which is the opposite of what that context
    manager does on the error path.
    """
    _require_written(part)
    with part.open("rb") as handle:
        os.fsync(handle.fileno())
    part.replace(final)
    dir_fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


@contextlib.contextmanager
def atomic_write(final: Path) -> Iterator[Path]:
    """Yield a temporary path; rename it onto `final` only if the block succeeds.

    The data file and its parent directory are both fsynced, so the rename is
    durable across a power loss rather than merely visible to this process.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + PART_SUFFIX)
    try:
        yield part
        finalize(part, final)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
