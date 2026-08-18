from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agbalu.acquire.storage import (
    PART_SUFFIX,
    StorageError,
    atomic_write,
    finalize,
    free_bytes,
    require_space,
    resolve_within,
    sha256_file,
)

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def test_hashes_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    target.write_bytes(b"")
    assert sha256_file(target) == (EMPTY_SHA256, 0)


def test_hash_matches_hashlib(tmp_path: Path) -> None:
    payload = "aɣbalu ɛ ɣ ḥ ḍ".encode()
    target = tmp_path / "kab.txt"
    target.write_bytes(payload)
    assert sha256_file(target) == (hashlib.sha256(payload).hexdigest(), len(payload))


def test_hash_is_chunk_size_independent(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 500
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert sha256_file(target, chunk_bytes=7) == sha256_file(target, chunk_bytes=1 << 20)


def test_hashing_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "nope.bin")


def test_atomic_write_publishes_on_success(tmp_path: Path) -> None:
    final = tmp_path / "nested" / "out.txt"
    with atomic_write(final) as part:
        part.write_text("done", encoding="utf-8")
    assert final.read_text(encoding="utf-8") == "done"
    assert not part.exists()


def test_atomic_write_leaves_nothing_behind_on_failure(tmp_path: Path) -> None:
    final = tmp_path / "out.txt"

    def write_then_fail() -> None:
        with atomic_write(final) as part:
            part.write_text("half", encoding="utf-8")
            msg = "interrupted"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="interrupted"):
        write_then_fail()
    # A truncated file that a later exists() check would trust is the failure mode.
    assert not final.exists()
    assert not final.with_name(final.name + PART_SUFFIX).exists()


def test_atomic_write_cleans_up_on_keyboard_interrupt(tmp_path: Path) -> None:
    final = tmp_path / "out.txt"

    def write_then_interrupt() -> None:
        with atomic_write(final) as part:
            part.write_text("half", encoding="utf-8")
            raise KeyboardInterrupt

    # BaseException, not Exception: a Ctrl-C must not leave a half file either.
    with pytest.raises(KeyboardInterrupt):
        write_then_interrupt()
    assert not final.exists()
    assert not final.with_name(final.name + PART_SUFFIX).exists()


def test_atomic_write_rejects_a_block_that_wrote_nothing(tmp_path: Path) -> None:
    final = tmp_path / "out.txt"
    with pytest.raises(StorageError, match="nothing was written"), atomic_write(final):
        pass


def test_atomic_write_overwrites_an_existing_file(tmp_path: Path) -> None:
    final = tmp_path / "out.txt"
    final.write_text("old", encoding="utf-8")
    with atomic_write(final) as part:
        part.write_text("new", encoding="utf-8")
    assert final.read_text(encoding="utf-8") == "new"


def test_finalize_moves_a_part_file(tmp_path: Path) -> None:
    final = tmp_path / "out.bin"
    part = final.with_name(final.name + PART_SUFFIX)
    part.write_bytes(b"payload")
    finalize(part, final)
    assert final.read_bytes() == b"payload"
    assert not part.exists()


def test_finalize_refuses_a_missing_part(tmp_path: Path) -> None:
    final = tmp_path / "out.bin"
    with pytest.raises(StorageError, match="nothing was written"):
        finalize(final.with_name(final.name + PART_SUFFIX), final)


def test_resolve_within_accepts_a_nested_relative_path(tmp_path: Path) -> None:
    assert resolve_within(tmp_path, "a/b/c.txt") == (tmp_path / "a/b/c.txt").resolve()


@pytest.mark.parametrize("candidate", ["../escape.txt", "a/../../escape.txt", "../../etc/passwd"])
def test_resolve_within_blocks_traversal(tmp_path: Path, candidate: str) -> None:
    with pytest.raises(StorageError, match="escapes its source root"):
        resolve_within(tmp_path, candidate)


def test_resolve_within_blocks_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="must be relative"):
        resolve_within(tmp_path, "/etc/passwd")


def test_free_bytes_walks_up_to_an_existing_parent(tmp_path: Path) -> None:
    assert free_bytes(tmp_path / "does" / "not" / "exist") > 0


def test_require_space_passes_when_there_is_room(tmp_path: Path) -> None:
    require_space(tmp_path, 1)


def test_require_space_raises_before_a_write_that_cannot_fit(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="insufficient space"):
        require_space(tmp_path, free_bytes(tmp_path) * 2)
