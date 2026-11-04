"""Exercise durable artifact writes and exclusive locks."""

from pathlib import Path

import pytest

from agbalu.codec_tts.artifacts import atomic_binary, run_lock, write_json

pytestmark = pytest.mark.integration


def test_interrupted_write_preserves_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"old": 1})

    def interrupt() -> None:
        with atomic_binary(path) as stream:
            stream.write(b"partial")
            raise InterruptedError

    with pytest.raises(InterruptedError):
        interrupt()
    assert path.read_text() == '{"old": 1}\n'
    assert sorted(item.name for item in tmp_path.iterdir()) == ["result.json"]


def test_lock_refuses_second_writer(tmp_path: Path) -> None:
    with (
        run_lock(tmp_path),
        pytest.raises(RuntimeError, match="another writer"),
        run_lock(tmp_path),
    ):
        pytest.fail("second writer acquired the lock")
