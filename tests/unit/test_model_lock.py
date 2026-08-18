"""The run lock, including the cases that would wedge a run out of its own checkpoints.

A lock that never expires turns one crash into a permanent outage; a lock that expires too
eagerly lets two trainers spend GPU hours overwriting each other. Both directions are
tested, and so is the truncated-write case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.model.config import TrainConfig
from agbalu.model.lock import (
    LOCK,
    STALE_AFTER,
    LockInfo,
    RunLockedError,
    acquire,
    default_owner,
    read,
    refresh,
    release,
)

SECONDS_PER_STEP = 1_048_576 / 69_212
"""Tokens per step over the throughput measured on an A10 (2026-08-08)."""

NOW = 1_000_000.0


class TestAcquire:
    def test_an_unlocked_directory_is_taken(self, tmp_path: Path) -> None:
        info = acquire(tmp_path, "a", now=NOW)
        assert info.owner == "a"
        assert (tmp_path / LOCK).is_file()

    def test_a_missing_directory_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "runs" / "encoder"
        acquire(target, "a", now=NOW)
        assert (target / LOCK).is_file()

    def test_a_live_lock_held_by_another_owner_raises(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        with pytest.raises(RunLockedError, match="held by a"):
            acquire(tmp_path, "b", now=NOW + 60)

    def test_the_error_names_the_owner_and_the_age(self, tmp_path: Path) -> None:
        acquire(tmp_path, "container-7", now=NOW)
        with pytest.raises(RunLockedError, match=r"container-7.*5\.0 min"):
            acquire(tmp_path, "b", now=NOW + 300)

    def test_the_same_owner_may_reacquire(self, tmp_path: Path) -> None:
        """A Modal retry lands in a container that may reuse the task id; it must not be
        locked out of the run it already owns."""
        acquire(tmp_path, "a", now=NOW)
        assert acquire(tmp_path, "a", now=NOW + 60).owner == "a"

    def test_a_stale_lock_is_taken_over(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        info = acquire(tmp_path, "b", now=NOW + STALE_AFTER)
        assert info.owner == "b"

    def test_a_lock_one_second_short_of_stale_still_blocks(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        with pytest.raises(RunLockedError):
            acquire(tmp_path, "b", now=NOW + STALE_AFTER - 1)

    def test_force_steals_a_live_lock(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        assert acquire(tmp_path, "b", force=True, now=NOW + 1).owner == "b"

    def test_stale_after_is_longer_than_the_checkpoint_interval(self) -> None:
        """The heartbeat is written at each checkpoint, so expiry must outlast that gap or
        a healthy run declares itself dead and a second launch steals its directory."""
        assert TrainConfig().save_every * SECONDS_PER_STEP < STALE_AFTER


class TestRefresh:
    def test_refresh_moves_the_heartbeat_but_keeps_the_start(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        refresh(tmp_path, "a", now=NOW + 120)
        held = read(tmp_path)
        assert held is not None
        assert held.started == NOW
        assert held.heartbeat == NOW + 120

    def test_refresh_keeps_a_long_run_out_of_staleness(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        for step in range(1, 20):
            refresh(tmp_path, "a", now=NOW + step * 600)
        with pytest.raises(RunLockedError):
            acquire(tmp_path, "b", now=NOW + 19 * 600 + 60)

    def test_refresh_recreates_a_deleted_lock(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        (tmp_path / LOCK).unlink()
        refresh(tmp_path, "a", now=NOW + 5)
        assert read(tmp_path) is not None


class TestRelease:
    def test_the_owner_can_release(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        release(tmp_path, "a")
        assert read(tmp_path) is None

    def test_releasing_twice_is_not_an_error(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a", now=NOW)
        release(tmp_path, "a")
        release(tmp_path, "a")

    def test_another_owner_cannot_release(self, tmp_path: Path) -> None:
        """Otherwise a stopped run tears down the lock of the one that took over."""
        acquire(tmp_path, "a", now=NOW)
        release(tmp_path, "b")
        held = read(tmp_path)
        assert held is not None
        assert held.owner == "a"

    def test_releasing_an_unlocked_directory_is_not_an_error(self, tmp_path: Path) -> None:
        release(tmp_path, "a")


class TestRead:
    def test_absent_is_none(self, tmp_path: Path) -> None:
        assert read(tmp_path) is None

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "not json",
            "{}",
            '{"owner": "a"}',
            '{"owner":"a","started":"x","heartbeat":1}',
            "\x00",
        ],
        ids=["empty", "garbage", "no fields", "partial", "wrong type", "null byte"],
    )
    def test_an_unreadable_lock_reads_as_unlocked(self, tmp_path: Path, content: str) -> None:
        """A lock truncated by a mid-write kill must not lock the run out permanently."""
        (tmp_path / LOCK).write_text(content, encoding="utf-8")
        assert read(tmp_path) is None
        assert acquire(tmp_path, "b", now=NOW).owner == "b"

    def test_a_directory_where_the_lock_should_be_reads_as_unlocked(self, tmp_path: Path) -> None:
        (tmp_path / LOCK).mkdir()
        assert read(tmp_path) is None


class TestStaleness:
    def test_age_is_measured_from_the_heartbeat(self) -> None:
        info = LockInfo(owner="a", started=NOW, heartbeat=NOW + 100)
        assert info.age(NOW + 400) == 300

    def test_a_fresh_lock_is_not_stale(self) -> None:
        assert not LockInfo("a", NOW, NOW).is_stale(NOW + 1)


class TestOwnerIdentity:
    def test_the_modal_task_id_is_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODAL_TASK_ID", "ta-123")
        assert default_owner() == "ta-123"

    def test_it_falls_back_to_the_process_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        assert default_owner().startswith("pid-")

    def test_an_empty_task_id_does_not_produce_an_empty_owner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODAL_TASK_ID", "")
        assert default_owner().startswith("pid-")
