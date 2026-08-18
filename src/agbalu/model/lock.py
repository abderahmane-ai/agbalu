"""Single-writer lock over a run directory.

Checkpoint writes are atomic, so two concurrent trainers cannot corrupt `latest.pt` — they
resume from the same step and overwrite each other, which costs GPU hours and produces a
`metrics.jsonl` interleaving two runs. The lock makes the second launch refuse instead.

A crashed run leaves its lock behind, so the lock expires: an owner that has not refreshed
within `STALE_AFTER` is taken over. `STALE_AFTER` must exceed the checkpoint interval,
since that is what refreshes it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

log: Final = logging.getLogger("agbalu.model")

LOCK: Final = "run.lock"

STALE_AFTER: Final = 45 * 60.0
"""Seconds without a refresh before an owner is presumed dead. Above the 25 minutes that
`save_every=100` takes at the measured step time, with margin for a slow volume write."""


class RunLockedError(Exception):
    """Another live trainer owns this run directory."""


@dataclass(frozen=True, slots=True)
class LockInfo:
    owner: str
    started: float
    heartbeat: float

    def age(self, now: float) -> float:
        return now - self.heartbeat

    def is_stale(self, now: float, stale_after: float = STALE_AFTER) -> bool:
        return self.age(now) >= stale_after


def default_owner() -> str:
    """Modal's task id where present, so a log line identifies the container."""
    return os.environ.get("MODAL_TASK_ID") or f"pid-{os.getpid()}"


def _write(path: Path, info: LockInfo) -> None:
    staging = path.with_suffix(".lock.partial")
    staging.write_text(json.dumps(asdict(info)) + "\n", encoding="utf-8")
    staging.replace(path)


def read(directory: Path) -> LockInfo | None:
    """The current lock, or None if absent or unreadable.

    An unparseable lock is treated as absent: a truncated write must not wedge a run out
    of its own checkpoints forever.
    """
    path = directory / LOCK
    if not path.is_file():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return LockInfo(
            owner=str(payload["owner"]),
            started=float(payload["started"]),
            heartbeat=float(payload["heartbeat"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        log.warning("%s is unreadable; treating the run as unlocked", path)
        return None


def acquire(
    directory: Path,
    owner: str | None = None,
    *,
    force: bool = False,
    now: float | None = None,
    stale_after: float = STALE_AFTER,
) -> LockInfo:
    """Take the lock, or raise `RunLockedError` if a live owner holds it."""
    directory.mkdir(parents=True, exist_ok=True)
    owner = owner or default_owner()
    moment = time.time() if now is None else now
    held = read(directory)

    if held is not None and held.owner != owner and not force:
        if not held.is_stale(moment, stale_after):
            msg = (
                f"{directory} is held by {held.owner}, last seen "
                f"{held.age(moment) / 60:.1f} min ago. Another trainer is running: stop it "
                f"with `make modal-cancel`, or take the directory over with "
                f"`make modal-train FORCE=1`."
            )
            raise RunLockedError(msg)
        log.warning(
            "taking over %s from %s, stale by %.1f min",
            directory,
            held.owner,
            held.age(moment) / 60,
        )

    info = LockInfo(owner=owner, started=moment, heartbeat=moment)
    _write(directory / LOCK, info)
    return info


def refresh(directory: Path, owner: str, *, now: float | None = None) -> None:
    """Renew the heartbeat. Recreates the lock if it was removed underneath."""
    moment = time.time() if now is None else now
    held = read(directory)
    started = held.started if held is not None and held.owner == owner else moment
    _write(directory / LOCK, LockInfo(owner=owner, started=started, heartbeat=moment))


def release(directory: Path, owner: str) -> None:
    """Drop the lock if this owner holds it. Another owner's lock is left alone."""
    held = read(directory)
    if held is not None and held.owner != owner:
        log.warning("not releasing %s: held by %s, not %s", directory, held.owner, owner)
        return
    (directory / LOCK).unlink(missing_ok=True)
