"""Progress lines for every long Phase 11 job, so a slow job is never mistaken for a stuck one.

Every line is one JSON object on stderr, which the launcher writes to the run's `log.txt`
and `make status` shows. Three shapes: `stage_start`/`stage_end` around a named piece of
work, `progress` through a count of units with elapsed time and an estimate of what is left,
and `heartbeat` from a background thread every minute, with GPU memory and the job's own
status, for the stretches where no unit completes.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Final, Self

PROGRESS_INTERVAL_S: Final = 30.0
HEARTBEAT_S: Final = 60.0


def progress(event: str, **values: object) -> None:
    """One JSON line on stderr."""
    line = json.dumps({"event": event, "time": round(time.time(), 1), **values}, default=str)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


@contextmanager
def stage(name: str, **values: object) -> Iterator[None]:
    """`stage_start` and `stage_end` with the seconds between; `stage_failed` if it raises."""
    started = time.perf_counter()
    progress("stage_start", stage=name, **values)
    try:
        yield
    except BaseException as error:
        elapsed = round(time.perf_counter() - started, 1)
        progress("stage_failed", stage=name, seconds=elapsed, error=repr(error)[:300])
        raise
    progress("stage_end", stage=name, seconds=round(time.perf_counter() - started, 1))


class Tracker:
    """Progress through `total` units, or an open count when `total` is None: a line on the
    first unit, at most every `interval_s` seconds after it, and on the last."""

    def __init__(
        self, task: str, total: int | None, interval_s: float = PROGRESS_INTERVAL_S
    ) -> None:
        self.task = task
        self.total = total
        self.interval_s = interval_s
        self.done = 0
        self.started = time.perf_counter()
        self.last: float | None = None

    def advance(self, units: int = 1, **values: object) -> None:
        self.done += units
        now = time.perf_counter()
        finished = self.total is not None and self.done >= self.total
        if self.last is not None and now - self.last < self.interval_s and not finished:
            return
        self.last = now
        elapsed = now - self.started
        remaining = (
            elapsed / self.done * (self.total - self.done)
            if self.total is not None and self.done
            else None
        )
        progress(
            "progress",
            task=self.task,
            done=self.done,
            total=self.total,
            elapsed_s=round(elapsed, 1),
            eta_s=round(remaining, 1) if remaining is not None else None,
            **values,
        )


class Heartbeat:
    """A line every `interval_s` seconds from a background thread while a block runs."""

    def __init__(
        self,
        label: str,
        status: Callable[[], dict[str, object]] | None = None,
        interval_s: float = HEARTBEAT_S,
    ) -> None:
        self.label = label
        self.status = status
        self.interval_s = interval_s
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.beat, daemon=True)
        self.started = time.perf_counter()

    def beat(self) -> None:
        while not self.stopped.wait(self.interval_s):
            extra = self.status() if self.status is not None else {}
            progress(
                "heartbeat",
                label=self.label,
                elapsed_s=round(time.perf_counter() - self.started, 1),
                gpu_allocated_gib=gpu_allocated_gib(),
                **extra,
            )

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.stopped.set()
        self.thread.join()


def gpu_allocated_gib() -> float | None:
    """GPU memory this process holds, or None where torch or CUDA is absent."""
    if "torch" not in sys.modules:
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.memory_allocated() / 2**30, 2)
