"""Suite-wide setup.

The interpreter segfaults in C-extension finalisation: one process holds torch,
sentencepiece, kenlm, numba, pyarrow and sklearn at once and their teardown order is not
ours. The crash lands after the terminal summary is written, so a run ending at [100%] with
no `N passed` line is this rather than a failure.

The process therefore leaves through `os._exit`, which runs no finaliser and no `atexit`
handler, from `pytest_unconfigure` — after the summary — carrying the session's real status.

Nothing may rely on an `atexit` handler after this. pytest registers both its temp-directory
pruning and its lock removal there, so `tmp_path_retention_policy` is set to `failed` in
`pyproject.toml`: that path deletes during the session instead.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.main import Session

_EXIT_STATUS: dict[str, int] = {}


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    _EXIT_STATUS["status"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    status = _EXIT_STATUS.get("status")
    if status is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
