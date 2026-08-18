"""The smoke's run-directory reset.

A smoke exists to measure throughput on real hardware. The first two invocations resumed
the previous one's finished checkpoint, trained zero steps, and reported the checkpoint's
counters as a measurement — so the reset is what makes the measurement repeatable, and it
deletes files on a network volume, which is worth pinning down.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from modal_app.train import _reset


class TestReset:
    def test_a_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        _reset(tmp_path / "never-ran")
        assert not (tmp_path / "never-ran").exists()

    def test_the_run_directory_is_emptied(self, tmp_path: Path) -> None:
        run = tmp_path / "smoke"
        run.mkdir()
        (run / "latest.pt").write_bytes(b"weights")
        (run / "latest.pt.sha256").write_text("digest\n", encoding="utf-8")
        (run / "metrics.jsonl").write_text('{"event": "finish"}\n', encoding="utf-8")

        _reset(run)
        assert not run.exists()

    def test_nested_state_goes_with_it(self, tmp_path: Path) -> None:
        run = tmp_path / "smoke"
        (run / "nested").mkdir(parents=True)
        (run / "nested" / "best.pt").write_bytes(b"weights")

        _reset(run)
        assert not run.exists()

    def test_sibling_runs_are_untouched(self, tmp_path: Path) -> None:
        """It is handed `/checkpoints/<run_name>`, never the volume root: the real run's
        checkpoints live one directory over and are the only cover against preemption."""
        keep = tmp_path / "agbalu-encoder-v1"
        keep.mkdir()
        (keep / "latest.pt").write_bytes(b"weights")
        run = tmp_path / "smoke"
        run.mkdir()
        (run / "latest.pt").write_bytes(b"weights")

        _reset(run)
        assert (keep / "latest.pt").is_file()

    def test_what_was_removed_is_named_in_the_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A destructive step that leaves no record reads exactly like a run that
        resumed, and the two have opposite consequences for a throughput number."""
        run = tmp_path / "smoke"
        run.mkdir()
        (run / "latest.pt").write_bytes(b"weights")
        (run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

        with caplog.at_level(logging.INFO, logger="agbalu.model"):
            _reset(run)
        assert "latest.pt" in caplog.text
        assert "metrics.jsonl" in caplog.text
