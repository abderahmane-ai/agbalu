"""Progress lines: stages, counted progress and heartbeats, each a JSON object on stderr."""

from __future__ import annotations

import json
import threading

import pytest

from agbalu.llm.progress import Heartbeat, Tracker, progress, stage


def lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    return [json.loads(line) for line in capsys.readouterr().err.splitlines()]


def test_a_line_is_json_with_its_event_and_time(capsys: pytest.CaptureFixture[str]) -> None:
    progress("x", value=1)
    (line,) = lines(capsys)
    assert line["event"] == "x"
    assert line["value"] == 1
    assert isinstance(line["time"], float)


class TestStage:
    def test_start_and_end_bracket_the_work(self, capsys: pytest.CaptureFixture[str]) -> None:
        with stage("load", source="m"):
            pass
        start, end = lines(capsys)
        assert (start["event"], start["source"]) == ("stage_start", "m")
        assert end["event"] == "stage_end"
        assert isinstance(end["seconds"], float)

    def test_a_failure_is_named_and_raised(self, capsys: pytest.CaptureFixture[str]) -> None:
        message = "boom"
        with pytest.raises(ValueError, match=message), stage("load"):
            raise ValueError(message)
        assert lines(capsys)[-1]["event"] == "stage_failed"


class TestTracker:
    def test_reports_the_first_and_the_last_unit(self, capsys: pytest.CaptureFixture[str]) -> None:
        tracker = Tracker("job", 10, interval_s=3600)
        for _ in range(10):
            tracker.advance()
        reported = lines(capsys)
        assert [line["done"] for line in reported] == [1, 10]
        assert reported[-1]["eta_s"] == 0.0

    def test_reports_again_once_the_interval_passes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tracker = Tracker("job", 100, interval_s=0.0)
        for _ in range(3):
            tracker.advance()
        assert [line["done"] for line in lines(capsys)] == [1, 2, 3]

    def test_an_open_count_has_no_estimate_and_carries_extra_values(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        Tracker("job", None).advance(5, kept=3)
        (line,) = lines(capsys)
        assert (line["done"], line["total"], line["eta_s"], line["kept"]) == (5, None, None, 3)


def test_the_heartbeat_beats_with_the_jobs_status(capsys: pytest.CaptureFixture[str]) -> None:
    with Heartbeat("job", status=lambda: {"part": "x"}, interval_s=0.05):
        threading.Event().wait(0.3)
    beats = lines(capsys)
    assert len(beats) >= 2
    assert all(beat["event"] == "heartbeat" and beat["part"] == "x" for beat in beats)
