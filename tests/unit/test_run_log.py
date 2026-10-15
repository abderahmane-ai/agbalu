"""The readable run-log view: each event becomes one line a person can read."""

from __future__ import annotations

import json

from tools.run_log import Printer, Rendered, View, bar, summary


def event(kind: str, **values: object) -> str:
    return json.dumps({"event": kind, "time": 1792049375.85, **values})


STEP = event(
    "step",
    step=20,
    loss=2.7167,
    lr=1.9999e-05,
    grad_norm=1.89,
    taken=True,
    tokens=10485760,
    tokens_per_s=4930.3,
    mfu=0.4977,
    memory=34108093952,
    total_steps=1527,
    eta_h=44.45,
)


class TestSteps:
    def test_a_step_carries_position_loss_rate_and_finish_time(self) -> None:
        rendered = View().render(STEP)
        assert rendered is not None
        assert not rendered.live
        for piece in ("step 20/1,527", "1.3%", "loss 2.7167", "4,930 tok/s", "MFU 50%"):
            assert piece in rendered.text
        assert "34.1 GB" in rendered.text
        assert "10.5M tok" in rendered.text
        assert "ETA 44.4 h, done" in rendered.text or "ETA 44.5 h, done" in rendered.text

    def test_the_moving_average_covers_the_steps_seen(self) -> None:
        view = View()
        for loss in (3.0, 2.0):
            line = json.loads(STEP)
            line["loss"] = loss
            rendered = view.render(json.dumps(line))
        assert rendered is not None
        assert "avg2 2.5000" in rendered.text

    def test_a_skipped_step_says_so(self) -> None:
        line = json.loads(STEP)
        line["taken"] = False
        rendered = View().render(json.dumps(line))
        assert rendered is not None
        assert "SKIPPED" in rendered.text


class TestLiveLines:
    def test_progress_is_a_live_bar(self) -> None:
        line = event("progress", task="step 21 micro-batches", done=16, total=32, eta_s=50.0)
        rendered = View().render(line)
        assert rendered == Rendered(
            "  step 21 micro-batches [" + "#" * 12 + "." * 12 + "] 16/32, 50 s left", live=True
        )

    def test_a_heartbeat_extends_the_current_progress(self) -> None:
        view = View()
        view.render(event("progress", task="step 3 micro-batches", done=1, total=32, eta_s=9.0))
        rendered = view.render(
            event("heartbeat", label="train", phase="training", gpu_allocated_gib=26.6)
        )
        assert rendered is not None
        assert rendered.live
        assert rendered.text.startswith("  step 3 micro-batches")
        assert "alive" in rendered.text
        assert "GPU 26.6 GiB" in rendered.text

    def test_tqdm_bars_are_live(self) -> None:
        rendered = View().render(
            "Running loglikelihood requests:  11%|█ | 6113/56168 [02:20<16:11, 51.54it/s]"
        )
        assert rendered is not None
        assert rendered.live

    def test_bar_bounds(self) -> None:
        assert bar(0, 4) == "[" + "." * 24 + "]"
        assert bar(4, 4) == "[" + "#" * 24 + "]"


class TestEvaluation:
    def test_an_eval_compares_against_its_reference(self) -> None:
        line = event(
            "eval",
            step=50,
            bpc={"kab": 1.911, "eng": 1.026, "fra": 1.27},
            reference={"kab": 4.258, "eng": 1.32, "fra": 1.658},
            eligible=True,
        )
        rendered = View().render(line)
        assert rendered is not None
        assert "kab 1.911 (-55.1% vs step 0 4.258)" in rendered.text
        assert "drift guard ok" in rendered.text

    def test_without_a_reference_the_first_eval_becomes_one(self) -> None:
        view = View()
        view.render(event("eval", step=0, bpc={"kab": 4.0, "eng": 1.0, "fra": 2.0}, eligible=True))
        rendered = view.render(
            event("eval", step=50, bpc={"kab": 2.0, "eng": 1.0, "fra": 2.0}, eligible=False)
        )
        assert rendered is not None
        assert "-50.0%" in rendered.text
        assert "DRIFT GUARD FAILED" in rendered.text


class TestPassThrough:
    def test_library_chatter_is_dropped(self) -> None:
        view = View()
        assert (
            view.render("fatal: not a git repository (or any parent up to mount point /home)")
            is None
        )
        assert view.render("Loading weights: 100%|██████████| 426/426") is None

    def test_a_traceback_passes_through(self) -> None:
        rendered = View().render("RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED")
        assert rendered == Rendered(
            "RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED", live=False
        )

    def test_a_failed_stage_is_loud(self) -> None:
        rendered = View().render(
            event("stage_failed", stage="save checkpoint", seconds=3.0, error="OSError()")
        )
        assert rendered is not None
        assert rendered.text.startswith("!! save checkpoint FAILED")

    def test_an_unknown_event_still_shows(self) -> None:
        rendered = View().render(event("something_new", value=3))
        assert rendered is not None
        assert "something_new: value 3" in rendered.text


class TestPrinter:
    def test_a_live_line_is_replaced_on_a_terminal(self) -> None:
        written: list[str] = []
        printer = Printer(written.append, terminal=True)
        printer.show(Rendered("progress", live=True))
        printer.show(Rendered("step", live=False))
        assert written == ["\r\x1b[2Kprogress", "\r\x1b[2Kstep\n"]

    def test_live_lines_are_skipped_off_a_terminal(self) -> None:
        written: list[str] = []
        printer = Printer(written.append, terminal=False)
        printer.show(Rendered("progress", live=True))
        printer.show(Rendered("step", live=False))
        assert written == ["step\n"]


def test_the_summary_keeps_recent_events_and_where_the_run_is_now() -> None:
    progress = event("progress", task="step 21 micro-batches", done=1, total=32, eta_s=90.0)
    lines = [
        STEP,
        "fatal: not a git repository",
        event("stage_start", stage="save checkpoint"),
        progress,
    ]
    result = summary(lines, keep=1)
    assert result[0] == "-> save checkpoint"
    assert result[1].startswith("step 21 micro-batches")
