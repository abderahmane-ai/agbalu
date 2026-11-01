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


def test_encoder_evaluation_is_compact_and_preserves_the_primary_score() -> None:
    rendered = View().render(
        event(
            "eval",
            family="encoder",
            step=4,
            mlm={"ce": 3.2, "accuracy": 0.4},
            downstream={"surface/canonical": {"nonpunct_accuracy": 0.8125, "by_label": ["VERB"]}},
        )
    )
    assert rendered is not None
    assert "surface POS 81.25%" in rendered.text
    assert "MLM CE 3.2" in rendered.text
    assert "by_label" not in rendered.text


def test_encoder_step_with_unmeasured_utilization_displays_a_dash() -> None:
    line = json.loads(STEP)
    line.update(family="encoder", mfu=None)
    rendered = View().render(json.dumps(line))
    assert rendered is not None
    assert "MFU -" in rendered.text
    assert "ETA " in rendered.text
    assert ", done " in rendered.text


def test_encoder_diagnostics_keep_arrays_in_the_raw_log() -> None:
    rendered = View().render(
        event("diagnostics", family="encoder", step=100, activation_rms=[0.2, 1.5, 0.7])
    )
    assert rendered is not None
    assert rendered.text == "   encoder health at step 100: layer activation RMS 0.200–1.500"
    assert View().render(event("result", family="encoder", by_label=["VERB"])) is None


def test_encoder_previews_show_complete_sentences_and_keep_other_families_short() -> None:
    sentence = "A" * 250 + " gar-aneɣ."
    encoder = View().render(
        event("probe", family="encoder", step=100, texts={"sentence 4 original": sentence})
    )
    other = View().render(event("probe", step=100, texts={"sample": sentence}))
    assert encoder is not None
    assert other is not None
    assert sentence in encoder.text
    assert sentence not in other.text
    assert "..." in other.text


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

    def test_an_sft_eval_and_best_show_the_validation_loss_not_the_guard(self) -> None:
        view = View()
        bpc = {"kab": 1.575, "eng": 1.042, "fra": 1.261}
        rendered = view.render(event("eval", step=0, bpc=bpc, validation_loss=1.23456))
        best = view.render(event("best", step=0, validation_loss=1.23456))
        assert rendered is not None
        assert best is not None
        assert "validation loss 1.2346" in rendered.text
        assert "GUARD" not in rendered.text
        assert best.text == "   new best checkpoint: validation loss 1.2346 at step 0"


class TestBaseline:
    def test_a_seeded_step_0_baseline_is_used_when_the_tail_misses_it(self) -> None:
        view = View()
        view.seed(event("eval", step=0, bpc={"kab": 4.258, "eng": 1.32, "fra": 1.658}))
        rendered = view.render(
            event("eval", step=50, bpc={"kab": 1.911, "eng": 1.026, "fra": 1.27}, eligible=True)
        )
        assert rendered is not None
        assert "kab 1.911 (-55.1% vs step 0 4.258)" in rendered.text

    def test_an_unseeded_view_never_calls_its_first_eval_step_0(self) -> None:
        rendered = View().render(
            event("eval", step=50, bpc={"kab": 1.911, "eng": 1.026, "fra": 1.27}, eligible=True)
        )
        assert rendered is not None
        assert "vs step" not in rendered.text
        assert "kab 1.911 |" in rendered.text

    def test_a_bad_seed_is_ignored(self) -> None:
        view = View()
        view.seed("")
        view.seed("{not json")
        assert view.reference is None


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

def test_dihya_evaluation_renders_accuracy_and_floor() -> None:
    rendered = View().render(
        event(
            "eval",
            family="dihya",
            step=250,
            accuracy=0.7801,
            worst_group_recall=0.6364,
            worst_group="shy_Latn/everyday",
            worst_class="shi_Tfng",
            worst_class_recall=0.6475,
        )
    )
    assert (
        rendered.line
        == "== dihya eval at step 250: accuracy 0.7801 | worst recall 0.6364 [shy_Latn/everyday] | worst class 0.6475 [shi_Tfng]"
    )


def test_dihya_best_checkpoint_preserves_group_and_score() -> None:
    rendered = View().render(
        event(
            "best",
            family="dihya",
            step=2072,
            score=0.7760,
            worst_group="shy_Latn/everyday",
            worst_class="shi_Tfng",
        )
    )
    assert (
        rendered.line
        == "   new dihya checkpoint: score 0.776 [shy_Latn/everyday] class [shi_Tfng] at step 2072"
    )


def test_dihya_result_preserves_rate_on_pass() -> None:
    rendered = View().render(event("result", family="dihya", passed=True, rate=576.5))
    assert rendered.line == "== dihya smoke passed (576.5 seq/s)"


def test_dihya_run_end_renders_speed_and_best_score() -> None:
    rendered = View().render(
        event(
            "end",
            family="dihya",
            ended="finished",
            step=4752,
            tokens_per_second=532.0,
            best_score=0.7760,
        )
    )
    assert rendered.line == "== dihya ended (finished) at step 4752: 532 tok/s, best score 0.776"

