"""The chat-mode suite's pure parts: task lists, seeded draws, decoding, score reading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.llm import chat_eval
from agbalu.llm.chat_eval import HarnessTask, Mode


def output(name: str, metric: str, value: object, effective: int = 7) -> dict[str, object]:
    return {"results": {name: {metric: value}}, "n-samples": {name: {"effective": effective}}}


class FakeHarness:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[str, int | None]] = []

    def score(self, task: HarnessTask, limit: int | None) -> object:
        self.calls.append((task.name, limit))
        return output(task.name, task.metric, self.value)


class TestTasks:
    def test_the_selection_suite_holds_no_code_task(self) -> None:
        assert not any(task.code for task in chat_eval.tasks_for("select"))

    def test_the_release_suite_contains_the_selection_suite(self) -> None:
        select = {task.name for task in chat_eval.tasks_for("select")}
        assert select < {task.name for task in chat_eval.tasks_for("release")}

    @pytest.mark.parametrize("mode", chat_eval.MODES)
    def test_a_mode_runs_only_its_own_tasks(self, mode: Mode) -> None:
        assert all(mode in task.modes for task in chat_eval.tasks_in("release", mode))

    def test_instruction_following_and_needles_run_without_thinking_only(self) -> None:
        names = {task.name for task in chat_eval.tasks_in("select", "think")}
        assert "ifeval" not in names
        assert not any(name.startswith("niah") for name in names)

    def test_the_engine_length_covers_the_needle_and_its_answer(self) -> None:
        assert chat_eval.MAX_MODEL_LENGTH >= chat_eval.LONG_CONTEXT + chat_eval.NO_THINK_TOKENS


class TestDraws:
    def test_a_draw_is_seeded_sorted_and_unique(self) -> None:
        first = chat_eval.drawn(1000, 50)
        assert first == chat_eval.drawn(1000, 50)
        assert first is not None
        assert first == sorted(set(first))
        assert len(first) == 50

    def test_a_draw_is_not_a_prefix(self) -> None:
        indices = chat_eval.drawn(1000, 50)
        assert indices != list(range(50))

    @pytest.mark.parametrize(("size", "count"), [(10, None), (10, 10), (10, 20)])
    def test_everything_is_scored_when_the_draw_would_cover_it(
        self, size: int, count: int | None
    ) -> None:
        assert chat_eval.drawn(size, count) is None


class TestDecoding:
    def test_without_thinking_decoding_is_greedy(self) -> None:
        assert chat_eval.generation("no_think")["do_sample"] is False

    def test_thinking_samples_with_qwens_precise_settings(self) -> None:
        settings = chat_eval.generation("think")
        assert (settings["temperature"], settings["top_p"], settings["top_k"]) == (0.6, 0.95, 20)
        assert settings["max_gen_toks"] == chat_eval.THINK_TOKENS


class TestScores:
    TASK = HarnessTask("gsm8k", "exact_match,flexible-extract", 5, chat_eval.MODES, ("select",))

    def test_the_metric_and_count_are_read(self) -> None:
        assert chat_eval.read_score(output("gsm8k", self.TASK.metric, 0.5), self.TASK) == (0.5, 7)

    @pytest.mark.parametrize("value", [None, "0.5", float("nan"), True])
    def test_a_missing_or_unusable_metric_is_named(self, value: object) -> None:
        with pytest.raises(chat_eval.ChatEvaluationError, match="flexible-extract"):
            chat_eval.read_score(output("gsm8k", self.TASK.metric, value), self.TASK)

    def test_a_missing_row_is_named(self) -> None:
        with pytest.raises(chat_eval.ChatEvaluationError, match="no result row for 'gsm8k'"):
            chat_eval.read_score({"results": {}}, self.TASK)

    def test_scores_are_reported_in_points(self) -> None:
        harness = FakeHarness(0.8125)
        rows = chat_eval.evaluate_tasks(harness, [self.TASK], limit=8)
        assert rows["gsm8k"]["score"] == 81.25
        assert harness.calls == [("gsm8k", 8)]


class TestReport:
    def test_the_report_is_written_whole(self, tmp_path: Path) -> None:
        path = tmp_path / "runs" / chat_eval.report_name("think")
        chat_eval.write_report(path, {"tasks": {"gsm8k": {}}})
        assert json.loads(path.read_text(encoding="utf-8")) == {"tasks": {"gsm8k": {}}}
        assert not path.with_name(f".{path.name}.partial").exists()

    def test_the_parser_requires_mode_suite_and_output(self) -> None:
        with pytest.raises(SystemExit):
            chat_eval.build_parser().parse_args(["--model", "m"])
