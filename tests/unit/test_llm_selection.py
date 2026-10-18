"""The merge-coefficient rule: gates against α = 0, the chrF++ choice, ties, refusals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from agbalu.llm import selection
from agbalu.llm.chat_eval import MODES, report_name

GOOD = {
    "chrf": 20.0,
    "retention": 74.0,
    "ifeval": 80.0,
    "gsm8k_no_think": 85.0,
    "gsm8k_think": 90.0,
    "mmlu_pro_no_think": 60.0,
    "mmlu_pro_think": 65.0,
    "niah": 95.0,
}


def candidate(alpha: float, **changes: float) -> selection.Candidate:
    numbers = {**GOOD, **changes}
    completion = {
        "limit": None,
        "mt": {"eng-kab": {"chrf++": numbers["chrf"]}},
        "retention": {"mean_points": numbers["retention"]},
    }
    plain = {
        "limit": None,
        "tasks": {
            "ifeval": {"score": numbers["ifeval"]},
            "gsm8k": {"score": numbers["gsm8k_no_think"]},
            "mmlu_pro": {"score": numbers["mmlu_pro_no_think"]},
            **{name: {"score": numbers["niah"]} for name in selection.NEEDLES},
        },
    }
    think = {
        "limit": None,
        "tasks": {
            "gsm8k": {"score": numbers["gsm8k_think"]},
            "mmlu_pro": {"score": numbers["mmlu_pro_think"]},
        },
    }
    return selection.Candidate(
        alpha, Path(f"a{alpha}"), completion, {"no_think": plain, "think": think}
    )


class TestRule:
    def test_the_best_chrf_among_passing_candidates_wins(self) -> None:
        verdict = selection.select_alpha(
            [candidate(0.0, chrf=10.0), candidate(0.5, chrf=25.0), candidate(1.0, chrf=30.0)]
        )
        assert verdict["selected_alpha"] == 1.0

    def test_a_tie_within_half_a_point_goes_to_the_lower_alpha(self) -> None:
        verdict = selection.select_alpha(
            [candidate(0.0, chrf=10.0), candidate(0.5, chrf=29.6), candidate(1.0, chrf=30.0)]
        )
        assert verdict["selected_alpha"] == 0.5

    def test_just_outside_the_tie_the_higher_chrf_wins(self) -> None:
        verdict = selection.select_alpha(
            [candidate(0.0, chrf=10.0), candidate(0.5, chrf=29.4), candidate(1.0, chrf=30.0)]
        )
        assert verdict["selected_alpha"] == 1.0

    @pytest.mark.parametrize(
        ("measure", "dropped"),
        [
            ("retention", 71.9),
            ("ifeval", 76.9),
            ("gsm8k_think", 86.9),
            ("mmlu_pro_no_think", 56.9),
            ("niah", 89.9),
        ],
    )
    def test_each_gate_can_reject_the_best_chrf(self, measure: str, dropped: float) -> None:
        verdict = selection.select_alpha(
            [
                candidate(0.0, chrf=10.0),
                candidate(0.5, chrf=20.0),
                candidate(1.0, chrf=30.0, **{measure: dropped}),
            ]
        )
        assert verdict["selected_alpha"] == 0.5
        rows = verdict["candidates"]
        assert isinstance(rows, dict)
        key = "niah" if measure == "niah" else measure
        assert rows["1.0"]["failed"] == [key]

    def test_a_drop_exactly_at_the_limit_passes(self) -> None:
        verdict = selection.select_alpha(
            [candidate(0.0), candidate(0.5, chrf=25.0, retention=72.0)]
        )
        assert verdict["selected_alpha"] == 0.5

    def test_no_passing_positive_alpha_falls_back_to_the_cpt_checkpoint(self) -> None:
        verdict = selection.select_alpha([candidate(0.0), candidate(0.5, retention=60.0)])
        assert verdict["selected_alpha"] is None
        assert verdict["fallback"] == "start SFT from the CPT checkpoint"

    def test_the_control_is_never_selected(self) -> None:
        verdict = selection.select_alpha([candidate(0.0, chrf=99.0), candidate(0.5, chrf=1.0)])
        assert verdict["selected_alpha"] == 0.5


class TestRefusals:
    def test_without_a_control(self) -> None:
        with pytest.raises(selection.SelectionError, match="α = 0"):
            selection.select_alpha([candidate(0.5), candidate(1.0)])

    def test_duplicate_coefficients(self) -> None:
        with pytest.raises(selection.SelectionError, match="duplicate"):
            selection.select_alpha([candidate(0.0), candidate(0.5), candidate(0.5)])

    def test_a_limited_report(self) -> None:
        limited = candidate(0.5)
        completion = {**limited.completion, "limit": 8}
        with pytest.raises(selection.SelectionError, match="limited to 8"):
            selection.select_alpha(
                [candidate(0.0), selection.Candidate(0.5, limited.run, completion, limited.chat)]
            )

    def test_a_missing_measure_is_named(self) -> None:
        broken = candidate(0.5)
        chat: dict[str, Mapping[str, object]] = {
            "no_think": broken.chat["no_think"],
            "think": {"limit": None, "tasks": {}},
        }
        with pytest.raises(selection.SelectionError, match=r"tasks\.gsm8k\.score"):
            selection.select_alpha(
                [candidate(0.0), selection.Candidate(0.5, broken.run, broken.completion, chat)]
            )


class TestReading:
    def test_a_run_directory_is_read_whole(self, tmp_path: Path) -> None:
        source = candidate(0.25)
        (tmp_path / "merge.json").write_text(json.dumps({"alpha": 0.25}), encoding="utf-8")
        (tmp_path / "evaluation.json").write_text(json.dumps(source.completion), encoding="utf-8")
        for mode in MODES:
            path = tmp_path / report_name(mode)
            path.write_text(json.dumps(source.chat[mode]), encoding="utf-8")
        read = selection.read_candidate(tmp_path)
        assert read.alpha == 0.25
        assert selection.measures(read) == selection.measures(source)

    def test_a_missing_report_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "merge.json").write_text(json.dumps({"alpha": 0.5}), encoding="utf-8")
        with pytest.raises(selection.SelectionError, match=r"chat-no_think\.json not found"):
            selection.read_candidate(tmp_path)
