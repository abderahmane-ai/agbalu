"""The smoke verdicts: each passes on agreement and fails where it must."""

from __future__ import annotations

import pytest
from tools.launch import build_parser as launch_parser

from agbalu.llm import smoke
from agbalu.llm.cli import build_parser


def curve(start: float, steps: int, offset: float = 0.0) -> dict[int, float]:
    return {s: start - 0.01 * s + 0.03 * (s % 3) + offset for s in range(1, steps + 1)}


class TestParity:
    def test_matching_arms_pass(self) -> None:
        verdict = smoke.parity_verdict(curve(4.0, 200, 0.001), curve(4.0, 200))
        assert verdict["passed"] is True
        stdev = verdict["reference_step_stdev"]
        assert isinstance(stdev, float)
        assert stdev > 0

    def test_a_two_percent_gap_fails(self) -> None:
        reference = curve(4.0, 200)
        worse = {s: v * 1.02 for s, v in reference.items()}
        assert smoke.parity_verdict(worse, reference)["passed"] is False

    def test_arms_with_different_steps_are_refused(self) -> None:
        with pytest.raises(smoke.SmokeError):
            smoke.parity_verdict(curve(4.0, 199), curve(4.0, 200))


class TestResume:
    def test_matching_losses_pass_and_the_control_differs(self) -> None:
        continuous = curve(4.0, 30)
        resumed = {s: v + 1e-4 for s, v in continuous.items()}
        verdict = smoke.resume_verdict(continuous, resumed)
        assert verdict["passed"] is True
        shifted, aligned = verdict["shifted_mean"], verdict["aligned_mean"]
        assert isinstance(shifted, float)
        assert isinstance(aligned, float)
        assert shifted > smoke.CONTROL_RATIO * aligned

    def test_a_resume_one_step_off_fails(self) -> None:
        continuous = curve(4.0, 31)
        shifted = {s: continuous[s + 1] for s in range(1, 31)}
        assert (
            smoke.resume_verdict({s: continuous[s] for s in range(1, 31)}, shifted)["passed"]
            is False
        )

    def test_missing_steps_are_refused(self) -> None:
        continuous = curve(4.0, 30)
        with pytest.raises(smoke.SmokeError):
            smoke.resume_verdict(continuous, {s: v for s, v in continuous.items() if s < 20})


def test_the_parts_are_the_ones_the_cli_and_launcher_offer() -> None:
    for parser in (build_parser(), launch_parser()):
        assert (
            parser.parse_known_args(["smoke", "--part", "parity", "--run", "r"])[0].part == "parity"
        )
    assert smoke.SMOKE_PARTS == ("throughput", "parity", "resume", "evaluate")
