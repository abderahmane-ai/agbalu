"""The Phase 11 command line: every documented task parses, and `criterion` exits by verdict."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from tools.launch import DATA_TASKS

from agbalu.llm import cli


def test_every_task_the_makefile_documents_is_a_subcommand() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    line = next(line for line in makefile.splitlines() if line.startswith("llm: ##"))
    tasks = re.search(r"TASK=([a-z|]+)", line)
    assert tasks is not None
    parser = cli.build_parser()
    for task in tasks.group(1).split("|"):
        extra = ["--candidate", "a", "--base", "b"] if task == "criterion" else []
        assert parser.parse_args([task, *extra]).command == task


def test_the_launcher_data_tasks_are_subcommands() -> None:
    for task in DATA_TASKS:
        assert cli.build_parser().parse_args([task]).command == task


def test_the_launcher_merge_arguments_parse() -> None:
    argv = ["merge", "--run", "m", "--alpha", "0.25", "--cpt", "c"]
    args = cli.build_parser().parse_args(argv)
    assert (args.alpha, args.cpt, args.verify) == (0.25, Path("c"), True)
    assert args.base is None
    assert args.instruct is None


def test_aligned_training_never_reads_the_mined_pairs() -> None:
    assert cli.PARALLEL_KAB_FRA not in cli.ALIGNED_SOURCES
    assert cli.PARALLEL_KAB_FRA in cli.HOLDOUT_SOURCES


def write_half_then_fail(target: Path) -> None:
    with cli.atomic(target) as handle:
        handle.write("half")
        raise RuntimeError


def test_atomic_files_appear_only_when_complete(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    with pytest.raises(RuntimeError):
        write_half_then_fail(target)
    assert not target.exists()
    with cli.atomic(target) as handle:
        handle.write("whole")
    assert target.read_text() == "whole"


@pytest.mark.parametrize(("chrf", "code"), [(30.0, 0), (10.0, 2)])
def test_criterion_exits_by_the_verdict(tmp_path: Path, chrf: float, code: int) -> None:
    def write(name: str, bpc: float, value: float) -> Path:
        path = tmp_path / name
        payload = {
            "limit": None,
            "heldout": {"kab": {"bits_per_character": bpc}},
            "mt": {"eng-kab": {"chrf++": value}},
            "retention": {"tasks": {"a": {"score": 0.5}}, "mean_points": 50.0},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    base = write("base.json", 1.5, 20.0)
    candidate = write("cand.json", 1.2, chrf)
    assert cli.main(["criterion", "--candidate", str(candidate), "--base", str(base)]) == code


def test_gpu_commands_refuse_without_cuda(capsys: pytest.CaptureFixture[str]) -> None:
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present")
    assert cli.main(["evaluate", "--out", "x.json", "--parts", "sib"]) == 1
    assert "no CUDA device" in capsys.readouterr().err
