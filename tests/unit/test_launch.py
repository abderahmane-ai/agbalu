"""The HPC launcher's commands, built without touching the pod."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from tools import launch

from agbalu.llm.decontaminate import FLORES_CORRECTED


class TestCommands:
    def test_remote_commands_start_from_the_environment_preamble(self) -> None:
        command = launch.remote("true")
        assert command[:4] == ["ssh", "-o", "ConnectTimeout=15", "hpc"]
        assert command[-1].startswith(". ~/hpc-env.sh")
        assert "PYTHONPATH=src" in command[-1]
        assert command[-1].endswith("&& true")

    def test_the_code_sync_excludes_data_artifacts_and_git(self) -> None:
        command = launch.code_sync()
        excluded = {command[i + 1] for i, part in enumerate(command) if part == "--exclude"}
        assert {".git", "artifacts", "data"} <= excluded
        assert "--delete" in command
        assert command[-1] == f"hpc:{launch.REMOTE_ROOT}/"

    def test_data_files_keep_their_relative_paths(self) -> None:
        assert launch.data_sync(["data/a.jsonl"])[:2] == ["rsync", "-azR"]

    def test_every_data_file_the_pod_reads_is_listed(self) -> None:
        assert str(FLORES_CORRECTED) in launch.DATA_FILES
        for language in ("kab", "eng", "fra"):
            assert f"data/processed/llm/heldout-{language}.jsonl" in launch.DATA_FILES

    def test_a_detached_job_logs_inside_its_run(self) -> None:
        command = launch.detached(
            "smoke-1/parity", ["smoke", "--part", "parity"], "smoke-1/parity/log.txt"
        )
        script = command[-1]
        assert "tmux new-session -d -s agbalu-smoke-1-parity" in script
        assert "artifacts/runs/smoke-1/parity/log.txt" in script

    def test_arguments_are_quoted(self) -> None:
        assert launch.module(["train", "--run", "a b"]).endswith(shlex.quote("a b"))

    def test_session_names_hold_no_slash_or_dot(self) -> None:
        assert launch.session_name("x/y.z") == "agbalu-x-y-z"


class TestParser:
    def test_the_target_defaults_to_hpc(self) -> None:
        assert launch.build_parser().parse_args(["status", "--run", "r"]).target == "hpc"

    def test_an_unknown_target_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            launch.build_parser().parse_args(["status", "--target", "modal"])

    def test_a_run_is_required_before_anything_runs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert launch.main(["train"]) == 1
        assert "needs --run" in capsys.readouterr().err

    def test_train_passes_only_what_was_given(self) -> None:
        args = launch.build_parser().parse_args(["train", "--run", "r", "--steps", "9"])
        assert launch.train_arguments(args) == ["train", "--run", "r", "--total-steps", "9"]
        args = launch.build_parser().parse_args(["train", "--run", "r", "--stop", "2"])
        assert launch.train_arguments(args) == ["train", "--run", "r", "--stop-after", "2"]

    def test_status_output_is_rendered_between_its_markers(self) -> None:
        step = (
            '{"event": "step", "time": 1.0, "step": 3, "loss": 2.5, "lr": 1e-05,'
            ' "grad_norm": 1.0, "tokens": 1572864, "tokens_per_s": 4900.0, "mfu": 0.49,'
            ' "memory": 34000000000, "total_steps": 1527, "eta_h": 45.0}'
        )
        output = (
            f"session: running\n{launch.LOG_MARKER}\n{step}\n"
            f"{launch.CHECKPOINT_MARKER}\n/x/runs/r/step-0000002\n/x/runs/r/step-0000003\n"
        )
        lines = launch.readable_status(output)
        assert lines[0] == "session: running"
        assert lines[1].startswith("step 3/1,527")
        assert lines[-1] == "checkpoints: step-0000002, step-0000003"

    def test_status_script_marks_its_sections(self) -> None:
        script = launch.status_script("r", 600)
        assert launch.LOG_MARKER in script
        assert launch.CHECKPOINT_MARKER in script
        assert "tail -n 600" in script

    def test_stop_sends_ctrl_c_to_the_runs_session(self) -> None:
        assert "tmux send-keys -t agbalu-a-b C-c" in launch.stop_script("a/b")

    def test_a_restart_archives_the_old_log_instead_of_appending(self) -> None:
        script = launch.detached("r", ["train"], "r/log.txt")[-1]
        assert "mv " in script
        assert "> /home/jovyan/agbalu/artifacts/runs/r/log.txt 2>&1" in script
        assert ">>" not in script


def test_the_stamp_records_the_checkout() -> None:
    path = launch.stamp(Path())
    try:
        assert '"git_sha"' in path.read_text(encoding="utf-8")
    finally:
        path.unlink()
