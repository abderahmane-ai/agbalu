"""The HPC launcher's commands, built without touching the pod."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from tools import launch

from agbalu.llm import cli
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
        assert str(cli.SFT_CONVERSATIONS) in launch.DATA_FILES

    def test_sft_training_passes_its_objective(self) -> None:
        argv = ["train", "--run", "s", "--model", "m", "--objective", "sft"]
        arguments = launch.train_arguments(launch.build_parser().parse_args(argv))
        assert arguments == ["train", "--run", "s", "--model", "m", "--objective", "sft"]

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

    def test_merge_passes_the_run_alpha_and_cpt_checkpoint(self) -> None:
        argv = ["merge", "--run", "m", "--alpha", "0.5", "--model", "artifacts/runs/c/best"]
        args = launch.build_parser().parse_args(argv)
        assert launch.merge_arguments(args) == [
            "merge",
            "--run",
            "m",
            "--alpha",
            "0.5",
            "--cpt",
            "artifacts/runs/c/best",
        ]

    @pytest.mark.parametrize(
        ("argv", "missing"),
        [(["--alpha", "0.5"], "--model"), (["--model", "x"], "--alpha")],
    )
    def test_merge_refuses_without_alpha_or_checkpoint(
        self, capsys: pytest.CaptureFixture[str], argv: list[str], missing: str
    ) -> None:
        assert launch.main(["merge", "--run", "m", *argv]) == 1
        assert f"needs {missing}" in capsys.readouterr().err

    def test_the_exit_suite_runs_the_completion_evaluation_alone(self) -> None:
        args = launch.build_parser().parse_args(["evaluate", "--run", "e", "--model", "m"])
        run, arguments, log, then = launch.job(args)
        assert (run, log, then) == ("e", "e/log.txt", [])
        assert arguments[:3] == ["evaluate", "--out", "artifacts/runs/e/evaluation.json"]

    def test_the_select_suite_chains_one_vllm_process_per_mode(self) -> None:
        argv = ["evaluate", "--run", "e", "--model", "m", "--suite", "select"]
        _, _, _, then = launch.job(launch.build_parser().parse_args(argv))
        assert len(then) == 2
        for mode, command in zip(("no_think", "think"), then, strict=True):
            assert f"{launch.VLLM_PYTHON} -m agbalu.llm.chat_eval" in command
            assert f"--mode {mode}" in command
            assert f"artifacts/runs/e/chat-{mode}.json" in command

    def test_the_screen_runs_translation_alone_into_its_own_file(self) -> None:
        argv = ["evaluate", "--run", "e", "--model", "m", "--suite", "screen"]
        run, arguments, log, then = launch.job(launch.build_parser().parse_args(argv))
        assert (run, log, then) == ("e", "e/log.txt", [])
        assert arguments == [
            "evaluate",
            "--out",
            "artifacts/runs/e/screen.json",
            "--parts",
            "mt",
            "--model",
            "m",
        ]

    def test_chat_evaluation_runs_without_the_flashinfer_sampler(self) -> None:
        argv = ["evaluate", "--run", "e", "--model", "m", "--suite", "select"]
        _, _, _, then = launch.job(launch.build_parser().parse_args(argv))
        assert all(c.startswith("VLLM_USE_FLASHINFER_SAMPLER=0 ") for c in then)

    def test_chat_only_skips_the_completion_suite(self) -> None:
        argv = ["evaluate", "--run", "e", "--model", "m", "--suite", "select", "--chat-only"]
        _, arguments, _, then = launch.job(launch.build_parser().parse_args(argv))
        assert arguments == []
        assert len(then) == 2
        script = launch.detached("e", arguments, "e/log.txt", then)[-1]
        assert "agbalu.llm.cli" not in script
        assert script.count("agbalu.llm.chat_eval") == 2

    def test_chat_only_is_refused_without_a_chat_suite(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = ["evaluate", "--run", "e", "--model", "m", "--suite", "screen", "--chat-only"]
        assert launch.main(argv) == 1
        assert "--chat-only needs --suite select or release" in capsys.readouterr().err

    def test_the_screen_needs_a_model_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert launch.main(["evaluate", "--run", "e", "--suite", "screen"]) == 1
        assert "needs --model" in capsys.readouterr().err

    def test_the_chat_suite_needs_a_model_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert launch.main(["evaluate", "--run", "e", "--suite", "release"]) == 1
        assert "needs --model" in capsys.readouterr().err

    def test_chained_commands_share_the_log_and_stop_at_the_first_failure(self) -> None:
        script = launch.detached("r", ["evaluate"], "r/log.txt", ["second", "third"])[-1]
        assert "&& second && third; }" in script
        assert script.count("artifacts/runs/r/log.txt 2>&1") == 1

    def test_pull_leaves_every_kind_of_weights_behind(self) -> None:
        excluded = launch.WEIGHT_EXCLUDES[1::2]
        assert set(excluded) == {"step-*", "best", "model"}

    def test_status_output_is_rendered_between_its_markers(self) -> None:
        step = (
            '{"event": "step", "time": 1.0, "step": 3, "loss": 2.5, "lr": 1e-05,'
            ' "grad_norm": 1.0, "tokens": 1572864, "tokens_per_s": 4900.0, "mfu": 0.49,'
            ' "memory": 34000000000, "total_steps": 1527, "eta_h": 45.0}'
        )
        output = (
            f"session: running\n{launch.BASELINE_MARKER}\n\n{launch.LOG_MARKER}\n{step}\n"
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
