"""The HPC launcher's commands, built without touching the pod."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

import pytest
from tools import launch

from agbalu.llm import cli
from agbalu.llm.decontaminate import FLORES_CORRECTED


class TestCommands:
    def test_teacher_staging_preserves_checkpoint_run_and_separate_logs(self) -> None:
        args = launch.build_parser().parse_args(
            [
                "stage",
                "--family",
                "codec_tts",
                "--run",
                "teacher",
                "--output",
                "artifacts/runs/teacher-stage/model",
                "--role",
                "best_ce",
            ]
        )
        run, command, log, _ = launch.job(args)
        assert run == "teacher-stage"
        assert command[:4] == ["__codec_tts__", "stage", "--run", "teacher"]
        assert "--role" in command
        assert log == "teacher-stage/log.txt"
        assert launch.module(command).startswith("timeout")

    def test_teacher_synthesis_syncs_only_inputs_and_rejects_unsafe_paths(self) -> None:
        args = launch.build_parser().parse_args(
            [
                "synthesize",
                "--family",
                "codec_tts",
                "--run",
                "teacher",
                "--text-file",
                "target.txt",
                "--reference",
                "voice.wav",
                "--reference-text-file",
                "voice.txt",
                "--output",
                "artifacts/speech.wav",
            ]
        )
        assert launch.speech_input_paths(args) == ["target.txt", "voice.wav", "voice.txt"]
        assert launch.job(args)[0] == "teacher-synthesis"
        args.output = "../outside.wav"
        with pytest.raises(launch.LaunchError, match="relative to the project"):
            launch.job(args)

    def test_codec_family_routes_without_qwen_defaults(self) -> None:
        args = launch.build_parser().parse_args(
            ["train", "--family", "codec_tts", "--run", "codec-r32"]
        )
        run, command, log, chained = launch.job(args)
        assert run == "codec-r32"
        assert command == ["__codec_tts__", "train", "--run", "codec-r32"]
        assert "agbalu.codec_tts.cli" in launch.module(command)
        assert "agbalu-codec/bin/python" in launch.module(command)
        assert log == "codec-r32/log.txt"
        assert chained == []

    def test_codec_smoke_uses_shared_part_and_bounded_execution(self) -> None:
        args = launch.build_parser().parse_args(
            ["smoke", "--family", "codec_tts", "--run", "r", "--part", "resume"]
        )
        command = launch.job(args)[1]
        assert command[-2:] == ["--part", "resume"]
        assert "840s" in launch.module(command)

    def test_codec_rejects_llm_model_and_task_options(self) -> None:
        parser = launch.build_parser()
        with pytest.raises(launch.LaunchError, match="config owns model"):
            launch.job(
                parser.parse_args(
                    ["train", "--family", "codec_tts", "--run", "r", "--model", "qwen"]
                )
            )
        with pytest.raises(launch.LaunchError, match="unsupported codec_tts data"):
            launch.job(
                parser.parse_args(["data", "--family", "codec_tts", "--run", "r", "--task", "pack"])
            )

    def test_report_pull_excludes_tensor_checkpoints(self) -> None:
        excluded = set(launch.WEIGHT_EXCLUDES[1::2])
        assert {"*.pt", "*.pth", "*.safetensors"} <= excluded
        assert not any("stats.json" in pattern for pattern in excluded)

    def test_dihya_sync_preserves_remote_processed_splits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands: list[list[str]] = []
        monkeypatch.setattr(launch, "execute", lambda command: commands.append(list(command)))
        # File discovery is controlled so this unit test never touches actual datasets.
        monkeypatch.setattr(Path, "is_file", lambda _path: False)
        monkeypatch.setattr(Path, "is_dir", lambda _path: True)
        monkeypatch.setattr(launch, "stamp", lambda: Path("scratch-stamp.json"))
        monkeypatch.setattr(Path, "unlink", lambda _path: None)
        launch.sync_dihya()
        assert len(commands) == 2
        assert not any(
            "data/processed/dihya" in argument for command in commands for argument in command
        )

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

    def test_expand_pdfs_resumes_only_when_asked(self) -> None:
        parser = launch.build_parser()
        fresh = launch.job(parser.parse_args(["data", "--task", "expand-pdfs"]))
        assert fresh == (
            "data-expand-pdfs",
            ["expand-pdfs"],
            "data-expand-pdfs/log.txt",
            [],
        )
        resumed = launch.job(parser.parse_args(["data", "--task", "expand-pdfs", "--resume"]))
        assert resumed[1] == ["expand-pdfs", "--resume"]

    def test_resume_is_refused_for_a_task_that_cannot_use_it(self) -> None:
        argv = ["data", "--task", "documents", "--resume"]
        with pytest.raises(launch.LaunchError):
            launch.job(launch.build_parser().parse_args(argv))

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

    def test_every_job_exports_the_quiet_settings_as_a_shell_reads_them(self) -> None:
        words = shlex.split(launch.PREAMBLE)
        assert "TQDM_MININTERVAL=60" in words
        assert not any(word.startswith("TQDM_DISABLE") for word in words)
        assert "VLLM_LOGGING_LEVEL=WARNING" in words
        assert (
            "PYTHONWARNINGS=ignore::DeprecationWarning,ignore::FutureWarning,"
            "ignore:NLTK will not authorize:UserWarning"
        ) in words

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

    def test_assemble_names_the_tower_and_the_template(self) -> None:
        argv = ["assemble", "--run", "f", "--model", "t", "--template", "m"]
        run, arguments, log, then = launch.job(launch.build_parser().parse_args(argv))
        assert (run, log, then) == ("f", "f/log.txt", [])
        assert arguments == ["assemble", "--run", "f", "--tower", "t", "--template", "m"]

    def test_assemble_needs_a_template(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert launch.main(["assemble", "--run", "f", "--model", "t"]) == 1
        assert "assemble needs --template" in capsys.readouterr().err

    def test_the_guard_scores_two_tasks_into_the_selection_report(self) -> None:
        argv = ["evaluate", "--run", "e", "--model", "m", "--suite", "guard"]
        _, arguments, _, then = launch.job(launch.build_parser().parse_args(argv))
        assert arguments == []
        (command,) = then
        assert command.endswith(
            "--model m --suite select --mode no_think "
            "--out artifacts/runs/e/chat-no_think.json --tasks ifeval gsm8k"
        )

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
        assert set(excluded) == {"step-*/", "best", "model", "*.pt", "*.pth", "*.safetensors"}

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


def test_encoder_training_dispatch_preserves_its_token_budget() -> None:
    args = launch.build_parser().parse_args(
        ["train", "--family", "encoder", "--run", "encoder-train", "--tokens", "1000000"]
    )
    run, arguments, log, then = launch.job(args)
    assert run == "encoder-train"
    assert arguments == ["__encoder__", "train", "--run", "encoder-train", "--tokens", "1000000"]
    assert log == "encoder-train/log.txt"
    assert then == []


def test_only_encoder_smokes_have_a_bounded_runtime() -> None:
    assert launch.module(["__encoder__", "smoke"]).startswith(
        "timeout --signal=TERM --kill-after=30s 840s "
    )
    assert not launch.module(["__encoder__", "train"]).startswith("timeout")
    assert not launch.module(["smoke"]).startswith("timeout")


def test_restoration_training_uses_the_native_family() -> None:
    args = launch.build_parser().parse_args(["train", "--family", "restoration", "--run", "belaid"])
    assert launch.job(args) == (
        "belaid",
        ["__restoration__", "train", "--run", "belaid"],
        "belaid/log.txt",
        [],
    )
    assert "agbalu.restoration.cli" in launch.module(["__restoration__", "train"])
    assert launch.module(["__restoration__", "smoke"]).startswith("timeout")
    assert not launch.module(["__restoration__", "train"]).startswith("timeout")


def test_restoration_data_preparation_has_a_separate_run() -> None:
    args = launch.build_parser().parse_args(["data", "--family", "restoration"])
    assert launch.job(args) == (
        "restoration-data",
        ["__restoration__", "prepare", "--run", "restoration-data"],
        "restoration-data/log.txt",
        [],
    )


@pytest.mark.integration
def test_pull_keeps_evaluation_reports_while_excluding_checkpoint_directories(
    tmp_path: Path,
) -> None:
    if shutil.which("rsync") is None:
        pytest.skip("rsync is unavailable")
    source, destination = tmp_path / "source", tmp_path / "destination"
    checkpoint = source / "step-0000100-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training.pt").write_bytes(b"weights")
    reports = source / "evaluations"
    reports.mkdir()
    (reports / "step-0000100.stats.json").write_text('{"score": 0.5}\n')
    launch.execute(["rsync", "-a", *launch.WEIGHT_EXCLUDES, f"{source}/", str(destination)])
    assert not (destination / checkpoint.name).exists()
    assert (
        destination / "evaluations" / "step-0000100.stats.json"
    ).read_text() == '{"score": 0.5}\n'
