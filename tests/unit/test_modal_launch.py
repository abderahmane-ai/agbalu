"""The launcher's log-following, which must never be able to stop the run.

The run is spawned on a *deployed* app, so no client shares its lifetime; the log stream is
a reader. These tests pin the argv — a shell string or an unresolved executable would be
the way that property gets lost — and that an interrupt is reported as harmless.
"""

from __future__ import annotations

import argparse
import inspect
import re
import subprocess
from pathlib import Path

import modal
import pytest
from modal.call_graph import InputInfo, InputStatus
from modal_app.asr import DEFAULT_RUN as ASR_RUN_NAME
from modal_app.asr import asr_finetune, asr_pipeline
from modal_app.common import APP_NAME
from modal_app.jugurtha import DEFAULT_RUN as JUGURTHA_RUN_NAME
from modal_app.jugurtha import jugurtha_train
from modal_app.launch import (
    FUNCTIONS,
    KWARGS,
    cancel,
    follow,
    log_command,
    main,
    spawn_kwargs,
    status,
    stop_command,
)
from modal_app.mt import finetune as mt_finetune
from modal_app.train import pretrain
from modal_app.translate import mt_predict

from agbalu.mt.finetune import DEFAULT_RUN_NAME as MT_RUN_NAME


class TestLogCommand:
    def test_it_names_the_deployed_app_and_follows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        assert log_command() == ["/usr/local/bin/modal", "app", "logs", APP_NAME, "--follow"]

    def test_the_executable_is_resolved_not_a_bare_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare `modal` would resolve through PATH at exec time."""
        monkeypatch.setattr("shutil.which", lambda _: "/opt/homebrew/bin/modal")
        assert Path(log_command()[0]).is_absolute()

    def test_it_is_a_list_so_no_shell_is_involved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        command = log_command()
        assert isinstance(command, list)
        assert all(isinstance(part, str) for part in command)

    def test_a_missing_cli_exits_with_a_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(SystemExit, match="not on PATH"):
            log_command()

    def test_a_call_id_scopes_the_stream_to_that_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every function shares one app, so an unscoped stream interleaves whatever else is
        running. Matoub's per-step lines arriving inside a distillation run's made both
        unreadable, which is what this flag exists to stop."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        command = log_command(call_id="fc-01M053BNZJHG6FC2KCDXXST96P")
        assert command[-2:] == ["--function-call", "fc-01M053BNZJHG6FC2KCDXXST96P"]
        assert APP_NAME in command

    def test_the_scope_is_keyed_on_the_call_not_the_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--status` resolves through the function and breaks once a subset deploy removes
        it. This resolves against the app name, so it survives that."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        assert "--function" not in log_command(call_id="fc-abc")


class TestStart:
    """`pretrain` grew `compile_model`, `schedule_start` and `resume_from`, and a keyword
    the launcher cannot pass is a feature that does not exist."""

    @staticmethod
    def _spawner(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seen: dict[str, object]
    ) -> list[str]:
        """Returns the list the spawned function names land in, so a test can assert which
        function was addressed as well as with what."""
        called: list[str] = []

        class _Function:
            def spawn(self, **kwargs: object) -> object:
                seen.update(kwargs)
                return type("_Call", (), {"object_id": "fc-new"})()

        def _from_name(_app: str, function: str) -> _Function:
            called.append(function)
            return _Function()

        monkeypatch.setattr(modal.Function, "from_name", staticmethod(_from_name))
        monkeypatch.setattr(
            "modal_app.launch.call_id_file", lambda function: tmp_path / f"{function}.txt"
        )
        monkeypatch.setattr("modal_app.launch.follow", lambda *_, **__: 0)
        return called

    def test_the_defaults_are_an_uncompiled_fresh_schedule_from_latest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)

        assert main([]) == 0
        assert seen["compile_model"] is False
        assert seen["schedule_start"] == 0
        assert seen["resume_from"] == "latest"
        assert seen["steps"] is None

    def test_every_flag_reaches_the_spawned_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)

        code = main(
            ["--compile", "--resume-from", "best", "--schedule-start", "4500", "--steps", "9000"]
        )
        assert code == 0
        assert seen["compile_model"] is True
        assert seen["resume_from"] == "best"
        assert seen["schedule_start"] == 4_500
        assert seen["steps"] == 9_000

    def test_the_call_id_is_written_where_a_later_invocation_can_find_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._spawner(monkeypatch, tmp_path, {})
        assert main([]) == 0
        assert (tmp_path / "pretrain.txt").read_text(encoding="utf-8").strip() == "fc-new"

    def test_the_finetune_is_spawned_not_run_so_no_client_can_cancel_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A multi-hour job launched by a blocking local entrypoint dies with its client.
        `finetune` must be addressable by name on the deployed app."""
        seen: dict[str, object] = {}
        called = self._spawner(monkeypatch, tmp_path, seen)

        assert main(["--function", "finetune", "--small", "--freeze", "--steps", "50"]) == 0
        assert called == ["finetune"]
        # Against the constant, not a literal: the run name is bumped every fine-tune, and
        # pinning the string here breaks two tests for a change that is not a defect.
        assert seen == {
            "run_name": MT_RUN_NAME,
            "freeze_embeddings": True,
            "max_steps": 50,
            "small": True,
        }

    def test_each_function_keeps_its_own_call_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One shared file would make `--cancel` after an MT launch terminate the encoder."""
        self._spawner(monkeypatch, tmp_path, {})
        assert main([]) == 0
        assert main(["--function", "finetune"]) == 0
        assert {p.name for p in tmp_path.iterdir()} == {"pretrain.txt", "finetune.txt"}

    def test_finetune_kwargs_bind_to_its_deployed_function(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)
        main(["--function", "finetune"])
        inspect.signature(mt_finetune.get_raw_f()).bind(**seen)

    def test_asr_kwargs_bind_to_its_deployed_function(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)
        main(["--function", "asr_finetune", "--epochs", "10", "--force"])
        inspect.signature(asr_finetune.get_raw_f()).bind(**seen)
        assert seen["run"] == ASR_RUN_NAME

    def test_the_fetch_and_train_pipeline_takes_the_same_kwargs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One spawned call covering both, so nothing local is needed between them."""
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)
        main(["--function", "asr_pipeline", "--epochs", "3"])
        inspect.signature(asr_pipeline.get_raw_f()).bind(**seen)

    def test_asr_without_epochs_leaves_the_function_default_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Passing `epochs=0` would train nothing; omitting it is what applies `EPOCHS`."""
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)
        main(["--function", "asr_finetune"])
        assert "epochs" not in seen

    def test_jugurtha_defaults_to_its_own_run_not_the_encoders(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`--run-name` carries the encoder's default, so a mapper reading it as falsy sends
        `agbalu-encoder-v1` and the run writes into another model's directory."""
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)
        main(["--function", "jugurtha_train"])
        inspect.signature(jugurtha_train.get_raw_f()).bind(**seen)
        assert seen["run"] == JUGURTHA_RUN_NAME

    def test_an_explicit_run_name_reaches_jugurtha(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)
        main(["--function", "jugurtha_train", "--run-name", "jugurtha-smoke"])
        assert seen["run"] == "jugurtha-smoke"

    def test_the_spawn_kwargs_bind_to_the_deployed_function(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A keyword the launcher sends and `pretrain` does not take fails on the GPU,
        after the image build has been paid for."""
        seen: dict[str, object] = {}
        self._spawner(monkeypatch, tmp_path, seen)

        main(["--compile", "--resume-from", "best", "--schedule-start", "4500", "--steps", "9000"])
        inspect.signature(pretrain.get_raw_f()).bind(**seen)

    def test_an_unknown_checkpoint_name_is_refused_before_any_gpu_is_paid_for(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._spawner(monkeypatch, tmp_path, {})
        with pytest.raises(SystemExit):
            main(["--resume-from", "penultimate"])


class TestKwargsTable:
    def test_every_spawnable_function_has_its_keywords(self) -> None:
        """The two lists must not drift: a function in `FUNCTIONS` with no entry here is a
        `KeyError` at launch, and one with wrong keywords is a `TypeError` on the worker."""
        assert set(KWARGS) == set(FUNCTIONS)

    def test_every_task_a_makefile_target_documents_is_spawnable(self) -> None:
        """A `modal-*` target builds `--function <prefix>_$(TASK)` by string concatenation, so
        a TASK its help line advertises but `FUNCTIONS` does not carry fails only when the
        operator runs it — after the deploy has already been paid for. It has happened: a
        target shipped advertising three tasks the launcher carried none of."""
        makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(encoding="utf-8")
        # One target at a time. A single regex over the whole file spans blank lines and
        # pairs one target's TASK list with another's launcher call, which is how this test
        # first reported five tifinagh tasks that `make acquire` advertises.
        advertised: set[str] = set()
        for block in makefile.split("\n\n"):
            tasks = re.search(r"TASK=([a-z|]+)", block)
            launched = re.search(r"modal_app\.launch --function ([a-z]+)_\$\(or \$\(TASK\)", block)
            if tasks and launched:
                advertised.update(f"{launched[1]}_{task}" for task in tasks[1].split("|"))
        assert advertised, "the Makefile scan matched nothing — the pattern has gone stale"
        unspawnable = sorted(name for name in advertised if name not in FUNCTIONS)
        assert not unspawnable, f"documented but not in FUNCTIONS: {unspawnable}"

    def test_a_document_needs_a_file_that_exists(self) -> None:
        args = argparse.Namespace(function="mt_predict", file="nope.txt", direction="eng-kab")
        with pytest.raises(SystemExit, match="does not exist"):
            spawn_kwargs(args)

    def test_a_document_sends_its_text_direction_and_name(self, tmp_path: Path) -> None:
        """`name` is what puts the translation on the volume, which is why a spawned
        document survives the client that asked for it."""
        source = tmp_path / "dracula.txt"
        source.write_text("Azul", encoding="utf-8")
        args = argparse.Namespace(function="mt_predict", file=str(source), direction="eng-kab")
        assert spawn_kwargs(args) == {
            "text": "Azul",
            "direction": "eng-kab",
            "name": "dracula",
        }

    def test_the_document_keywords_bind_to_the_deployed_function(self, tmp_path: Path) -> None:
        source = tmp_path / "doc.txt"
        source.write_text("Azul", encoding="utf-8")
        args = argparse.Namespace(function="mt_predict", file=str(source), direction="eng-kab")
        inspect.signature(mt_predict.get_raw_f()).bind(**spawn_kwargs(args))


class TestStopCommand:
    def test_it_stops_the_deployed_app_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        assert stop_command() == ["/usr/local/bin/modal", "app", "stop", APP_NAME, "--yes"]

    def test_it_does_not_wait_on_a_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`modal app stop` confirms interactively; a Makefile target that blocks on an
        unseen prompt looks like a hang."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        assert "--yes" in stop_command()

    def test_a_missing_cli_exits_with_a_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(SystemExit, match="not on PATH"):
            stop_command()


class TestCancel:
    """Cancelling the call frees the GPU; stopping the app is what lets the next
    `modal-train` deploy into a clean slot."""

    @staticmethod
    def _cancellable(monkeypatch: pytest.MonkeyPatch, cancelled: list[str]) -> None:
        class _Call:
            def __init__(self, call_id: str) -> None:
                self.call_id = call_id

            def cancel(self) -> None:
                cancelled.append(self.call_id)

        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(_Call))

    def test_it_cancels_the_call_and_then_stops_the_app(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cancelled: list[str] = []
        commands: list[list[str]] = []
        self._cancellable(monkeypatch, cancelled)

        def _record(command: list[str]) -> int:
            commands.append(command)
            return 0

        monkeypatch.setattr(subprocess, "call", _record)

        assert cancel("fc-1") == 0
        assert cancelled == ["fc-1"]
        assert commands == [stop_command()]

    def test_an_app_that_was_already_stopped_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`modal app stop` exits non-zero on an app that is already stopped, which is the
        normal state after a run finished on its own."""
        self._cancellable(monkeypatch, [])
        monkeypatch.setattr(subprocess, "call", lambda _: 1)

        assert cancel("fc-1") == 0
        assert "exited 1" in capsys.readouterr().out


class TestFollow:
    def test_it_returns_the_readers_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")
        monkeypatch.setattr(subprocess, "call", lambda _: 0)
        assert follow() == 0

    def test_an_interrupt_is_reported_as_harmless_not_propagated(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ctrl-C is the expected way to stop watching, so it must not look like a failure
        or leave the operator wondering whether the run died with it."""
        monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/modal")

        def interrupted(_command: list[str]) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(subprocess, "call", interrupted)
        assert follow() == 0
        assert "untouched" in capsys.readouterr().out


def _info(state: InputStatus, task_id: str = "ta-1") -> InputInfo:
    return InputInfo(
        input_id="in-1",
        function_call_id="fc-1",
        task_id=task_id,
        status=state,
        function_name="pretrain",
        module_name="modal_app.train",
        children=[],
    )


def _graph(*infos: InputInfo) -> object:
    class _Call:
        def get_call_graph(self) -> list[InputInfo]:
            return list(infos)

    return _Call()


class TestStatus:
    """The interrupt message sends the operator here, so it must answer without attaching —
    and without consuming the call's output, which would be its result, not its state."""

    def test_a_live_run_reads_as_pending_with_its_task(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            modal.FunctionCall,
            "from_id",
            staticmethod(lambda _: _graph(_info(InputStatus.PENDING))),
        )
        assert status("fc-1") == 0
        out = capsys.readouterr().out
        assert "pending" in out
        assert "ta-1" in out

    def test_a_dead_run_is_not_reported_as_pending(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            modal.FunctionCall,
            "from_id",
            staticmethod(lambda _: _graph(_info(InputStatus.FAILURE))),
        )
        assert status("fc-1") == 0
        out = capsys.readouterr().out
        assert "failed" in out
        assert "pending" not in out

    def test_a_queued_input_has_no_task_to_name(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            modal.FunctionCall,
            "from_id",
            staticmethod(lambda _: _graph(_info(InputStatus.PENDING, task_id=""))),
        )
        assert status("fc-1") == 0
        assert "task" not in capsys.readouterr().out

    def test_an_expired_call_says_so_rather_than_printing_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Modal's input retention is limited, so an old call id returns an empty graph.
        Silence there reads as "no run"."""
        monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda _: _graph()))
        assert status("fc-1") == 0
        assert "retention" in capsys.readouterr().out

    def test_the_flag_queries_the_given_call_id_and_never_starts_a_run(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[str] = []

        def _from_id(call_id: str) -> object:
            seen.append(call_id)
            return _graph(_info(InputStatus.SUCCESS))

        monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(_from_id))
        monkeypatch.setattr(
            "modal_app.launch.start", lambda *_, **__: pytest.fail("--status started a run")
        )
        assert main(["--status", "--call-id", "fc-42"]) == 0
        assert seen == ["fc-42"]
        assert "finished" in capsys.readouterr().out
