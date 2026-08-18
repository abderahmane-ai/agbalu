"""Start and inspect the pretraining run without staying attached to it.

`modal run` keeps a client connected for as long as the entrypoint blocks on `.remote()`,
and Modal propagates that client's disconnect as an input cancellation — a sleeping laptop
is enough to end the run. `--detach` does not cover it: its guarantee is about the app
surviving a parent that was killed or disconnected, and the call has a live parent
throughout.

So the run is started against a *deployed* app, whose lifetime is not tied to any local
session, with `spawn`, which returns a call id instead of waiting. Nothing stays connected,
so there is no connection whose loss can cancel anything.

    modal deploy -m modal_app.train      # once per code change
    python -m modal_app.launch           # start, print the call id, exit
    python -m modal_app.launch --status  # poll without attaching
    python -m modal_app.launch --cancel  # terminate the run and stop the app
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import modal
from modal.call_graph import InputStatus

from agbalu.model.config import DEFAULT_RUN_NAME
from agbalu.mt.finetune import DEFAULT_RUN_NAME as MT_RUN_NAME
from agbalu.tts.corpus import VOICE_NAMES
from agbalu.tts.training import DEFAULT_RUN_NAME as MATOUB_RUN_NAME
from modal_app.asr import DEFAULT_RUN as ASR_RUN_NAME
from modal_app.common import APP_NAME, MATOUB_BATCH
from modal_app.jugurtha import DEFAULT_RUN as JUGURTHA_RUN_NAME

FUNCTIONS: Final = (
    "pretrain",
    "finetune",
    "synthesise",
    "asr_finetune",
    "asr_pipeline",
    "asr_repack",
    "mt_predict",
    "tifinagh_train",
    "tifinagh_evaluate",
    "punctuation_train",
    "punctuation_evaluate",
    "tts_corpus",
    "matoub_train",
    "jugurtha_train",
)
"""The long-running functions. Each outlives any client, so each is spawned, never run —
synthesis included, because a full pivot pass is an hour of generation and `modal run`
would end it with the laptop, and `asr_finetune` because ten epochs is about seventeen.
`asr_repack` is here for the same reason and not because it needs a GPU: it reads 182,483
clips off a cold volume, which is hours on the one measurement that exists, so it is exactly
the length that must not hang off a laptop's connection. `tifinagh_evaluate` is minutes on a
GPU and spawned anyway, so that both halves of one job are addressed the same way.

The ASR and Tifinagh names carry their prefix because that is what they are registered as:
`modal_app.mt` already owns `finetune`, and a second `train` would silently override the
encoder's — modal logs a collision and carries on."""


def call_id_file(function: str) -> Path:
    """Git-ignored, one per function. Keeps the id off the terminal's scrollback, which is
    where the first run's app id was lost, and keeps two runs from overwriting each other's."""
    return Path(f"artifacts/{function}-call-id.txt")


def _remember(function: str, call_id: str) -> None:
    path = call_id_file(function)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(call_id + "\n", encoding="utf-8")


def _recall(function: str) -> str:
    path = call_id_file(function)
    if not path.is_file():
        message = f"no call id at {path}; start a {function} run first"
        raise SystemExit(message)
    return path.read_text(encoding="utf-8").strip()


def _modal_cli(*arguments: str, purpose: str) -> list[str]:
    """An argv for the installed `modal`. Resolved, fixed, and never a shell."""
    executable = shutil.which("modal")
    if executable is None:
        message = f"the `modal` CLI is not on PATH, so {purpose}"
        raise SystemExit(message)
    return [executable, *arguments]


def log_command(app_name: str = APP_NAME, call_id: str | None = None) -> list[str]:
    """The log stream, scoped to one call when its id is known.

    Every function in this project shares one app, so an unscoped stream interleaves whatever
    else is running — a Kokoro fine-tune's per-step lines arriving in the middle of a
    distillation run's is unreadable, and neither can be followed. `--function-call` filters
    by the call id, which is resolved against the *app name*: it therefore keeps working when
    the function itself is no longer deployed, unlike `--status`.
    """
    scope = ("--function-call", call_id) if call_id else ()
    return _modal_cli(
        "app", "logs", app_name, "--follow", *scope, purpose="logs cannot be followed"
    )


def stop_command(app_name: str = APP_NAME) -> list[str]:
    """`--yes` because the operator already asked, by cancelling."""
    return _modal_cli("app", "stop", app_name, "--yes", purpose="the app cannot be stopped")


def follow(app_name: str = APP_NAME, call_id: str | None = None) -> int:
    """Stream one call's logs as a reader.

    This process is not the run's parent — the job was spawned on a *deployed* app, whose
    lifetime no client shares. Interrupting here stops the reader and nothing else, which is
    the whole point of the deploy-and-spawn arrangement.
    """
    try:
        return subprocess.call(log_command(app_name, call_id))
    except KeyboardInterrupt:
        print("\nStopped watching. The run is untouched — `make modal-status` to check on it.")
        return 0


def start(function: str, **kwargs: object) -> str:
    """Spawn the named function and return its call id. Does not wait."""
    if function not in FUNCTIONS:
        message = f"{function!r} is not spawnable; have {list(FUNCTIONS)}"
        raise SystemExit(message)
    call = modal.Function.from_name(APP_NAME, function).spawn(**kwargs)
    call_id = call.object_id
    _remember(function, call_id)
    print(f"spawned {function} on the deployed app {APP_NAME!r}")
    print(f"  call id   {call_id}   (saved to {call_id_file(function)})")
    print(
        "\nThe run is detached. Ctrl-C, closing this terminal, sleeping or shutting down\n"
        f"affects only the log view below — never the run. `make modal-logs FUNCTION={function}`\n"
        "comes back, and shows this call's lines only.\n"
    )
    return call_id


_STATUS_TEXT: Final[dict[InputStatus, str]] = {
    InputStatus.PENDING: "pending — queued or running",
    InputStatus.SUCCESS: "finished",
    InputStatus.FAILURE: "failed",
    InputStatus.INIT_FAILURE: "failed to start",
    InputStatus.TERMINATED: "terminated",
    InputStatus.TIMEOUT: "timed out",
}


def status(call_id: str) -> int:
    """Report the call's state without attaching to it or consuming its result.

    Modal's input status does not separate queued from running, so both read as pending;
    the step counter exists only in the log.
    """
    graph = modal.FunctionCall.from_id(call_id).get_call_graph()
    if not graph:
        print(f"{call_id}: no input record — Modal's retention is limited. Try `make modal-logs`.")
        return 0
    for info in graph:
        state = _STATUS_TEXT.get(info.status, info.status.name.lower())
        task = f"   task {info.task_id}" if info.task_id else ""
        print(f"{call_id}: {info.function_name} {state}{task}")
    return 0


def cancel(call_id: str, app_name: str = APP_NAME) -> int:
    """Terminate the run, then stop the deployed app.

    Stopping the app terminates any container the cancelled call left behind, so the next
    `modal deploy` starts from nothing rather than rolling a live deployment forward.
    """
    modal.FunctionCall.from_id(call_id).cancel()
    print(f"{call_id}: cancelled")
    code = subprocess.call(stop_command(app_name))
    if code:
        print(f"`modal app stop {app_name}` exited {code} — read its message above.")
    else:
        print(f"app {app_name!r} stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--cancel", action="store_true", help="terminate the saved call id and stop the app"
    )
    mode.add_argument("--logs", action="store_true", help="follow the logs without starting a run")
    mode.add_argument("--status", action="store_true", help="report the call's state and exit")
    parser.add_argument("--call-id", help="override the saved call id")
    parser.add_argument(
        "--function",
        choices=FUNCTIONS,
        default="pretrain",
        help="which long-running function to spawn, follow or cancel",
    )
    parser.add_argument("--preset", default="kab")
    parser.add_argument("--steps", type=int, default=0, help="0 uses the configured max_steps")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--force",
        action="store_true",
        help="take over a run directory whose lock is still held (use after a hard kill)",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the forward pass (1.8x on an A10)"
    )
    parser.add_argument(
        "--schedule-start",
        type=int,
        default=0,
        help="continue a finished run: the previous max_steps, with a larger --steps",
    )
    parser.add_argument(
        "--resume-from",
        choices=("latest", "best"),
        default="latest",
        help="which checkpoint to continue from",
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="asr_finetune: 0 uses the configured EPOCHS"
    )
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="punctuation_evaluate: which checkpoint file to score",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=-1.0,
        help="asr_finetune: stop above this multiple of the expected s/step (0 disables)",
    )
    parser.add_argument("--split", default="test", help="tifinagh_evaluate: which held-out split")
    parser.add_argument(
        "--limit", type=int, default=0, help="tifinagh_evaluate: 0 scores the whole split"
    )
    parser.add_argument(
        "--stage", default="stage1", choices=("stage1", "stage2"), help="matoub_train: which stage"
    )
    parser.add_argument(
        "--arm", default="restored", choices=("restored", "raw"), help="matoub_train: corpus arm"
    )
    parser.add_argument(
        "--voice",
        default="",
        choices=("", *sorted(VOICE_NAMES.values())),
        help="matoub_train stage2: which voice to fine-tune. Stage 1 refuses it",
    )
    parser.add_argument(
        "--first-stage-epoch",
        type=int,
        default=-1,
        help="matoub_train stage2: which Stage 1 epoch to continue from (-1 is its last)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="matoub_train: frames per example, 0 taking the stage's own. The OOM lever",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="matoub_train: examples per step, 0 taking the default. Smoke it before a run",
    )
    parser.add_argument("--file", default="", help="mt_predict: the document to translate")
    parser.add_argument("--direction", default="eng-kab", help="mt_predict: translation direction")
    parser.add_argument(
        "--threads", type=int, default=0, help="asr_repack: decode threads (0 derives from cores)"
    )
    parser.add_argument(
        "--splits", nargs="*", help="asr_repack: which splits to pack, e.g. --splits train dev"
    )
    parser.add_argument("--small", action="store_true", help="finetune: the 600M ablation arm")
    parser.add_argument(
        "--targets", nargs="*", help="synthesise: NLLB codes to generate, e.g. arb_Arab"
    )
    parser.add_argument("--threshold", type=float, default=50.0, help="synthesise: chrF agreement")
    parser.add_argument(
        "--two-teacher-only",
        action="store_true",
        help="synthesise: keep only sentences both teachers translated",
    )
    parser.add_argument("--freeze", action="store_true", help="finetune: freeze the embeddings")
    args = parser.parse_args(argv)

    if args.cancel:
        return cancel(args.call_id or _recall(args.function))
    if args.status:
        return status(args.call_id or _recall(args.function))
    if args.logs:
        return follow(call_id=args.call_id or _recall(args.function))
    call_id = start(args.function, **spawn_kwargs(args))
    return follow(call_id=call_id)


def _document_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """`mt_predict`'s keywords: the document's text, read here rather than in the container.

    `name` is what makes the translation land on the volume, which is the whole reason a
    document is spawned — *Dracula* is thirteen GPU-minutes and a `modal run` client's
    disconnect cancelled it at ten, with nothing kept.
    """
    source = Path(args.file)
    if not source.is_file():
        message = f"{source} does not exist; --file is required to spawn a document"
        raise SystemExit(message)
    return {
        "text": source.read_text(encoding="utf-8"),
        "direction": args.direction,
        "name": source.stem,
    }


def _mt_finetune_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "run_name": MT_RUN_NAME if args.run_name == DEFAULT_RUN_NAME else args.run_name,
        "freeze_embeddings": args.freeze,
        "max_steps": args.steps,
        "small": args.small,
    }


def _repack_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {"threads": args.threads, "splits": ",".join(args.splits or ()), "force": args.force}


def _asr_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "run": ASR_RUN_NAME if args.run_name == DEFAULT_RUN_NAME else args.run_name,
        "max_steps": args.steps,
        "force": args.force,
    }
    # Omitted rather than passed as 0, so the function's own default is what applies.
    if args.epochs:
        kwargs["epochs"] = args.epochs
    # 0 is a meaningful value here — it disables the guard — so the sentinel is negative
    # rather than falsy. `asr_pipeline` does not take it.
    if args.function == "asr_finetune" and args.floor >= 0:
        kwargs["floor"] = args.floor
    return kwargs


def _tifinagh_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.function == "tifinagh_evaluate":
        return {"run": args.run_name, "split": args.split, "limit": args.limit}
    return {"run": args.run_name, "max_steps": args.steps, "force": args.force}


def _punctuation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.function == "punctuation_evaluate":
        return {"run": args.run_name, "split": args.split, "name": args.checkpoint}
    kwargs: dict[str, object] = {"run": args.run_name, "force": args.force}
    if args.epochs:
        kwargs["epochs"] = args.epochs
    return kwargs


def _synthesise_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "targets": ",".join(args.targets or ()),
        "threshold": args.threshold,
        "limit": args.steps,
        "two_teacher_only": args.two_teacher_only,
    }


def _jugurtha_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "epochs": args.epochs or 1,
        "steps": args.steps or 0,
        "run": JUGURTHA_RUN_NAME if args.run_name == DEFAULT_RUN_NAME else args.run_name,
        "force": args.force,
    }


def _pretrain_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "preset": args.preset,
        "steps": args.steps or None,
        "run_name": args.run_name,
        "force": args.force,
        "compile_model": args.compile,
        "schedule_start": args.schedule_start,
        "resume_from": args.resume_from,
    }


def _tts_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """`tts_corpus`'s keywords. `--limit` is its smoke and stays meaningful at 0, which is
    what the function reads as "every clip"."""
    return {
        "run": ASR_RUN_NAME if args.run_name == DEFAULT_RUN_NAME else args.run_name,
        "limit": args.limit,
        "force": args.force,
    }


def _matoub_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """`matoub_train`'s keywords. `--limit` is its smoke and routes the whole run to its own
    directory, so a capped launch can never write over a real stage's checkpoints.

    The stage/voice pairing is rejected here rather than in the container: it is pure Python,
    and the same refusal costs one A10G container per retry on the other side of a spawn.
    """
    if args.stage == "stage2" and not args.voice:
        message = (
            f"stage2 fine-tunes one voice: pass VOICE={'|'.join(sorted(VOICE_NAMES.values()))}"
        )
        raise SystemExit(message)
    if args.stage == "stage1" and args.voice:
        message = f"stage1 trains every voice at once; drop VOICE={args.voice}"
        raise SystemExit(message)
    return {
        "stage": args.stage,
        "arm": args.arm,
        "voice": args.voice,
        "epochs": args.epochs,
        "limit": args.limit,
        "first_stage_epoch": args.first_stage_epoch,
        "max_len": args.max_len,
        "batch": args.batch or MATOUB_BATCH,
        "run": MATOUB_RUN_NAME if args.run_name == DEFAULT_RUN_NAME else args.run_name,
    }


KWARGS: Final[dict[str, Callable[[argparse.Namespace], dict[str, object]]]] = {
    "finetune": _mt_finetune_kwargs,
    "mt_predict": _document_kwargs,
    "asr_repack": _repack_kwargs,
    "asr_finetune": _asr_kwargs,
    "asr_pipeline": _asr_kwargs,
    "tifinagh_train": _tifinagh_kwargs,
    "tifinagh_evaluate": _tifinagh_kwargs,
    "punctuation_train": _punctuation_kwargs,
    "punctuation_evaluate": _punctuation_kwargs,
    "synthesise": _synthesise_kwargs,
    "pretrain": _pretrain_kwargs,
    "jugurtha_train": _jugurtha_kwargs,
    "tts_corpus": _tts_kwargs,
    "matoub_train": _matoub_kwargs,
}
"""One builder per spawnable function. A table rather than a chain of branches because the
two lists must not drift: the test asserts every name in `FUNCTIONS` has an entry, so adding
a function without its keywords fails locally instead of raising `TypeError` on the worker
hours after launch."""


def spawn_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """The keywords the chosen function takes. They do not overlap, so passing one
    function's flags to another would be a `TypeError` on the worker, hours after launch."""
    return KWARGS[args.function](args)


if __name__ == "__main__":
    sys.exit(main())
