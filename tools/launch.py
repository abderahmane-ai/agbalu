"""The school HPC launcher (`~/.claude/standards/hpc.md` §6).

Every remote action goes through the `hpc` SSH alias. Long work runs in a detached tmux
session on the pod and this returns at once with the run id; `status`, `logs` and `pull`
read it back later. Nothing here waits on a job.

    python3 -m tools.launch setup|sync|status|logs|pull|train|smoke|data|evaluate|merge
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from agbalu.llm.provenance import STAMP_FILE, git_stamp
from agbalu.llm.smoke import SMOKE_PARTS
from tools.run_log import Printer, View, summary

HOST: Final = "hpc"
PROJECT: Final = "agbalu"
REMOTE_ROOT: Final = f"/home/jovyan/{PROJECT}"
ENVIRONMENT: Final = f"/home/jovyan/envs/{PROJECT}"
PYTHON: Final = f"{ENVIRONMENT}/bin/python"
VLLM_PYTHON: Final = "/home/jovyan/envs/vllm/bin/python"
"""vLLM 0.30.0 pins torch 2.13, so it lives beside the training environment, not in it."""
LOCK: Final = Path("requirements/hpc.lock")
RUNS: Final = Path("artifacts/runs")

COMPILER: Final = f"{ENVIRONMENT}/bin/x86_64-conda-linux-gnu"
PREAMBLE: Final = (
    ". ~/hpc-env.sh"
    f" && export CC={COMPILER}-gcc CXX={COMPILER}-g++"
    f" HF_HOME={REMOTE_ROOT}/artifacts/hf PYTHONPATH=src"
    f" && cd {REMOTE_ROOT}"
)
"""`hpc-env.sh` puts the tools on `PATH`; `CC`/`CXX` then point at the project environment's
gcc 13, which TileLang's nvcc JIT needs, rather than the tools environment's."""

EXCLUDES: Final = (
    ".git",
    "artifacts",
    "data",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "*.egg-info",
    "graphify-out",
)

DATA_FILES: Final = (
    "data/processed/llm/heldout-kab.jsonl",
    "data/processed/llm/heldout-eng.jsonl",
    "data/processed/llm/heldout-fra.jsonl",
    "data/processed/llm/kab-docs.jsonl",
    "data/processed/llm/kab-docs.stats.json",
    "data/processed/llm/aligned.jsonl",
    "data/processed/llm/aligned.stats.json",
    "data/processed/bench/flores-plus-kab_Latn-corrected.jsonl",
    "data/raw/hf.flores-plus-kab/dev/kab_Latn.jsonl",
    "data/raw/hf.flores-plus-kab/dev/eng_Latn.jsonl",
    "data/raw/hf.flores-plus-kab/dev/fra_Latn.jsonl",
    "data/raw/hf.flores-plus-kab/devtest/kab_Latn.jsonl",
    "data/raw/hf.flores-plus-kab/devtest/eng_Latn.jsonl",
    "data/raw/hf.flores-plus-kab/devtest/fra_Latn.jsonl",
    "data/raw/hf.sib200-kab/kab_Latn/train-00000-of-00001.parquet",
    "data/raw/hf.sib200-kab/kab_Latn/validation-00000-of-00001.parquet",
    "data/raw/hf.sib200-kab/kab_Latn/test-00000-of-00001.parquet",
    "data/processed/text/agbalu-text-v1.jsonl",
    "data/processed/mt/train.jsonl",
    "data/raw/hf.hplt2-cleaned-kab",
    "data/raw/hf.fineweb2-kab",
    "data/raw/hf.finetranslations-kab",
    "data/raw/hf.glotcc-v1-kab",
    "data/raw/hf.finepdfs-kab",
    "data/raw/hf.dcad2000-kab",
    "data/raw/hf.imsidag.kabyle-datasets-mono",
    "data/raw/wikimedia.kabwiki",
    "artifacts/runs/dihya",
    "data/processed/llm/sft/sft-v1.jsonl",
    "data/interim/llm/sft/candidates.jsonl",
)
"""What the pod reads that the code sync excludes: every evaluation set, the inputs of the
document and aligned builds, and Dihya. The decontaminator refuses to run unless every
evaluation set is present."""

DATA_TASKS: Final = ("documents", "aligned", "replay", "pack", "sft", "fluency")
WEIGHT_EXCLUDES: Final = ("--exclude", "step-*", "--exclude", "best", "--exclude", "model")
"""What `pull` leaves on the pod unless `--weights`: checkpoints, the best model and a merge."""
LOG_LINES: Final = 20
STATUS_BACKLOG: Final = 600
"""Raw log lines `status` reads: about four training steps with their progress lines."""

STATUS_EVENTS: Final = 12
STREAM_BACKLOG: Final = 400
LOG_MARKER: Final = "@@log@@"
CHECKPOINT_MARKER: Final = "@@checkpoints@@"
BASELINE_MARKER: Final = "@@baseline@@"


class LaunchError(Exception):
    """A remote action failed or was asked for something it cannot do."""


def remote(command: str) -> list[str]:
    """`ssh hpc` running `command` after the environment preamble."""
    return ["ssh", "-o", "ConnectTimeout=15", HOST, f"{PREAMBLE} && {command}"]


def code_sync() -> list[str]:
    excludes = [argument for pattern in EXCLUDES for argument in ("--exclude", pattern)]
    return ["rsync", "-az", "--delete", *excludes, "./", f"{HOST}:{REMOTE_ROOT}/"]


def data_sync(files: Sequence[str]) -> list[str]:
    """Paths kept relative, so each lands at the same place under the remote root."""
    return ["rsync", "-azR", *files, f"{HOST}:{REMOTE_ROOT}/"]


def session_name(run: str) -> str:
    return f"{PROJECT}-{run}".replace("/", "-").replace(".", "-")


def module(arguments: Sequence[str]) -> str:
    return " ".join([PYTHON, "-m", "agbalu.llm.cli", *(shlex.quote(a) for a in arguments)])


def chat_module(arguments: Sequence[str]) -> str:
    """The chat-mode evaluator in the vLLM environment, which shares nothing with training's."""
    quoted = (shlex.quote(a) for a in arguments)
    return " ".join([VLLM_PYTHON, "-m", "agbalu.llm.chat_eval", *quoted])


def detached(run: str, arguments: Sequence[str], log: str, then: Sequence[str] = ()) -> list[str]:
    """A tmux session running the CLI, then each command of `then` while all succeed, with
    their output in the run's log file.

    A log left by an earlier start is renamed with its time first, so `status` and `logs`
    show this start only; a resumed run's history stays beside it.
    """
    log_path = shlex.quote(f"{REMOTE_ROOT}/{RUNS}/{log}")
    chain = " && ".join([module(arguments), *then])
    job = f"{PREAMBLE} && {{ {chain}; }} > {log_path} 2>&1"
    command = (
        f"mkdir -p {shlex.quote(str(Path(f'{REMOTE_ROOT}/{RUNS}/{log}').parent))}"
        f" && (test ! -f {log_path} || mv {log_path} {log_path}.$(date +%Y%m%d-%H%M%S))"
        f" && tmux new-session -d -s {session_name(run)} {shlex.quote(job)}"
    )
    return remote(command)


def setup_script() -> str:
    """Build the environment from the lock if it is missing, then report what is there."""
    probe = (
        "import torch, transformers, shutil;"
        "free, total = torch.cuda.mem_get_info();"
        "print('gpu', torch.cuda.get_device_name(0), round(free / 2**30, 1), 'GiB free');"
        "print('torch', torch.__version__, 'transformers', transformers.__version__);"
        f"print('disk', round(shutil.disk_usage('{REMOTE_ROOT}').free / 2**30), 'GiB free')"
    )
    build = (
        f"test -x {PYTHON} || (mamba create -y -p {ENVIRONMENT} -c conda-forge "
        "python=3.12 gcc_linux-64=13 gxx_linux-64=13"
        f" && {PYTHON} -m pip install -r {LOCK})"
    )
    return f"{build} && {PYTHON} -c {shlex.quote(probe)}"


def baseline_command(run: str) -> str:
    """The run's first evaluation from its event file, which a log tail may not reach."""
    events = f"{REMOTE_ROOT}/{RUNS}/{run}/log.jsonl"
    return f'grep -m 1 \'"event": "eval"\' {events} 2>/dev/null || true'


def status_script(run: str, lines: int) -> str:
    """Session state, the log's tail and the newest checkpoints, between section markers."""
    runs = f"{REMOTE_ROOT}/{RUNS}/{run}"
    return (
        f"(tmux has-session -t {session_name(run)} 2>/dev/null && echo 'session: running'"
        " || echo 'session: not running');"
        f" echo {BASELINE_MARKER}; {baseline_command(run)};"
        f" echo {LOG_MARKER}; tail -n {lines} {runs}/log.txt 2>/dev/null;"
        f" echo {CHECKPOINT_MARKER}; ls -d {runs}/step-* {runs}/*/step-* 2>/dev/null | tail -n 3"
    )


def readable_status(output: str) -> list[str]:
    """The status script's output with its log section rendered for a person."""
    head, _, rest = output.partition(BASELINE_MARKER)
    baseline, _, rest = rest.partition(LOG_MARKER)
    log, _, saved = rest.partition(CHECKPOINT_MARKER)
    lines = [head.strip(), *summary(log.splitlines(), STATUS_EVENTS, baseline.strip())]
    found = [line.strip() for line in saved.splitlines() if line.strip()]
    lines.append("checkpoints: " + (", ".join(Path(p).name for p in found) if found else "none"))
    return lines


def show_status(run: str, *, raw: bool) -> None:
    if raw:
        execute(remote(status_script(run, LOG_LINES)))
        return
    result = subprocess.run(
        remote(status_script(run, STATUS_BACKLOG)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        message = f"status failed with exit {result.returncode}: {result.stderr.strip()[:300]}"
        raise LaunchError(message)
    print("\n".join(readable_status(result.stdout)))


def write_now(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_log(run: str, *, raw: bool) -> None:
    """Follow the log until Ctrl-C, which ends the stream on this machine and nothing else."""
    path = f"{REMOTE_ROOT}/{RUNS}/{run}/log.txt"
    if raw:
        execute(["ssh", "-t", HOST, f"tail -f {path}"])
        return
    view = View()
    baseline = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", HOST, baseline_command(run)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    view.seed(baseline.stdout)
    printer = Printer(write_now, terminal=sys.stdout.isatty())
    process = subprocess.Popen(
        ["ssh", "-o", "ConnectTimeout=15", HOST, f"tail -n {STREAM_BACKLOG} -F {path}"],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        if process.stdout is not None:
            for line in process.stdout:
                printer.show(view.render(line))
    except KeyboardInterrupt:
        write_now("\n")
    finally:
        process.terminate()
        process.wait()


def stop_script(run: str) -> str:
    """Ctrl-C into the session: the trainer checkpoints at the end of its current step and
    exits; any other job stops at once."""
    session = session_name(run)
    return (
        f"tmux send-keys -t {session} C-c 2>/dev/null && echo 'sent Ctrl-C to {session}'"
        f" || echo 'no session {session}'"
    )


def execute(command: Sequence[str]) -> None:
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        message = f"exit {result.returncode}: {shlex.join(command)[:300]}"
        raise LaunchError(message)


def stamp(root: Path = Path()) -> Path:
    """The laptop checkout's revision, for a tree synced without `.git`."""
    payload = {**git_stamp(root), "synced": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
    return Path(handle.name)


def sync() -> None:
    execute(code_sync())
    present = [path for path in DATA_FILES if Path(path).exists()]
    absent = sorted(set(DATA_FILES) - set(present))
    if present:
        execute(data_sync(present))
    stamp_file = stamp()
    try:
        execute(["rsync", "-az", str(stamp_file), f"{HOST}:{REMOTE_ROOT}/{STAMP_FILE}"])
    finally:
        stamp_file.unlink()
    print(f"synced code and {len(present)} data files; absent here: {absent or 'none'}")


def record(run: str, arguments: Sequence[str], log: str, then: Sequence[str] = ()) -> None:
    """`artifacts/runs/<run>/hpc.session` on the laptop: what was started and where it logs."""
    path = RUNS / run / "hpc.session"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": run,
        "session": session_name(run),
        "command": " && ".join([module(arguments), *then]),
        "log": f"{REMOTE_ROOT}/{RUNS}/{log}",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def start(run: str, arguments: Sequence[str], log: str, then: Sequence[str] = ()) -> None:
    sync()
    execute(detached(run, arguments, log, then))
    record(run, arguments, log, then)
    print(f"started {session_name(run)}; run id {run}; log {REMOTE_ROOT}/{RUNS}/{log}")


def train_arguments(args: argparse.Namespace) -> list[str]:
    arguments = ["train", "--run", args.run]
    options = (
        ("--model", args.model),
        ("--total-steps", args.steps),
        ("--stop-after", args.stop),
        ("--objective", args.objective),
    )
    for flag, value in options:
        if value is not None:
            arguments += [flag, str(value)]
    return arguments


def merge_arguments(args: argparse.Namespace) -> list[str]:
    """`--model` names the CPT checkpoint the delta is taken from."""
    if args.alpha is None:
        message = "merge needs --alpha"
        raise LaunchError(message)
    cpt = require(args.model, "--model", "merge")
    return ["merge", "--run", args.run, "--alpha", str(args.alpha), "--cpt", cpt]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="launch", description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "setup",
            "sync",
            "status",
            "logs",
            "pull",
            "stop",
            "train",
            "smoke",
            "data",
            "evaluate",
            "merge",
        ),
    )
    parser.add_argument("--target", choices=("hpc",), default="hpc")
    parser.add_argument("--run")
    parser.add_argument("--part", choices=SMOKE_PARTS)
    parser.add_argument("--task", choices=DATA_TASKS)
    parser.add_argument("--model")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop", type=int, help="end the train invocation at this step")
    parser.add_argument("--alpha", type=float, help="merge coefficient")
    parser.add_argument("--objective", choices=("cpt", "sft"), help="train only; CLI default cpt")
    parser.add_argument(
        "--suite",
        choices=("exit", "select", "release"),
        default="exit",
        help="exit: completion suite only; select and release add the chat suite",
    )
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--raw", action="store_true", help="status and logs as the raw log")
    return parser


def require(value: str | None, flag: str, action: str) -> str:
    if not value:
        message = f"{action} needs {flag}"
        raise LaunchError(message)
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.action:
            case "setup":
                execute(remote(f"mkdir -p {REMOTE_ROOT}"))
                sync()
                execute(remote(setup_script()))
            case "stop":
                execute(remote(stop_script(require(args.run, "--run", "stop"))))
            case "sync":
                sync()
            case "status":
                show_status(require(args.run, "--run", "status"), raw=args.raw)
            case "logs":
                stream_log(require(args.run, "--run", "logs"), raw=args.raw)
            case "pull":
                run = require(args.run, "--run", "pull")
                skip = [] if args.weights else [*WEIGHT_EXCLUDES]
                local = RUNS / run
                local.mkdir(parents=True, exist_ok=True)
                execute(["rsync", "-az", *skip, f"{HOST}:{REMOTE_ROOT}/{RUNS}/{run}/", f"{local}/"])
            case _:
                start(*job(args))
    except LaunchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


Job = tuple[str, list[str], str, list[str]]
"""Run id, CLI arguments, log file, and the commands chained after the CLI."""


def job(args: argparse.Namespace) -> Job:
    """What an action that starts a detached job runs."""
    match args.action:
        case "train":
            run = require(args.run, "--run", "train")
            return run, train_arguments(args), f"{run}/log.txt", []
        case "smoke":
            run = require(args.run, "--run", "smoke")
            part = require(args.part, "--part", "smoke")
            arguments = ["smoke", "--part", part, "--run", run]
            return f"{run}/{part}", arguments, f"{run}/{part}/log.txt", []
        case "data":
            task = require(args.task, "--task", "data")
            return f"data-{task}", [task], f"data-{task}/log.txt", []
        case "merge":
            run = require(args.run, "--run", "merge")
            return run, merge_arguments(args), f"{run}/log.txt", []
        case _:
            return evaluation_job(args)


def evaluation_job(args: argparse.Namespace) -> Job:
    """The completion-mode suite, then for `select` and `release` the chat-mode suite in each
    thinking mode, each mode its own process because one vLLM engine is one process."""
    run = require(args.run, "--run", "evaluate")
    arguments = ["evaluate", "--out", f"{RUNS}/{run}/evaluation.json"]
    if args.model:
        arguments += ["--model", args.model]
    if args.suite == "exit":
        return run, arguments, f"{run}/log.txt", []
    model = require(args.model, "--model", f"evaluate --suite {args.suite}")
    chained = [
        chat_module(
            [
                "--model",
                model,
                "--suite",
                args.suite,
                "--mode",
                mode,
                "--out",
                f"{RUNS}/{run}/chat-{mode}.json",
            ]
        )
        for mode in ("no_think", "think")
    ]
    return run, arguments, f"{run}/log.txt", chained


if __name__ == "__main__":
    raise SystemExit(main())
