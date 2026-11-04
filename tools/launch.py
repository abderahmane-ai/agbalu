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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from agbalu.codec_tts.configuration import SMOKE_PARTS, is_run_name
from agbalu.llm.provenance import STAMP_FILE, git_stamp
from tools.run_log import Printer, View, summary

HOST: Final = "hpc"
PROJECT: Final = "agbalu"
REMOTE_ROOT: Final = f"/home/jovyan/{PROJECT}"
ENVIRONMENT: Final = f"/home/jovyan/envs/{PROJECT}"
PYTHON: Final = f"{ENVIRONMENT}/bin/python"
CODEC_ENVIRONMENT: Final = f"/home/jovyan/envs/{PROJECT}-codec"
CODEC_PYTHON: Final = f"{CODEC_ENVIRONMENT}/bin/python"
VLLM_PYTHON: Final = "/home/jovyan/envs/vllm/bin/python"
"""vLLM 0.30.0 pins torch 2.13, so it lives beside the training environment, not in it."""
LOCK: Final = Path("requirements/hpc.lock")
RUNS: Final = Path("artifacts/runs")

QUIET: Final = (
    "TQDM_MININTERVAL=60",
    "HF_HUB_DISABLE_PROGRESS_BARS=1",
    "HF_DATASETS_DISABLE_PROGRESS_BARS=1",
    "TRANSFORMERS_VERBOSITY=error",
    "LMEVAL_LOG_LEVEL=ERROR",
    "VLLM_LOGGING_LEVEL=WARNING",
    "PYTHONWARNINGS='ignore::DeprecationWarning,ignore::FutureWarning,"
    "ignore:NLTK will not authorize:UserWarning'",
)
"""Library chatter kept out of run logs: INFO lines, deprecation notices, NLTK's warning about
the pod's group-writable home, which we cannot change, and progress bars cut to one line a
minute. Errors, other warnings and our own events still print (probed on the pod,
2026-10-17). Bars are slowed, not disabled: vLLM divides by a bar's elapsed time, which a
disabled bar reports as 0, and the chat evaluation died on it."""

COMPILER: Final = f"{ENVIRONMENT}/bin/x86_64-conda-linux-gnu"
PREAMBLE: Final = (
    ". ~/hpc-env.sh"
    f" && export CC={COMPILER}-gcc CXX={COMPILER}-g++ LD_LIBRARY_PATH={ENVIRONMENT}/lib"
    f" HF_HOME={REMOTE_ROOT}/artifacts/hf PYTHONPATH=src"
    " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_NO_CUDA_NVML=1"
    f" {' '.join(QUIET)}"
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
    "data/raw/hf.imsidag.kabyle-corpus-ummto",
    "data/raw/hf.imsidag.kabyle-corpus-ubouira",
    "data/raw/hf.imsidag.kabyle-corpus-hca",
    "data/raw/wikimedia.kabwiki",
    "artifacts/runs/dihya",
    "data/processed/llm/sft/sft-v1.jsonl",
    "data/interim/llm/sft/candidates.jsonl",
)
"""What the pod reads that the code sync excludes: every evaluation set, the inputs of the
document and aligned builds, and Dihya. The decontaminator refuses to run unless every
evaluation set is present."""

DATA_TASKS: Final = (
    "documents",
    "aligned",
    "replay",
    "pack",
    "sft",
    "fluency",
    "expand-fetch",
    "expand-pdfs",
    "expand-audio",
    "expand-clean",
    "expand-aligned",
    "compile",
)
RESUMABLE_TASKS: Final = ("expand-pdfs", "compile")
"""Data tasks whose output a later run appends to instead of rebuilding (`--resume`)."""
WEIGHT_EXCLUDES: Final = (
    "--exclude",
    "step-*/",
    "--exclude",
    "best",
    "--exclude",
    "model",
    "--exclude",
    "*.pt",
    "--exclude",
    "*.pth",
    "--exclude",
    "*.safetensors",
)
"""What `pull` leaves on the pod unless `--weights`: checkpoints, the best model and a merge."""
LOG_LINES: Final = 20
STATUS_BACKLOG: Final = 600
"""Raw log lines `status` reads: about four training steps with their progress lines."""

STATUS_EVENTS: Final = 12
STREAM_BACKLOG: Final = 400
LOG_MARKER: Final = "@@log@@"
CHECKPOINT_MARKER: Final = "@@checkpoints@@"
BASELINE_MARKER: Final = "@@baseline@@"
ROLES_MARKER: Final = "@@roles@@"


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
    if arguments and arguments[0] == "__raw__":
        return " ".join(arguments[1:])
    families = {
        "__encoder__": "agbalu.encoder.cli",
        "__restoration__": "agbalu.restoration.cli",
        "__dihya__": "agbalu.dihya.cli",
        "__codec_tts__": "agbalu.codec_tts.cli",
    }
    marker = arguments[0] if arguments else ""
    package = families.get(marker, "agbalu.llm.cli")
    selected = arguments[1:] if marker in families else arguments
    interpreter = CODEC_PYTHON if marker == "__codec_tts__" else PYTHON
    command = " ".join([interpreter, "-m", package, *(shlex.quote(a) for a in selected)])
    if marker in families and selected and selected[0] == "smoke":
        return f"timeout --signal=TERM --kill-after=30s 840s {command}"
    if marker == "__restoration__" and selected and selected[0] == "prepare":
        return f"timeout --signal=TERM --kill-after=30s 1800s {command}"
    return command


VLLM_ENVIRONMENT: Final = "VLLM_USE_FLASHINFER_SAMPLER=0"
"""FlashInfer's sampler compiles its kernel on first use and the pod has no CUDA toolkit, so
the engine died in warmup (merge-a000, 2026-10-17). vLLM's PyTorch sampler needs no compiler."""


def chat_module(arguments: Sequence[str]) -> str:
    """The chat-mode evaluator in the vLLM environment, which shares nothing with training's."""
    quoted = (shlex.quote(a) for a in arguments)
    return " ".join([VLLM_ENVIRONMENT, VLLM_PYTHON, "-m", "agbalu.llm.chat_eval", *quoted])


def chain(arguments: Sequence[str], then: Sequence[str]) -> str:
    """The CLI on `arguments`, when there are any, then each command of `then`."""
    return " && ".join([*([module(arguments)] if arguments else []), *then])


def detached(run: str, arguments: Sequence[str], log: str, then: Sequence[str] = ()) -> list[str]:
    """A tmux session running the CLI, then each command of `then` while all succeed, with
    their output in the run's log file.

    A log left by an earlier start is renamed with its time first, so `status` and `logs`
    show this start only; a resumed run's history stays beside it.
    """
    log_path = shlex.quote(f"{REMOTE_ROOT}/{RUNS}/{log}")
    job = f"{PREAMBLE} && {{ {chain(arguments, then)}; }} > {log_path} 2>&1"
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


def setup_codec_script() -> str:
    build = (
        f"test -x {CODEC_PYTHON} || mamba create -y -p {CODEC_ENVIRONMENT} "
        "-c conda-forge python=3.12"
    )
    return (
        f"({build}) && {CODEC_PYTHON} -m pip install -r requirements/codec_tts.lock"
        f" && {CODEC_PYTHON} -m agbalu.codec_tts.cli environment"
    )


def baseline_command(run: str) -> str:
    """The run's first evaluation from its event file, which a log tail may not reach."""
    events = f"{REMOTE_ROOT}/{RUNS}/{run}/log.jsonl"
    return f'grep -m 1 \'"event": "eval"\' {events} 2>/dev/null || true'


def status_script(run: str, lines: int) -> str:
    """Session state, the log's tail and the newest checkpoints, between section markers."""
    runs = f"{REMOTE_ROOT}/{RUNS}/{run}"
    roles = (
        "import json; from pathlib import Path; "
        f"root=Path({runs!r}); "
        "print(json.dumps({name:json.loads((root/(name+'.json')).read_text()) "
        "for name in ('latest','best_mlm','best_deployment','best_ce','best_speech') "
        "if (root/(name+'.json')).is_file()}))"
    )
    return (
        f"(tmux has-session -t {session_name(run)} 2>/dev/null && echo 'session: running'"
        " || echo 'session: not running');"
        f" echo {BASELINE_MARKER}; {baseline_command(run)};"
        f" echo {LOG_MARKER}; tail -n {lines} {runs}/log.txt 2>/dev/null;"
        f" echo {CHECKPOINT_MARKER}; "
        f"find {runs} -maxdepth 2 \\( -type d -name 'step-*' -o -name '*.pt' \\) "
        "2>/dev/null | sort | tail -n 3;"
        f" echo {ROLES_MARKER}; {PYTHON} -c {shlex.quote(roles)}"
    )


def readable_status(output: str) -> list[str]:
    """The status script's output with its log section rendered for a person."""
    head, _, rest = output.partition(BASELINE_MARKER)
    baseline, _, rest = rest.partition(LOG_MARKER)
    log, _, saved = rest.partition(CHECKPOINT_MARKER)
    summarised = summary(log.splitlines(), STATUS_EVENTS, baseline.strip())
    if summarised:
        lines = [head.strip(), *summarised]
    else:
        recent_log = [line.strip() for line in log.splitlines() if line.strip()][-12:]
        lines = [head.strip(), *recent_log]
    saved, _, roles = saved.partition(ROLES_MARKER)
    found = [line.strip() for line in saved.splitlines() if line.strip()]
    lines.append("checkpoints: " + (", ".join(Path(p).name for p in found) if found else "none"))
    if roles.strip():
        payload = json.loads(roles.strip())
        if isinstance(payload, dict) and payload:
            for role in ("latest", "best_mlm", "best_deployment", "best_ce", "best_speech"):
                item = payload.get(role)
                selected = item.get("snapshot") if isinstance(item, dict) else "none eligible"
                lines.append(f"{role}: {selected}")
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
    if raw or run == "encbench":
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


def sync_encoder() -> None:
    """Sync code and tokenizer inputs without replacing the HPC's existing Kabyle corpus."""
    files = [
        "artifacts/encoder/measurements/agbalu-tok-base-16k.model",
        "data/raw/git.ud-kabyle-adpt/kab_adpt-ud-train.conllu",
        "data/raw/git.ud-kabyle-adpt/kab_adpt-ud-test.conllu",
    ]
    local_tokenizer = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
    destination = Path(files[0])
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(local_tokenizer.read_bytes())
    for file in files:
        if not Path(file).is_file():
            message = f"encoder sync requires {file}"
            raise LaunchError(message)
    execute(code_sync())
    execute(data_sync(files))
    stamp_file = stamp()
    try:
        execute(["rsync", "-az", str(stamp_file), f"{HOST}:{REMOTE_ROOT}/{STAMP_FILE}"])
    finally:
        stamp_file.unlink()
    print("synced encoder code, Mammeri and treebank; existing HPC corpus preserved")


def sync_restoration() -> None:
    """Sync restoration sources while retaining the HPC's tokenizer, corpus and checkpoints."""
    execute(code_sync())
    stamp_file = stamp()
    try:
        execute(["rsync", "-az", str(stamp_file), f"{HOST}:{REMOTE_ROOT}/{STAMP_FILE}"])
    finally:
        stamp_file.unlink()
    print("synced restoration code; existing HPC data and pretrained checkpoints preserved")


def sync_dihya() -> None:
    """Sync Dihya-22M code and sources without replacing the HPC's existing data."""
    execute(code_sync())
    data_to_sync: list[str] = []
    negatives = Path("data/raw/numalia_negatives.jsonl")
    if negatives.is_file():
        data_to_sync.append(str(negatives))
    cmn = Path("data/raw/cmn_sentences.tsv.bz2")
    if cmn.is_file():
        data_to_sync.append(str(cmn))
    if data_to_sync:
        execute(data_sync(data_to_sync))
    stamp_file = stamp()
    try:
        execute(["rsync", "-az", str(stamp_file), f"{HOST}:{REMOTE_ROOT}/{STAMP_FILE}"])
    finally:
        stamp_file.unlink()
    print("synced Dihya-22M code and data registries; existing HPC data preserved")


def sync_codec_tts() -> None:
    files = [
        "src/agbalu/codec_tts",
        "src/agbalu/__init__.py",
        "src/agbalu/_about.py",
        "src/agbalu/llm/__init__.py",
        "src/agbalu/llm/progress.py",
        "src/agbalu/llm/provenance.py",
        "resources/codec_tts",
        "requirements/codec_tts.lock",
        "tools/export_checkpoint.py",
    ]
    execute(data_sync(files))
    stamp_file = stamp()
    try:
        execute(["rsync", "-az", str(stamp_file), f"{HOST}:{REMOTE_ROOT}/{STAMP_FILE}"])
    finally:
        stamp_file.unlink()
    print("synced codec TTS sources; existing HPC data and other model families preserved")


def sync_family(family: str) -> None:
    actions: dict[str, Callable[[], None]] = {
        "restoration": sync_restoration,
        "encoder": sync_encoder,
        "dihya": sync_dihya,
        "codec_tts": sync_codec_tts,
        "llm": sync,
    }
    actions[family]()


def record(run: str, arguments: Sequence[str], log: str, then: Sequence[str] = ()) -> None:
    """`artifacts/runs/<run>/hpc.session` on the laptop: what was started and where it logs."""
    path = RUNS / run / "hpc.session"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": run,
        "session": session_name(run),
        "command": chain(arguments, then),
        "log": f"{REMOTE_ROOT}/{RUNS}/{log}",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def start(run: str, arguments: Sequence[str], log: str, then: Sequence[str] = ()) -> None:
    if arguments and arguments[0] == "__restoration__":
        sync_restoration()
    elif arguments and arguments[0] == "__encoder__":
        sync_encoder()
    elif arguments and arguments[0] == "__dihya__":
        sync_dihya()
    elif arguments and arguments[0] == "__codec_tts__":
        sync_codec_tts()
    else:
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


def assemble_arguments(args: argparse.Namespace) -> list[str]:
    """`--model` names the trained tower, `--template` the full model it was trained from."""
    tower = require(args.model, "--model", "assemble")
    template = require(args.template, "--template", "assemble")
    return ["assemble", "--run", args.run, "--tower", tower, "--template", template]


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
            "assemble",
            "encbench",
        ),
    )
    parser.add_argument("--target", choices=("hpc",), default="hpc")
    parser.add_argument(
        "--family",
        choices=("llm", "encoder", "restoration", "dihya", "codec_tts"),
        default="llm",
    )
    parser.add_argument("--run")
    parser.add_argument("--part")
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--task",
        choices=(*DATA_TASKS, "prepare", "inventory", "partition", "encode", "language", "fetch"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="data: append to an earlier run's output instead of rebuilding it "
        f"({', '.join(RESUMABLE_TASKS)})",
    )
    parser.add_argument("--model")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-tokens", type=int)
    parser.add_argument("--micro-tokens", type=int)
    parser.add_argument("--micro-windows", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--config")
    parser.add_argument("--stop", type=int, help="end the train invocation at this step")
    parser.add_argument("--alpha", type=float, help="merge coefficient")
    parser.add_argument("--template", help="assemble: the full model the tower was trained from")
    parser.add_argument("--objective", choices=("cpt", "sft"), help="train only; CLI default cpt")
    parser.add_argument(
        "--suite",
        choices=(
            "exit",
            "screen",
            "guard",
            "kabyle",
            "select",
            "release",
            "dev",
            "baseline",
            "validation",
        ),
        default="exit",
        help="exit: completion suite only; screen: translation only, into screen.json; "
        "kabyle: heldout, mt, and sib into evaluation-kabyle.json; "
        "guard: no-think IFEval and GSM8K into chat-no_think.json; "
        "select and release add the chat suite",
    )
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="select and release: the chat suite alone, when evaluation.json is already written",
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
        validate_codec_options(args)
        match args.action:
            case "setup":
                execute(remote(f"mkdir -p {REMOTE_ROOT}"))
                if args.family == "codec_tts":
                    sync_codec_tts()
                    execute(remote(setup_codec_script()))
                else:
                    sync()
                    execute(remote(setup_script()))
            case "stop":
                execute(remote(stop_script(require(args.run, "--run", "stop"))))
            case "sync":
                sync_family(args.family)
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


def job(args: argparse.Namespace) -> Job:  # noqa: PLR0912
    """What an action that starts a detached job runs."""
    if args.family == "codec_tts":
        return codec_tts_job(args)
    if args.action == "encbench":
        run = args.run or "encbench"
        models = args.models or ["masinissa-73m", "masinissa-31m", "xlm-roberta-base"]
        if len(models) > 1:
            cmd = [
                "__raw__",
                PYTHON,
                "-u",
                "tools/run_parallel_encbench.py",
                "--models",
                *models,
                "--workers",
                str(getattr(args, "workers", None) or 3),
                "--out-dir",
                f"{RUNS}/{run}",
            ]
        else:
            cmd = [
                "__raw__",
                PYTHON,
                "-u",
                "tools/run_encbench.py",
                "--models",
                *models,
                "--out",
                f"{RUNS}/{run}/encbench_results.json",
            ]
        return run, cmd, f"{run}/log.txt", []
    if args.family == "restoration":
        return restoration_job(args)
    if args.family == "encoder":
        return encoder_job(args)
    if args.family == "dihya":
        return dihya_job(args)
    match args.action:
        case "train":
            run = require(args.run, "--run", "train")
            return run, train_arguments(args), f"{run}/log.txt", []
        case "smoke":
            run = require(args.run, "--run", "smoke")
            part = require(args.part, "--part", "smoke")
            arguments = ["smoke", "--part", part, "--run", run]
            if args.model:
                arguments += ["--model", args.model]
            return f"{run}/{part}", arguments, f"{run}/{part}/log.txt", []
        case "data":
            task = require(args.task, "--task", "data")
            if task not in DATA_TASKS:
                message = f"unsupported llm data task={task!r}"
                raise LaunchError(message)
            if args.resume and task not in RESUMABLE_TASKS:
                message = f"--resume is for {' and '.join(RESUMABLE_TASKS)}, not {task}"
                raise LaunchError(message)
            flags = ["--resume"] if args.resume else []
            if args.model:
                flags += ["--model", args.model]
            return f"data-{task}", [task, *flags], f"data-{task}/log.txt", []
        case "merge":
            run = require(args.run, "--run", "merge")
            return run, merge_arguments(args), f"{run}/log.txt", []
        case "assemble":
            run = require(args.run, "--run", "assemble")
            return run, assemble_arguments(args), f"{run}/log.txt", []
        case _:
            return evaluation_job(args)


def validate_codec_options(args: argparse.Namespace) -> None:
    if args.family != "codec_tts":
        return
    if args.run is not None and not is_run_name(args.run):
        message = f"invalid codec TTS run name={args.run!r}"
        raise LaunchError(message)
    if args.steps is not None and args.action != "smoke":
        message = "codec TTS STEPS is only valid for smoke; use STOP for a bounded train invocation"
        raise LaunchError(message)
    if args.part is not None and (args.action != "smoke" or args.part not in SMOKE_PARTS):
        message = f"invalid codec TTS smoke part={args.part!r}"
        raise LaunchError(message)
    for name in (
        "model",
        "objective",
        "alpha",
        "template",
        "tokens",
        "lr",
        "batch_tokens",
        "micro_tokens",
        "micro_windows",
    ):
        if getattr(args, name) is not None:
            message = (
                f"codec_tts config owns {name}; use --config rather than unrelated launcher options"
            )
            raise LaunchError(message)
    if args.workers is not None:
        message = "codec_tts worker count belongs in --config"
        raise LaunchError(message)


def codec_tts_job(args: argparse.Namespace) -> Job:
    run = require(args.run, "--run", args.action)
    validate_codec_options(args)
    if args.action == "data":
        action = args.task or "prepare"
        if action not in {"inventory", "prepare", "partition", "encode", "language", "fetch"}:
            message = f"unsupported codec_tts data task={action!r}"
            raise LaunchError(message)
    elif args.action in {"train", "smoke", "evaluate"}:
        action = args.action
    else:
        message = f"unsupported codec_tts action={args.action!r}"
        raise LaunchError(message)
    arguments = ["__codec_tts__", action, "--run", run]
    for flag, value in (
        ("--config", args.config),
        ("--part", args.part),
        ("--stop-after", args.stop),
        ("--steps", args.steps),
    ):
        if value is not None:
            arguments += [flag, str(value)]
    if args.action == "evaluate":
        suite = "validation" if args.suite == "exit" else args.suite
        if suite not in {"validation", "release", "baseline"}:
            message = f"unsupported codec_tts evaluation suite={suite!r}"
            raise LaunchError(message)
        arguments += ["--suite", suite]
    return run, arguments, f"{run}/log.txt", []


def encoder_job(args: argparse.Namespace) -> Job:
    run = args.run or ("encoder-data" if args.action == "data" else None)
    run = require(run, "--run", args.action)
    action = "prepare" if args.action == "data" else args.action
    if action not in {"prepare", "train", "smoke", "evaluate"}:
        message = f"encoder does not support action {action!r}"
        raise LaunchError(message)
    arguments = ["__encoder__", action, "--run", run]
    options = (
        ("--part", args.part),
        ("--tokens", args.tokens),
        ("--lr", args.lr),
        ("--batch-tokens", args.batch_tokens),
        ("--micro-tokens", args.micro_tokens),
        ("--workers", args.workers),
        ("--config", args.config),
        ("--stop-after", args.stop),
        ("--steps", args.steps),
    )
    for flag, value in options:
        if value is not None:
            arguments += [flag, str(value)]
    if action == "evaluate":
        arguments += ["--suite", "dev" if args.suite == "exit" else args.suite]
        if args.model:
            arguments += ["--model", args.model]
    return run, arguments, f"{run}/log.txt", []


def restoration_job(args: argparse.Namespace) -> Job:
    run = require(
        args.run or ("restoration-data" if args.action == "data" else None), "--run", args.action
    )
    action = "prepare" if args.action == "data" else args.action
    if action not in {"prepare", "train", "smoke", "evaluate"}:
        message = f"restoration does not support action {action!r}"
        raise LaunchError(message)
    arguments = ["__restoration__", action, "--run", run]
    for flag, value in (
        ("--part", args.part),
        ("--config", args.config),
        ("--workers", args.workers),
        ("--micro-windows", args.micro_windows),
        ("--stop-after", args.stop),
        ("--model", args.model),
    ):
        if value is not None:
            arguments += [flag, str(value)]
    if action == "evaluate":
        arguments += ["--suite", "dev" if args.suite == "exit" else args.suite]
    return run, arguments, f"{run}/log.txt", []


def dihya_job(args: argparse.Namespace) -> Job:
    run = require(
        args.run or ("dihya-data" if args.action == "data" else None), "--run", args.action
    )
    action = "prepare" if args.action == "data" else args.action
    if action not in {"prepare", "train", "smoke", "evaluate", "calibrate"}:
        message = f"dihya does not support action {action!r}"
        raise LaunchError(message)
    arguments = ["__dihya__", action, "--run", run]
    if args.part is not None:
        arguments += ["--part", args.part]
    if args.stop is not None:
        arguments += ["--stop-after", str(args.stop)]
    return run, arguments, f"{run}/log.txt", []


SCREEN_FILE: Final = "screen.json"
"""Translation alone, ranking merge candidates before the full selection suite. Its own file,
so a later full run never overwrites it and `make llm TASK=select` never reads it."""


GUARD_TASKS: Final = ("ifeval", "gsm8k")
"""The no-think chat tasks merge-a100 failed (2026-10-17), scored alone before a full run.
They land in the selection report, so the full run keeps them."""


def evaluation_job(args: argparse.Namespace) -> Job:
    """The completion-mode suite, then for `select` and `release` the chat-mode suite in each
    thinking mode, each mode its own process because one vLLM engine is one process."""
    run = require(args.run, "--run", "evaluate")
    if args.chat_only and args.suite not in ("select", "release"):
        message = f"--chat-only needs --suite select or release, not {args.suite}"
        raise LaunchError(message)
    if args.suite == "screen":
        model = require(args.model, "--model", "evaluate --suite screen")
        arguments = ["evaluate", "--out", f"{RUNS}/{run}/{SCREEN_FILE}", "--parts", "mt"]
        return run, [*arguments, "--model", model], f"{run}/log.txt", []
    if args.suite == "kabyle":
        model = require(args.model, "--model", "evaluate --suite kabyle")
        arguments = [
            "evaluate",
            "--out",
            f"{RUNS}/{run}/evaluation-kabyle.json",
            "--parts",
            "heldout",
            "mt",
            "sib",
            "--model",
            model,
        ]
        return run, arguments, f"{run}/log.txt", []
    if args.suite == "guard":
        model = require(args.model, "--model", "evaluate --suite guard")
        guard = chat_module(
            [
                *("--model", model, "--suite", "select", "--mode", "no_think"),
                *("--out", f"{RUNS}/{run}/chat-no_think.json", "--tasks", *GUARD_TASKS),
            ]
        )
        return run, [], f"{run}/log.txt", [guard]
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
    return run, [] if args.chat_only else arguments, f"{run}/log.txt", chained


if __name__ == "__main__":
    raise SystemExit(main())
