#!/usr/bin/env python3
"""AƔBALU: HPC Launcher and Status Monitor for Fadhma 2 Sovereign Kabyle ASR.

Unified interface matching Phase 11 / Jugurtha operational standards:
- sync: Push code to HPC pod without touching data or checkpoints
- data: Ingest Common Voice, TutlaytAI, and Tatoeba with Dihya filtering in tmux
- augment: Download MUSAN noise and BUT ReverbDB RIR assets in tmux
- smoke: Run mandatory 20-step smoke validation drill on H100 MIG slice
- train: Launch full 5-epoch training loop in detached tmux session
- status: Unified status reporting across all pipeline stages with live clip/step counters
- logs: Stream live logs with clean rendering and live line rewriting (like Jugurtha)
- stop: Send graceful SIGINT/Ctrl-C to trainer to save atomic checkpoint
- pull: Fetch checkpoints and evaluation reports to local artifacts
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.run_log import Printer, View, summary

HOST: Final = "hpc"
PROJECT: Final = "agbalu"
REMOTE_ROOT: Final = f"/home/jovyan/{PROJECT}"
ENVIRONMENT: Final = f"/home/jovyan/envs/{PROJECT}"
PYTHON: Final = f"{ENVIRONMENT}/bin/python"
RUN_DIR: Final = "artifacts/runs/fadhma-v2"

PREAMBLE: Final = (
    ". ~/hpc-env.sh"
    f" && export PYTHONUNBUFFERED=1 HF_HOME={REMOTE_ROOT}/artifacts/hf PYTHONPATH=src"
    f" && cd {REMOTE_ROOT}"
)

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

STAGE_CONFIGS: Final = {
    "data": {
        "session": "agbalu-speech-data",
        "log": f"{RUN_DIR}/data_build.log",
        "title": "Stage 1: Multi-Source Ingestion & Dihya Language Filter",
    },
    "augment": {
        "session": "agbalu-speech-augment",
        "log": f"{RUN_DIR}/augment_download.log",
        "title": "Stage 2: Multi-Condition Acoustic Assets (MUSAN & BUT ReverbDB)",
    },
    "smoke": {
        "session": "agbalu-speech-smoke",
        "log": f"{RUN_DIR}/smoke/smoke.log",
        "title": "Stage 3: 20-Step Smoke Validation Drill (H100 MIG 3g.47gb)",
    },
    "train": {
        "session": "agbalu-speech-train",
        "log": f"{RUN_DIR}/train.log",
        "title": "Stage 4: Sovereign Fadhma 2 Training (600M Conformer CTC)",
    },
}


def remote(command: str) -> list[str]:
    """Execute SSH command on HPC host after environment preamble."""
    return ["ssh", "-o", "ConnectTimeout=15", HOST, f"{PREAMBLE} && {command}"]


def code_sync() -> list[str]:
    excludes = [arg for pattern in EXCLUDES for arg in ("--exclude", pattern)]
    return ["rsync", "-az", "--delete", *excludes, "./", f"{HOST}:{REMOTE_ROOT}/"]


def execute(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        err_msg = result.stderr.strip()[:300]
        msg = f"Exit {result.returncode}: {shlex.join(command)[:300]}\n{err_msg}"
        raise RuntimeError(msg)
    return result


def run_sync() -> None:
    sys.stdout.write("--> Syncing codebase to HPC pod...\n")
    subprocess.run(code_sync(), check=True)
    sys.stdout.write("--> Sync complete.\n")


def detect_active_stage() -> str:
    """Detect which speech session is currently active on the pod."""
    res = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", HOST, "tmux ls 2>/dev/null || true"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = res.stdout.strip()
    for stage, cfg in STAGE_CONFIGS.items():
        if cfg["session"] in output:
            return stage
    return "train"


def start_session(session_name: str, script: str, log_file: str) -> None:
    run_sync()
    cmd = (
        f"tmux has-session -t {session_name} 2>/dev/null"
        f" && echo '[-] Session {session_name} is already running.'"
        f" || (mkdir -p {REMOTE_ROOT}/{Path(log_file).parent}"
        f" && tmux new-session -d -s {session_name} '{script} > {REMOTE_ROOT}/{log_file} 2>&1'"
        f" && echo '[+] Started session {session_name}. Log: {REMOTE_ROOT}/{log_file}')"
    )
    res = execute(remote(cmd))
    sys.stdout.write(res.stdout)


def show_status(target_stage: str | None = None, *, raw: bool = False) -> None:
    stage = target_stage or detect_active_stage()
    cfg = STAGE_CONFIGS.get(stage, STAGE_CONFIGS["train"])
    session = cfg["session"]
    log = cfg["log"]

    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write(f"  Fadhma 2 Status: {cfg['title']}\n")
    sys.stdout.write("=" * 70 + "\n")

    probe_script = (
        f"(tmux has-session -t {session} 2>/dev/null && echo 'session: running' || "
        f"echo 'session: not running');"
        f" echo '===SESSIONS===';"
        f" tmux ls 2>/dev/null || echo 'none';"
        f" echo '===DISK===';"
        f" df -h /home/jovyan | tail -n 1;"
        f" echo '===AUDIO===';"
        f" (find {REMOTE_ROOT}/data/raw/speech/audio -type f 2>/dev/null | wc -l || echo '0');"
        f" du -sh {REMOTE_ROOT}/data/raw/speech/audio 2>/dev/null || echo '0 MB';"
        f" echo '===CHECKPOINTS===';"
        f" ls -la {REMOTE_ROOT}/{RUN_DIR}/*.pt 2>/dev/null || echo 'none';"
        f" echo '===LOG===';"
        f" tail -n 50 {REMOTE_ROOT}/{log} 2>/dev/null || echo 'no log yet.'"
    )

    res = execute(remote(probe_script))
    output = res.stdout

    def _sec(tag: str) -> str:
        marker = f"==={tag}===\n"
        if marker not in output:
            return ""
        return output.split(marker, 1)[1].split("\n===", 1)[0].strip()

    status_header = output.split("===SESSIONS===")[0].strip()
    sessions_text = _sec("SESSIONS")
    disk_text = _sec("DISK")
    audio_text = _sec("AUDIO")
    chk_text = _sec("CHECKPOINTS")
    raw_log = _sec("LOG")

    sys.stdout.write(f"Session State : {status_header}\n")
    if sessions_text:
        active_list = [
            s.split(":")[0] for s in sessions_text.splitlines() if "agbalu-speech" in s
        ]
        sys.stdout.write(f"Active Speech Sessions: {', '.join(active_list) or 'none'}\n")

    if disk_text:
        sys.stdout.write(f"Disk Capacity : {disk_text}\n")

    if audio_text:
        lines = audio_text.splitlines()
        clip_cnt = lines[0] if lines else "0"
        disk_sz = lines[1] if len(lines) > 1 else ""
        try:
            cnt_val = f"{int(clip_cnt):,}"
        except ValueError:
            cnt_val = clip_cnt
        sys.stdout.write(f"Audio Footage : {cnt_val} clips extracted ({disk_sz})\n")

    if chk_text and chk_text != "none":
        chk_lines = [
            line.split()[-1] for line in chk_text.splitlines() if ".pt" in line
        ]
        sys.stdout.write(f"Checkpoints   : {', '.join(Path(p).name for p in chk_lines)}\n")

    sys.stdout.write("\n--- Log Tail (" + log + ") ---\n")
    if raw or not raw_log:
        sys.stdout.write(raw_log + "\n")
    else:
        rendered = summary(raw_log.splitlines(), keep=15)
        for r_line in rendered:
            sys.stdout.write("  " + r_line + "\n")
    sys.stdout.write("=" * 70 + "\n")


def write_now(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def stream_logs(target_stage: str | None = None, *, raw: bool = False) -> None:
    stage = target_stage or detect_active_stage()
    cfg = STAGE_CONFIGS.get(stage, STAGE_CONFIGS["train"])
    log = f"{REMOTE_ROOT}/{cfg['log']}"

    if raw:
        subprocess.run(["ssh", "-t", HOST, f"tail -n 100 -f {log}"], check=False)
        return

    view = View()
    printer = Printer(write_now, terminal=sys.stdout.isatty())
    process = subprocess.Popen(
        ["ssh", "-o", "ConnectTimeout=15", HOST, f"tail -n 100 -F {log}"],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        if process.stdout is not None:
            for line in process.stdout:
                rendered = view.render(line)
                printer.show(rendered)
    except KeyboardInterrupt:
        write_now("\n")
    finally:
        process.terminate()
        process.wait()


def stop_session(target_stage: str | None = None) -> None:
    stage = target_stage or detect_active_stage()
    cfg = STAGE_CONFIGS.get(stage, STAGE_CONFIGS["train"])
    session = cfg["session"]
    cmd = (
        f"tmux send-keys -t {session} C-c 2>/dev/null"
        f" && echo '[+] Sent SIGINT/Ctrl-C to {session}. Waiting for graceful termination.'"
        f" || echo '[-] No active session {session} found.'"
    )
    res = execute(remote(cmd))
    sys.stdout.write(res.stdout)


def pull_artifacts() -> None:
    dest = Path(RUN_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    sys.stdout.write(f"--> Pulling {RUN_DIR} from HPC to local...\n")
    subprocess.run(["rsync", "-az", f"{HOST}:{REMOTE_ROOT}/{RUN_DIR}/", f"{RUN_DIR}/"], check=True)
    sys.stdout.write("--> Pull complete.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("sync", "data", "augment", "smoke", "train", "status", "logs", "stop", "pull"),
    )
    parser.add_argument(
        "--task",
        choices=("data", "augment", "smoke", "train"),
        default=None,
        help="Pipeline task stage (default: auto-detected)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Custom tmux session name override",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw log text without formatting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = args.action

    if action == "sync":
        run_sync()
    elif action == "data":
        session = args.session or STAGE_CONFIGS["data"]["session"]
        script = f"{PYTHON} tools/build_speech_data.py --dihya"
        log = STAGE_CONFIGS["data"]["log"]
        start_session(session, script, log)
    elif action == "augment":
        session = args.session or STAGE_CONFIGS["augment"]["session"]
        script = f"{PYTHON} tools/acquire_augment_data.py"
        log = STAGE_CONFIGS["augment"]["log"]
        start_session(session, script, log)
    elif action == "smoke":
        session = args.session or STAGE_CONFIGS["smoke"]["session"]
        script = (
            f"{PYTHON} -m agbalu.speech.cli smoke"
            f" --train-manifest data/processed/speech/train.jsonl"
            f" --vocab artifacts/asr/vocab.json"
            f" --out-dir {RUN_DIR}/smoke"
            f" --device cuda:0"
        )
        log = STAGE_CONFIGS["smoke"]["log"]
        start_session(session, script, log)
    elif action == "train":
        session = args.session or STAGE_CONFIGS["train"]["session"]
        script = (
            f"{PYTHON} -m agbalu.speech.cli train"
            f" --train-manifest data/processed/speech/train.jsonl"
            f" --dev-manifest data/processed/speech/dev.jsonl"
            f" --vocab artifacts/asr/vocab.json"
            f" --output-dir {RUN_DIR}"
        )
        log = STAGE_CONFIGS["train"]["log"]
        start_session(session, script, log)
    elif action == "status":
        show_status(args.task, raw=args.raw)
    elif action == "logs":
        stream_logs(args.task, raw=args.raw)
    elif action == "stop":
        stop_session(args.task)
    elif action == "pull":
        pull_artifacts()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
