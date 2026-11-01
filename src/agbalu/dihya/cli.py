"""Command-line interface for Dihya-22M Pan-Amazigh LID.

Dispatches data preparation, HPC training, 20-step smoke verification,
and evaluation suites. Compatible with tools.launch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from pydantic import TypeAdapter

from agbalu.dihya.artifacts import write_json
from agbalu.dihya.config import DEFAULT_DATA_DIR, DEFAULT_RUNS_DIR, ModelConfig, TrainConfig
from agbalu.dihya.dataset import TextDataset
from agbalu.dihya.evaluate import (
    calibrate_conformal,
    calibration_scores,
    evaluate_dataset,
    partition_development,
)
from agbalu.dihya.model import Classifier
from agbalu.dihya.prepare import build_dihya_dataset
from agbalu.dihya.smoke import run_smoke
from agbalu.dihya.train import emit_event, train
from agbalu.llm.provenance import file_digest


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI commands."""
    parser = argparse.ArgumentParser(description="Dihya-22M Pan-Amazigh LID CLI")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # prepare
    prep_p = subparsers.add_parser("prepare", help="Prepare dataset from sources")
    prep_p.add_argument("--run", type=str, default="dihya-22m")
    prep_p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prep_p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    audit_p = subparsers.add_parser("audit", help="Independently audit dataset files")
    audit_p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    audit_p.add_argument("--out", type=Path, default=Path("artifacts/dihya/audit.stats.json"))

    # train
    train_p = subparsers.add_parser("train", help="Train Dihya-22M on HPC")
    train_p.add_argument("--run", type=str, required=True)
    train_p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_p.add_argument("--stop-after", type=int, default=None)
    train_p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # smoke
    smoke_p = subparsers.add_parser("smoke", help="Run 20-step smoke suite")
    smoke_p.add_argument("--run", type=str, default="dihya-smoke")
    smoke_p.add_argument(
        "--part",
        type=str,
        default="all",
        choices=("all", "throughput", "parity"),
    )
    smoke_p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate model checkpoint")
    eval_p.add_argument("--run", type=str, required=True)
    eval_p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    eval_p.add_argument("--split", choices=("dev", "test"), default="dev")
    eval_p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    eval_p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # calibrate
    cal_p = subparsers.add_parser("calibrate", help="Compute split-conformal quantile")
    cal_p.add_argument("--run", type=str, required=True)
    cal_p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    cal_p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    cal_p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser


def main(args: list[str] | None = None) -> int:
    """CLI execution entrypoint."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = build_parser()
    parsed = parser.parse_args(args)
    device = torch.device(getattr(parsed, "device", "cpu"))

    if parsed.action == "prepare":
        build_dihya_dataset(raw_dir=parsed.raw_dir, out_dir=parsed.data_dir)
        return 0

    if parsed.action == "audit":
        from agbalu.dihya.audit import audit_directory

        audit_report = audit_directory(parsed.data_dir)
        write_json(parsed.out, audit_report)
        emit_event(
            "audit",
            passed=audit_report["passed"],
            errors=audit_report["errors"],
            output=str(parsed.out),
        )
        return 0 if audit_report["passed"] else 1

    if parsed.action == "smoke":
        run_smoke(parsed.part, device, DEFAULT_RUNS_DIR / parsed.run)
        return 0

    if parsed.action == "train":
        train_ds = TextDataset.load(parsed.data_dir, "train")
        dev_ds = TextDataset.load(parsed.data_dir, "dev")
        dev_ds, _, _ = partition_development(dev_ds, TrainConfig().seed)

        m_cfg = ModelConfig()
        t_cfg = TrainConfig()
        run_dir = DEFAULT_RUNS_DIR / parsed.run

        train(
            model_config=m_cfg,
            train_config=t_cfg,
            train_dataset=train_ds,
            dev_dataset=dev_ds,
            run_dir=run_dir,
            device=device,
            stop_after_steps=parsed.stop_after,
        )
        return 0

    if parsed.action in ("evaluate", "calibrate"):
        run_dir = DEFAULT_RUNS_DIR / parsed.run
        ckpt_path = run_dir / f"{parsed.checkpoint}.pt"
        if not ckpt_path.exists():
            print(f"error: checkpoint not found at {ckpt_path}", file=sys.stderr)
            return 1

        metadata = json.loads(ckpt_path.with_suffix(".stats.json").read_text(encoding="utf-8"))
        if file_digest(ckpt_path) != metadata["sha256"]:
            msg = f"{ckpt_path}: SHA256 mismatch"
            raise ValueError(msg)
        m_cfg = TypeAdapter(ModelConfig).validate_python(metadata["model_config"])
        model = Classifier(m_cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        dev_ds = TextDataset.load(
            parsed.data_dir, parsed.split if parsed.action == "evaluate" else "dev"
        )

        if parsed.action == "evaluate":
            if parsed.split == "dev":
                dev_ds, _, _ = partition_development(dev_ds, TrainConfig().seed)
            report = evaluate_dataset(model, dev_ds, m_cfg, device)
            write_json(run_dir / f"{parsed.split}-report.stats.json", report)
            emit_event("eval", **report)
        else:
            _, fit, check = partition_development(dev_ds, TrainConfig().seed)
            q_hat = calibrate_conformal(model, fit, m_cfg, device)
            scores = calibration_scores(model, check, m_cfg, device)
            coverage = sum(score <= q_hat for score in scores) / len(scores)
            result = {
                "q_hat": q_hat,
                "coverage": coverage,
                "fit_rows": len(fit),
                "check_rows": len(check),
                "checkpoint_sha256": metadata["sha256"],
            }
            write_json(run_dir / "calibration.stats.json", result)
            emit_event("calibrate", **result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
