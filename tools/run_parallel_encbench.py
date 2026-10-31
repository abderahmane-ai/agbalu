"""Parallel runner for KabEncBench across multiple models on HPC GPU.

Dispatches up to N concurrent worker processes, each running one encoder on cuda:0,
logging to separate log files, and synthesizing the final unified Bradley-Terry Elo
leaderboard once all runs complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agbalu.bench.elo import Leaderboard, compute_leaderboard

MODEL_ALIASES: dict[str, str] = {
    "masinissa-73m": "agbalu/Masinissa-73M",
    "masinissa-31m": "agbalu/Masinissa-31M",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "serengeti-e110": "UBC-NLP/serengeti-E110",
    "serengeti-e250": "UBC-NLP/serengeti-E250",
    "serengeti": "UBC-NLP/serengeti",
    "mmbert-small": "jhu-clsp/mmBERT-small",
    "mmbert-base": "jhu-clsp/mmBERT-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "afro-xlmr-base": "Davlan/afro-xlmr-base-76L_script",
    "rembert": "google/rembert",
}


def _model_slug(name: str) -> str:
    return name.replace("/", "_").replace("-", "_").lower()


def run_one_model(
    model_arg: str,
    python_bin: str,
    out_dir: Path,
    data_dir: Path,
) -> tuple[str, bool, float]:
    """Execute evaluation of one model and stream output to its dedicated log."""
    slug = _model_slug(model_arg)
    log_path = out_dir / f"log_{slug}.txt"
    json_path = out_dir / f"encbench_{slug}.json"

    # Resolve HF alias if applicable
    model_id = MODEL_ALIASES.get(model_arg, model_arg)

    # Check if already completed
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if "results" in payload:
                print(f"[Skip] {model_id} already evaluated in {json_path}", flush=True)
                return model_id, True, 0.0
        except Exception:  # noqa: S110
            pass

    cmd = [
        python_bin,
        "-u",
        "tools/run_encbench.py",
        "--models",
        model_arg,
        "--data-dir",
        str(data_dir),
        "--out",
        str(out_dir / "encbench_results.json"),
    ]

    t0 = time.time()
    print(f"[Start] Evaluating {model_id} (log: {log_path})", flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        ret = proc.wait()

    elapsed = time.time() - t0
    success = ret == 0
    status_str = "Completed" if success else f"Failed (exit {ret})"
    print(f"[{status_str}] {model_id} in {elapsed:.1f}s", flush=True)
    return model_id, success, elapsed


def build_unified_leaderboard(out_dir: Path) -> Leaderboard | None:
    """Scan out_dir for all encbench_*.json files and compute Bradley-Terry Elo."""
    model_task_items: dict[str, dict[str, Any]] = {}
    json_files = list(out_dir.glob("encbench_*.json"))
    # Exclude results / full leaderboard files
    candidate_files = [
        f
        for f in json_files
        if not f.name.endswith("encbench_results.json")
        and not f.name.endswith("encbench_full_leaderboard.json")
    ]

    for f in candidate_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            m_id = data.get("model_id", f.stem.replace("encbench_", ""))
            # If item_scores was saved:
            if "item_scores" in data:
                model_task_items[m_id] = data["item_scores"]
            elif "results" in data:
                # Synthesize 5 pseudo-seeds using accuracy / primary metric if raw items missing
                res = data["results"]
                m_items: dict[str, list[list[float]]] = {}
                for task in ("syntax", "morph", "ner", "sent", "topic", "punct"):
                    if task in res:
                        score = float(res[task].get("primary_metric", 0.5))
                        # 100 representative items matching task accuracy
                        n_items = 100
                        n_pos = round(score * n_items)
                        items = [1.0] * n_pos + [0.0] * (n_items - n_pos)
                        m_items[task] = [items for _ in range(5)]
                if m_items:
                    model_task_items[m_id] = m_items
        except Exception as err:
            print(f"Warning reading {f}: {err}", flush=True)

    if not model_task_items:
        print("No valid model results found to build leaderboard.", flush=True)
        return None

    anchor = (
        "FacebookAI/xlm-roberta-base"
        if "FacebookAI/xlm-roberta-base" in model_task_items
        else next(iter(model_task_items.keys()))
    )

    print(
        f"\nComputing Global Bradley-Terry Tournament across {len(model_task_items)} models "
        f"(Anchor: {anchor}, 2000 bootstraps)...",
        flush=True,
    )
    leaderboard = compute_leaderboard(
        "frozen",
        model_task_items,
        anchor_model=anchor,
        num_bootstraps=2000,
    )

    print("\n" + "=" * 80, flush=True)
    print("KabEncBench Unified Leaderboard (Track A: Frozen Scale-Normalized Probe)")
    print("=" * 80, flush=True)
    print(f"{'Rank':<5} {'Model':<35} {'Elo (95% CI)':<22} {'Tier':<6}", flush=True)
    print("-" * 80, flush=True)
    for r in leaderboard.ratings:
        elo_str = f"{r.elo_median:.1f} [{r.ci_low:.1f}, {r.ci_high:.1f}]"
        print(f"{r.rank:<5} {r.model:<35} {elo_str:<22} {r.tier:<6}", flush=True)
    print("=" * 80 + "\n", flush=True)

    out_payload = {
        "track": "frozen",
        "anchor_model": anchor,
        "num_models": len(leaderboard.ratings),
        "ratings": [asdict(r) for r in leaderboard.ratings],
    }
    (out_dir / "encbench_full_leaderboard.json").write_text(
        json.dumps(out_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved unified leaderboard to {out_dir / 'encbench_full_leaderboard.json'}", flush=True)
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel KabEncBench runner across models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "xlm-roberta-base",
            "serengeti-e110",
            "mmbert-small",
            "serengeti",
        ],
        help="List of model keys or HF identifiers to evaluate in parallel",
    )
    parser.add_argument("--workers", type=int, default=3, help="Concurrent worker count")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/runs/encbench"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/tasks/encbench"))
    parser.add_argument(
        "--python", type=str, default=sys.executable, help="Python interpreter path"
    )
    parser.add_argument(
        "--leaderboard-only",
        action="store_true",
        help="Only compute leaderboard from existing runs",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.leaderboard_only:
        build_unified_leaderboard(args.out_dir)
        return 0

    print(
        f"Parallel KabEncBench launching {len(args.models)} models with {args.workers} workers.",
        flush=True,
    )
    print(f"Logs and artifacts written to {args.out_dir}\n", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_model, m, args.python, args.out_dir, args.data_dir): m
            for m in args.models
        }
        for future in as_completed(futures):
            m = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[Error] {m} crashed with exception: {exc}", flush=True)

    build_unified_leaderboard(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
