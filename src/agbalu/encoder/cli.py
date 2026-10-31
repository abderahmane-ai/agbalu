"""Prepare, verify, train, and evaluate the Kabyle encoder."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from agbalu.encoder import checkpoint, evaluation, smoke
from agbalu.encoder.configuration import Geometry, Locations, Recipe
from agbalu.encoder.corpus import prepare, read_json, verify_data, write_json
from agbalu.encoder.model import Encoder
from agbalu.encoder.training import Trainer
from agbalu.llm.progress import progress


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("prepare", "inspect", "train", "smoke", "evaluate"))
    result.add_argument("--run", default="encoder-pretrain")
    result.add_argument(
        "--part",
        choices=("throughput", "parity", "resume", "evaluate", "failure"),
        default="throughput",
    )
    result.add_argument("--suite", choices=("dev", "baseline"), default="dev")
    result.add_argument("--model", type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--tokens", type=int)
    result.add_argument("--steps", type=int)
    result.add_argument("--stop-after", type=int)
    result.add_argument("--lr", type=float)
    result.add_argument("--batch-tokens", type=int)
    result.add_argument("--micro-tokens", type=int)
    result.add_argument("--workers", type=int)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--corpus", type=Path)
    result.add_argument("--tokenizer", type=Path)
    result.add_argument("--data", type=Path)
    result.add_argument("--treebank", type=Path)
    return result


def configuration(args: argparse.Namespace) -> tuple[Recipe, Locations]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.run):
        message = f"invalid run identifier {args.run!r}"
        raise ValueError(message)
    source = read_json(args.config) if args.config else Recipe().model_dump()
    for attribute, name in (
        ("tokens", "total_tokens"),
        ("lr", "peak_lr"),
        ("batch_tokens", "batch_tokens"),
        ("micro_tokens", "micro_tokens"),
        ("workers", "workers"),
    ):
        value = getattr(args, attribute)
        if value is not None:
            source[name] = value
    recipe = Recipe.model_validate(source)
    if args.steps is not None:
        recipe = Recipe.model_validate(
            {**recipe.model_dump(), "total_tokens": args.steps * recipe.batch_tokens}
        )
    paths = Locations().model_dump()
    for name in ("corpus", "tokenizer", "data", "treebank"):
        value = getattr(args, name)
        if value is not None:
            paths[name] = value
    return recipe, Locations.model_validate(paths)


def evaluate(
    args: argparse.Namespace, locations: Locations, device: torch.device
) -> dict[str, object]:
    if args.suite == "baseline":
        result = evaluation.baseline(locations, device)
        write_json(locations.data / "baseline.stats.json", result)
        return result
    if args.model is None:
        message = "evaluate --suite dev requires --model pointing to a trained run"
        raise ValueError(message)
    pointer = read_json(args.model / "best_deployment.json")
    snapshot = args.model / str(pointer["snapshot"])
    checkpoint.verify_snapshot(snapshot)
    geometry = Geometry.model_validate(read_json(snapshot / "geometry.json"))
    model = Encoder(geometry.model_copy(update={"backend": "reference"})).to(device)
    payload = torch.load(snapshot / "training.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["weights"], strict=True)
    return evaluation.downstream(model, locations, device)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    recipe, locations = configuration(args)
    if args.action == "inspect":
        print(
            json.dumps(
                {
                    "parameters": recipe.geometry.parameters,
                    "recipe": recipe.model_dump(),
                    "locations": locations.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return 0
    if args.action == "prepare":
        result = prepare(locations)
        progress("end", family="encoder", **result)
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        message = "CUDA is unavailable; run the encoder compute commands on the school HPC"
        raise RuntimeError(message)
    if args.action == "smoke":
        verify_data(locations.data) if args.part in {"throughput", "evaluate"} else None
        result = smoke.run(args.part, recipe, locations, args.run, device)
    elif args.action == "evaluate":
        result = evaluate(args, locations, device)
        write_json(locations.runs / args.run / "evaluation.stats.json", result)
    else:
        result = Trainer(recipe, locations, args.run, device).run(stop_after=args.stop_after)
    print(json.dumps({"event": "result", "family": "encoder", **result}, allow_nan=False))
    if str(result.get("ended", "")).startswith("signal "):
        return 128 + int(str(result["ended"]).split()[1])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        progress("stage_failed", stage="encoder command", seconds=None, error=str(error))
        raise
