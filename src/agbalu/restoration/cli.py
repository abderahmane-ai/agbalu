"""Prepare, train and inspect the document restoration model."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from agbalu.encoder.corpus import read_json, write_json
from agbalu.lid.infer import Identifier
from agbalu.llm.progress import progress
from agbalu.restoration import smoke
from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.corpus import LanguageGate, prepare, verify
from agbalu.restoration.dataset import Windows
from agbalu.restoration.evaluation import evaluate
from agbalu.restoration.inference import Inference
from agbalu.restoration.training import Trainer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action", choices=("prepare", "train", "smoke", "evaluate", "restore", "inspect")
    )
    result.add_argument("--run", default="belaid")
    result.add_argument("--config", type=Path)
    result.add_argument(
        "--part",
        choices=("throughput", "parity", "resume", "failure", "pilot", "assessment", "selection"),
        default="throughput",
    )
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--stop-after", type=int)
    result.add_argument("--micro-windows", type=int)
    result.add_argument("--workers", type=int)
    result.add_argument("--data", type=Path)
    result.add_argument("--corpus", type=Path)
    result.add_argument("--model", type=Path)
    result.add_argument("--suite", choices=("dev", "release", "baseline"), default="dev")
    result.add_argument("--text")
    return result


def configuration(args: argparse.Namespace) -> tuple[Settings, Locations]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", args.run):
        msg = f"invalid run identifier {args.run!r}"
        raise ValueError(msg)
    payload = read_json(args.config) if args.config else {}
    for name in ("micro_windows", "workers"):
        value = getattr(args, name)
        if value is not None:
            payload[name] = value
    settings = Settings.model_validate(payload)
    paths = Locations().model_dump()
    for name in ("data", "corpus"):
        value = getattr(args, name)
        if value is not None:
            paths[name] = value
    return settings, Locations.model_validate(paths)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings, locations = configuration(args)
    if args.action == "inspect":
        print(
            json.dumps(
                {"settings": settings.model_dump(), "locations": locations.model_dump(mode="json")},
                indent=2,
            )
        )
        return 0
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA is unavailable; run restoration compute on the school HPC"
        raise RuntimeError(msg)
    if args.action == "prepare":
        gate = LanguageGate(Identifier.load(locations.identifier, device=device.type))
        result = prepare(locations, settings, gate)
    elif args.action == "train":
        result = Trainer(settings, locations, args.run, device).run(stop_after=args.stop_after)
    elif args.action == "smoke":
        result = smoke.run(settings, locations, args.run, args.part, device)
    else:
        inference = Inference.load(args.model or locations.runs / args.run, device)
        if args.action == "restore":
            text = args.text if args.text is not None else sys.stdin.read()
            print(inference.predict(text).text)
            return 0
        verify(locations.data)
        split = 2 if args.suite == "release" else 1
        result = evaluate(
            inference.model,
            Windows(locations.data, settings, split),
            device,
            baseline=args.suite == "baseline",
        )
        write_json(locations.runs / args.run / "evaluation.stats.json", result)
    progress("result", **{**result, "family": "restoration"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        progress("stage_failed", stage="restoration command", seconds=None, error=str(error))
        raise
