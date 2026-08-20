"""Command-line interface for orthography standardisation, published as `agbalu/Boulifa-48M`.

Training runs on Modal (`make modal-boulifa`). `standardise` rewrites text; `evaluate` scores
a checkpoint on the held-out split beside the baseline that changes nothing, without which a
character accuracy in the high nineties says little — leaving the input untouched already
scores 89.70.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from agbalu.ocr.evaluate import compute_cer, compute_exact_match
from agbalu.standardise.corpus import load_jsonl
from agbalu.standardise.infer import DEFAULT_CHECKPOINT, Standardiser

DEFAULT_TEST = Path("data/kabstandard/test.jsonl")
DEFAULT_RESULT = Path("data/processed/bench/boulifa-v1-test.json")
EVAL_SEED = 4711


def handle_standardise(args: argparse.Namespace) -> None:
    engine = Standardiser.load(args.checkpoint)
    if args.text:
        print(engine.standardise(args.text))
        return
    for line in sys.stdin:
        stripped = line.strip()
        if stripped:
            print(engine.standardise(stripped))


def handle_evaluate(args: argparse.Namespace) -> None:
    """Score a checkpoint free-running, beside the copy-the-input floor.

    Free-running is the point. The trainer's `correct / tokens` is computed from a decode
    over the gold prefix — it ranks checkpoints, and it is not what a caller gets.
    """
    pairs = load_jsonl(args.split)
    drawn = random.Random(EVAL_SEED).sample(pairs, min(args.limit, len(pairs)))
    sources = [pair.source for pair in drawn]
    targets = [pair.target for pair in drawn]

    engine = Standardiser.load(args.checkpoint)
    hypotheses = engine.standardise_batch(sources, batch_size=args.batch_size)

    model_cer = compute_cer(hypotheses, targets)
    floor_cer = compute_cer(sources, targets)
    exact = compute_exact_match(hypotheses, targets)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "pairs": len(drawn),
        "seed": EVAL_SEED,
        "decoding": "greedy, free-running",
        "character_error_rate": model_cer,
        "character_accuracy": 1.0 - model_cer,
        "exact_match": exact,
        # The input, unmodified. Any system is only worth the distance between these two.
        "copy_input_character_error_rate": floor_cer,
        "copy_input_exact_match": compute_exact_match(sources, targets),
        "samples": [
            {"source": source, "hypothesis": hypothesis, "target": target}
            for source, hypothesis, target in list(zip(sources, hypotheses, targets, strict=True))[
                :3
            ]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"{len(drawn)} pairs: character accuracy {100 * (1 - model_cer):.2f}% "
        f"(CER {100 * model_cer:.2f}%), exact match {100 * exact:.2f}%"
    )
    print(f"copy-the-input floor: CER {100 * floor_cer:.2f}% -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kabyle orthography standardisation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite = subparsers.add_parser("standardise", help="rewrite text, or stdin")
    rewrite.add_argument("text", nargs="?", help="text to standardise; reads stdin if omitted")
    rewrite.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)

    evaluate = subparsers.add_parser("evaluate", help="score a checkpoint on the held-out split")
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--split", type=Path, default=DEFAULT_TEST)
    evaluate.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    evaluate.add_argument("--limit", type=int, default=1000, help="pairs to draw")
    evaluate.add_argument("--batch-size", type=int, default=32)

    return parser


HANDLERS = {"standardise": handle_standardise, "evaluate": handle_evaluate}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        HANDLERS[args.command](args)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
