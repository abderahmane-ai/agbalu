"""Script conversion with the trained model.

python3 -m agbalu.tifinagh.cli convert --text "ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?"
python3 -m agbalu.tifinagh.cli evaluate --split test --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from agbalu.bench.tifinagh import (
    BenchmarkError,
    read_split,
    score_conversion,
    score_schwa,
    to_latin,
)
from agbalu.tifinagh.infer import DEFAULT_CHECKPOINT, MAX_LENGTH, CheckpointError, Transliterator

log: Final = logging.getLogger("agbalu.tifinagh")

RESULTS: Final = Path("data/processed/bench/tifinagh.json")
BATCH: Final = 64


def command_convert(args: argparse.Namespace) -> int:
    engine = Transliterator.load(args.checkpoint, device=args.device)
    lines = args.file.read_text(encoding="utf-8").splitlines() if args.file else [args.text]
    for line in lines:
        if not line.strip():
            print()
            continue
        print(engine.transliterate(line, num_beams=args.beams, max_length=args.max_length))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Free-running greedy decoding over a split, scored against its Latin side.

    Greedy rather than beam because this is the number a caller gets by default, and
    because a beam that re-ranks 49,795 sentences is a GPU job. `--beams` on `convert` is
    where the beam lives.
    """
    pairs = read_split(args.split, limit=args.limit)
    engine = Transliterator.load(args.checkpoint, device=args.device)

    hypotheses: list[str] = []
    for start in range(0, len(pairs), BATCH):
        window = pairs[start : start + BATCH]
        hypotheses.extend(engine.greedy_batch([pair.tifinagh for pair in window], args.max_length))
        log.info("decoded %d/%d", len(hypotheses), len(pairs))

    references = [pair.latin.lower() for pair in pairs]
    baseline = [to_latin(pair.tifinagh) for pair in pairs]
    report = {
        "split": args.split,
        "sentences": len(pairs),
        "decoding": "greedy, free-running",
        "model": {
            "conversion": score_conversion(references, hypotheses).as_dict(),
            "schwa": score_schwa(references, hypotheses).as_dict(),
        },
        "character_table": {
            "conversion": score_conversion(references, baseline).as_dict(),
            "schwa": score_schwa(references, baseline).as_dict(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-tifinagh", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Tifinagh to Kabyle Latin, restoring the schwa")
    source = convert.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    convert.add_argument("--beams", type=int, default=4)

    evaluate = sub.add_parser("evaluate", help="score a split against the character table")
    evaluate.add_argument("--split", choices=("dev", "test"), default="test")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--out", type=Path, default=RESULTS)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        match args.command:
            case "convert":
                return command_convert(args)
            case "evaluate":
                return command_evaluate(args)
    except (BenchmarkError, CheckpointError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
