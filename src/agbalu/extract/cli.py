"""Corpus extraction CLI.

python -m agbalu.extract.cli build
python -m agbalu.extract.cli build --out data/processed/text/agbalu-text-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

from agbalu.extract.pipeline import CorpusBuilder, summary
from agbalu.registry.loader import RegistryError, load_registry

DEFAULT_REGISTRY: Final = Path("resources/corpus_registry.yaml")
DEFAULT_ROOT: Final = Path("data/raw")
DEFAULT_OUT: Final = Path("data/processed/text/agbalu-text-v1.jsonl")

log: Final = logging.getLogger("agbalu.extract")


def command_build(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    builder = CorpusBuilder(registry, args.root)
    stats = builder.build(args.out)

    report = args.out.with_name(args.out.stem + ".stats.json")
    report.write_text(
        json.dumps(
            {"summary": summary(stats), "sources": [asdict(s) for s in stats if s.read]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    totals = summary(stats)
    print(f"{totals['kept']:,} sentences from {totals['sources']} sources -> {args.out}")
    print(
        f"read {totals['read']:,} | duplicate {totals['duplicate']:,} | "
        f"repaired {totals['repaired']:,} | short {totals['too_short']:,} | "
        f"long {totals['too_long']:,} | non-prose {totals['not_prose']:,}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-extract", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="extract, normalise and deduplicate the text corpus")
    build.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return command_build(args)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
