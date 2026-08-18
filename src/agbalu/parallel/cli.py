"""Parallel corpus CLI.

python -m agbalu.parallel.cli build
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Final

from agbalu.bench.flores import DEFAULT_ROOT as FLORES_ROOT
from agbalu.bench.flores import FloresError, read_all
from agbalu.parallel.nllb import STRATA, analyse
from agbalu.parallel.pipeline import ParallelBuilder, SourceStats, summary
from agbalu.registry.loader import RegistryError, load_registry

DEFAULT_REGISTRY: Final = Path("resources/corpus_registry.yaml")
DEFAULT_ROOT: Final = Path("data/raw")
DEFAULT_OUT: Final = Path("data/interim/parallel")

log: Final = logging.getLogger("agbalu.parallel")


def _plain(stats: SourceStats) -> dict[str, object]:
    """`dataclasses.asdict` cannot round-trip a `Counter` field.

    It rebuilds every dict subclass from an iterable of key-value tuples, and
    `Counter` reads that as a sequence of elements to count — turning
    `{"eng": 226}` into `{("eng", 226): 1}`, which is then not JSON-serialisable.
    """
    out: dict[str, object] = {}
    for item in fields(stats):
        value = getattr(stats, item.name)
        out[item.name] = dict(value) if isinstance(value, Counter) else value
    return out


def command_build(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    try:
        benchmark = list(read_all(args.flores))
    except FloresError:
        log.warning("flores absent at %s; building without decontamination", args.flores)
        benchmark = []

    builder = ParallelBuilder(registry, args.root, benchmark)
    stats = builder.build(args.out)
    totals = summary(stats)

    (args.out / "agbalu-parallel-v1.stats.json").write_text(
        json.dumps(
            {"summary": asdict(totals), "sources": [_plain(s) for s in stats if s.read]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{totals.kept:,} pairs from {totals.sources} sources -> {args.out}")
    print(f"  by language        {totals.by_language}")
    print(
        f"  read {totals.read:,} | duplicate {totals.duplicate:,} | "
        f"contaminated {totals.contaminated:,} | repaired {totals.repaired:,}"
    )
    print(
        f"  hard defects  {totals.hard_defective:,} ({totals.hard_defect_rate:.2%})"
        " — unusable pairs; a lower bound on the error rate"
    )
    print(
        f"  any defect    {totals.defective:,} ({totals.defect_rate:.2%})"
        " — includes soft inconsistencies that may be formatting"
    )
    for kind, count in totals.by_defect.items():
        print(f"    {count:>9,}  {kind}")
    return 0


def command_agreement(args: argparse.Namespace) -> int:
    boffire = sorted((args.root / "hf.boffire.nllb-en-kab").rglob("*.parquet"))
    imsidag = sorted((args.root / "hf.imsidag.nllb-en-kab").rglob("*.parquet"))
    pool = args.root / "opus.nllb-kab" / "en-kab.txt.zip"
    for path in (pool, *boffire, *imsidag):
        if not path.exists():
            print(f"error: missing {path}", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    report = analyse(pool, boffire, imsidag, sample_out=args.out / "nllb-en-kab-strata.jsonl")

    print(f"NLLB en-kab pool: {report.pool:,} pairs")
    print(f"  boffire kept    {report.boffire:,}")
    print(f"  imsidag kept    {report.imsidag:,}")
    print(
        f"  unmatched       boffire {report.boffire_unmatched:,} / "
        f"imsidag {report.imsidag_unmatched:,} ({report.matched_rate:.2%} traced to pool)"
    )
    print(f"  filters agree   {report.agreement_rate:.2%}")
    print(f"  contested band  {report.contested:,}")
    print(f"\n  {'stratum':18} {'pairs':>11} {'hard':>9} {'any':>9}")
    for name in STRATA:
        item = report.strata[name]
        print(f"  {name:18} {item.pairs:>11,} {item.hard_rate:>8.2%} {item.any_rate:>8.2%}")
    (args.out / "nllb-agreement.json").write_text(
        json.dumps(
            {
                "pool": report.pool,
                "boffire": report.boffire,
                "imsidag": report.imsidag,
                "agreement_rate": round(report.agreement_rate, 4),
                "boffire_unmatched": report.boffire_unmatched,
                "imsidag_unmatched": report.imsidag_unmatched,
                "matched_rate": round(report.matched_rate, 4),
                "contested": report.contested,
                "strata": {
                    n: {
                        "pairs": s.pairs,
                        "hard_defective": s.hard_defective,
                        "hard_rate": round(s.hard_rate, 4),
                        "any_defective": s.any_defective,
                        "any_rate": round(s.any_rate, 4),
                        "by_defect": dict(s.by_defect.most_common()),
                    }
                    for n, s in report.strata.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-parallel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="unify, clean and label every parallel source")
    build.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--flores", type=Path, default=FLORES_ROOT)

    agreement = sub.add_parser(
        "agreement", help="partition the mined NLLB pool by what two filters kept"
    )
    agreement.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    agreement.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return command_build(args)
        if args.command == "agreement":
            return command_agreement(args)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
