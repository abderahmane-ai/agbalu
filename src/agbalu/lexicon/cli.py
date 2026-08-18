"""Lexicon CLI.

python -m agbalu.lexicon.cli build
python -m agbalu.lexicon.cli validate
python -m agbalu.lexicon.cli coverage
python -m agbalu.lexicon.cli analyse "yeţwaṛfaɛ"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from agbalu.lexicon.analyser import Analyser
from agbalu.lexicon.coverage import DEFAULT_CORPUS, scan, totals, unknown_types
from agbalu.lexicon.models import LexiconError
from agbalu.lexicon.pipeline import DEFAULT_OUT, LexiconBuilder, summary
from agbalu.lexicon.validate import score, treebank_tokens
from agbalu.normalise import Normaliser
from agbalu.registry.loader import load_registry
from agbalu.treebank import DEFAULT_ROOT as DEFAULT_TREEBANK
from agbalu.treebank import TreebankError

DEFAULT_RAW: Final = Path("data/raw")
DEFAULT_REGISTRY: Final = Path("resources/corpus_registry.yaml")
DEFAULT_REPORTS: Final = Path("data/processed/lexicon")

TOP_CONFUSIONS: Final = 12
TOP_UNKNOWN: Final = 40

log: Final = logging.getLogger("agbalu.lexicon")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_build(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    builder = LexiconBuilder(registry=registry, root=args.root)
    stats = builder.build(args.out)
    totals = summary(stats, args.out)

    stats_path = args.out.with_suffix(".stats.json")
    _write_json(stats_path, totals)

    print(f"{totals['entries']:,} entries from {totals['sources']} sources -> {args.out}")
    print(f"  distinct forms     {totals['distinct_forms']:,}")
    print(f"  distinct lemmas    {totals['distinct_lemmas']:,}")
    print(f"  glossed            {totals['glossed_entries']:,}")
    print(f"  multiword          {totals['multiword_forms']:,}")
    print(f"  merged duplicates  {totals['merged']:,}")
    print(f"  repaired           {totals['repaired']:,}")
    print(f"  by upos            {totals['by_upos']}")
    print(f"  redistribution     {totals['by_redistribution']}")
    print(f"\n  stats -> {stats_path}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    analyser = Analyser.from_lexicon(args.lexicon)
    tokens = treebank_tokens(args.treebank)
    if not tokens:
        msg = f"no treebank tokens under {args.treebank}"
        raise LexiconError(msg)
    report = score(analyser, tokens)

    payload: dict[str, object] = {
        "normaliser_version": Normaliser().version,
        "lexicon": str(args.lexicon),
        "treebank": str(args.treebank),
        "tokens": report.tokens,
        "scored": report.scored,
        "known": report.known,
        "coverage": round(report.coverage, 6),
        "unambiguous": report.unambiguous,
        "correct": report.correct,
        "upos_accuracy": round(report.accuracy, 6),
        "lemma_agreement": round(report.lemma_agreement, 6),
        "by_gold": dict(report.by_gold.most_common()),
        "recall_by_gold": {
            tag: round(report.correct_by_gold[tag] / n, 4)
            for tag, n in report.by_gold.most_common()
            if n
        },
        "confusions": {f"{g}->{p}": n for (g, p), n in report.confusions.most_common()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lexicon-vs-treebank.json"
    _write_json(path, payload)

    print(f"gold treebank: {report.tokens:,} tokens, {report.scored:,} scored")
    print(f"  known to the lexicon  {report.known:,} ({report.coverage:.1%})")
    print(f"  unambiguous upos      {report.unambiguous:,}, agreement {report.accuracy:.1%}")
    print(f"  gold lemma recovered  {report.lemma_agreement:.1%}")
    print("  worst confusions (gold -> predicted):")
    for (gold, predicted), count in report.confusions.most_common(TOP_CONFUSIONS):
        print(f"    {count:>5,}  {gold:6} -> {predicted}")
    print(f"\n  report -> {path}")
    return 0


def command_coverage(args: argparse.Namespace) -> int:
    analyser = Analyser.from_lexicon(args.lexicon)
    per_source = scan(args.corpus, analyser, limit=args.limit)
    if not per_source:
        msg = f"no corpus lines read from {args.corpus}"
        raise LexiconError(msg)
    combined = totals(per_source)
    ranked = sorted(per_source.values(), key=lambda r: r.token_coverage)

    payload: dict[str, object] = {
        "normaliser_version": Normaliser().version,
        "lexicon": str(args.lexicon),
        "corpus": str(args.corpus),
        "lines_per_source_limit": args.limit,
        "lines": combined.lines,
        "tokens": combined.tokens,
        "types": len(combined.types),
        "token_coverage": round(combined.token_coverage, 6),
        "type_coverage": round(combined.type_coverage, 6),
        "routes": dict(combined.routes.most_common()),
        "by_source": {
            r.source: {
                "lines": r.lines,
                "tokens": r.tokens,
                "types": len(r.types),
                "token_coverage": round(r.token_coverage, 6),
                "type_coverage": round(r.type_coverage, 6),
            }
            for r in sorted(per_source.values(), key=lambda r: -r.token_coverage)
        },
    }
    if args.unknown:
        payload["frequent_unknown"] = unknown_types(
            args.corpus, analyser, top=TOP_UNKNOWN, limit=args.limit
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lexicon-coverage.json"
    _write_json(path, payload)

    print(
        f"{combined.tokens:,} tokens / {len(combined.types):,} types over {combined.lines:,} lines"
    )
    print(f"  token coverage     {combined.token_coverage:.1%}")
    print(f"  type coverage      {combined.type_coverage:.1%}")
    print(f"  routes             {dict(combined.routes.most_common())}")
    print("  lowest-coverage sources:")
    for report in ranked[:TOP_CONFUSIONS]:
        print(
            f"    {report.token_coverage:6.1%} tok / {report.type_coverage:6.1%} typ  "
            f"{report.source}"
        )
    print(f"\n  report -> {path}")
    return 0


def command_analyse(args: argparse.Namespace) -> int:
    analyser = Analyser.from_lexicon(args.lexicon)
    for form in args.forms:
        analyses = analyser.analyse(form)
        if not analyses:
            print(f"{form}: no analysis")
            continue
        print(f"{form}: {len(analyses)} reading(s), upos={analyser.upos(form)}")
        for analysis in analyses:
            print(
                f"  {analysis.route:9} lemma={analysis.lemma} upos={analysis.upos} "
                f"{analysis.feats()}  [{analysis.source}]"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-lexicon", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="merge every lexical source into one lexicon")
    build.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build.add_argument("--root", type=Path, default=DEFAULT_RAW)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)

    validate = sub.add_parser("validate", help="score the lexicon against the gold treebank")
    validate.add_argument("--lexicon", type=Path, default=DEFAULT_OUT)
    validate.add_argument("--treebank", type=Path, default=DEFAULT_TREEBANK)
    validate.add_argument("--out", type=Path, default=DEFAULT_REPORTS)

    coverage = sub.add_parser("coverage", help="lexicon coverage of the corpus, per source")
    coverage.add_argument("--lexicon", type=Path, default=DEFAULT_OUT)
    coverage.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    coverage.add_argument("--out", type=Path, default=DEFAULT_REPORTS)
    coverage.add_argument("--limit", type=int, default=None, help="lines per source")
    coverage.add_argument("--unknown", action="store_true", help="list frequent unknown types")

    analyse = sub.add_parser("analyse", help="show every reading of a form")
    analyse.add_argument("forms", nargs="+")
    analyse.add_argument("--lexicon", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return command_build(args)
        if args.command == "validate":
            return command_validate(args)
        if args.command == "coverage":
            return command_coverage(args)
        if args.command == "analyse":
            return command_analyse(args)
    except (LexiconError, TreebankError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
