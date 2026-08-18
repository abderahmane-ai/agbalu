"""Normalisation CLI.

python -m agbalu.normalise.cli check      # idempotence + impact over the seed
python -m agbalu.normalise.cli evalset    # build the corrupted<->canonical set
python -m agbalu.normalise.cli text "..."  # normalise one string, show the diff
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Final

from agbalu.normalise.normaliser import Normaliser

csv.field_size_limit(10**9)

SEED_MONO: Final = Path("data/raw/tatoeba/tatoeba_kab_mono_2026-08-05.tsv")
SEED_PARALLEL: Final = Path("data/raw/tatoeba/tatoeba_kab_eng_2026-08-05.tsv")
DEFAULT_EVALSET: Final = Path("data/processed/normalisation/evalset.jsonl")
"""Derived data, so it stays out of git (CLAUDE.md 3.5). `make evalset` rebuilds it."""


def read_column(path: Path, *, encoding: str, columns: int, index: int) -> list[str]:
    """Read one column of a seed TSV.

    `QUOTE_NONE` is mandatory: the default silently drops 5,904 sentences
    (CLAUDE.md 2.1a).
    """
    with path.open(encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [row[index] for row in reader if len(row) == columns]


def seed_sentences() -> list[str]:
    sentences: list[str] = []
    if SEED_MONO.is_file():
        sentences += read_column(SEED_MONO, encoding="utf-8", columns=3, index=2)
    if SEED_PARALLEL.is_file():
        sentences += read_column(SEED_PARALLEL, encoding="utf-8-sig", columns=4, index=1)
    return sentences


def command_check(_: argparse.Namespace) -> int:
    normaliser = Normaliser()
    sentences = seed_sentences()
    if not sentences:
        print("no seed corpus found; nothing to check", file=sys.stderr)
        return 1

    changed = violations = 0
    kinds: collections.Counter[str] = collections.Counter()
    for sentence in sentences:
        once = normaliser.normalise(sentence)
        if once != sentence:
            changed += 1
            for change in normaliser.analyse(sentence).changes:
                kinds[change.kind] += 1
        if normaliser.normalise(once) != once:
            violations += 1

    total = len(sentences)
    print(f"normaliser {normaliser.version}")
    print(f"{total:,} sentences, {changed:,} changed ({100 * changed / total:.2f}%)")
    for kind, count in kinds.most_common():
        print(f"  {kind:20} {count:>9,}")
    print(f"idempotence violations: {violations}")
    if violations:
        print("EXIT CRITERION FAILED", file=sys.stderr)
        return 1
    print("exit criterion met: normalise(normalise(x)) == normalise(x)")
    return 0


def command_evalset(args: argparse.Namespace) -> int:
    """Emit every sentence the normaliser repairs, as a corrupted->canonical pair.

    These are real defects from real data, not synthesised noise, which is what
    makes the set a fair benchmark for any competing normaliser (task 2.7).
    """
    normaliser = Normaliser()
    written = 0
    kinds: collections.Counter[str] = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for sentence in seed_sentences():
            result = normaliser.analyse(sentence)
            if not result.changed:
                continue
            for change in result.changes:
                kinds[change.kind] += 1
            handle.write(
                json.dumps(
                    {
                        "corrupted": result.original,
                        "canonical": result.text,
                        "kinds": sorted({c.kind for c in result.changes}),
                        "flags": sorted({f.kind for f in result.flags}),
                        "version": result.version,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    print(f"wrote {written:,} pairs to {args.out}")
    for kind, count in kinds.most_common():
        print(f"  {kind:20} {count:>9,}")
    return 0


def command_text(args: argparse.Namespace) -> int:
    normaliser = Normaliser()
    result = normaliser.analyse(args.text)
    print(f"in  : {result.original}")
    print(f"out : {result.text}")
    for change in result.changes:
        print(f"  {change.kind:20} {change.before!r} -> {change.after!r}")
    for flag in result.flags:
        print(f"  FLAG {flag.kind:16} {flag.text!r}: {flag.detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-normalise", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="idempotence and impact over the seed corpus")

    evalset = sub.add_parser("evalset", help="build the corrupted<->canonical benchmark")
    evalset.add_argument("--out", type=Path, default=DEFAULT_EVALSET)

    text = sub.add_parser("text", help="normalise one string and explain every edit")
    text.add_argument("text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match args.command:
        case "check":
            return command_check(args)
        case "evalset":
            return command_evalset(args)
        case "text":
            return command_text(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
