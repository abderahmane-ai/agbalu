"""Tokenizer CLI.

python -m agbalu.tokenizer.cli prepare
python -m agbalu.tokenizer.cli build --vocab-size 16000 --seeded
python -m agbalu.tokenizer.cli sweep
python -m agbalu.tokenizer.cli evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from agbalu.normalise import Normaliser
from agbalu.tokenizer.corpus import DEFAULT_CORPUS, sample_sentences, word_frequencies, write_plain
from agbalu.tokenizer.evaluate import evaluate
from agbalu.tokenizer.seed import (
    DEFAULT_LEXICON,
    DEFAULT_SEED_SIZE,
    build_pool,
    write_seed_file,
)
from agbalu.tokenizer.spec import (
    DEFAULT_VOCAB_SIZE,
    SWEEP_SIZES,
    TOKENIZER_VERSION,
    TokenizerError,
    TokenizerSpec,
)
from agbalu.tokenizer.train import DEFAULT_OUT_DIR, train

log: Final = logging.getLogger("agbalu.tokenizer")

DEFAULT_REPORTS: Final = Path("data/processed/tokenizer")
EVAL_SAMPLE: Final = 30_000

PLAIN_NAME: Final = "corpus.txt"
SEED_NAME: Final = "seed-pool.tsv"


def _plain(out_dir: Path) -> Path:
    return out_dir / PLAIN_NAME


def _seed(out_dir: Path) -> Path:
    return out_dir / SEED_NAME


def _require_prepared(out_dir: Path, *, seeded: bool) -> tuple[Path, Path | None]:
    plain = _plain(out_dir)
    if not plain.is_file():
        msg = f"{plain} missing; run `make tokenizer-prepare`"
        raise TokenizerError(msg)
    if not seeded:
        return plain, None
    seed = _seed(out_dir)
    if not seed.is_file():
        msg = f"{seed} missing; run `make tokenizer-prepare`"
        raise TokenizerError(msg)
    return plain, seed


def command_prepare(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plain = _plain(args.out_dir)
    sentences = write_plain(args.corpus, plain)
    print(f"{sentences:,} sentences -> {plain}")

    freq = word_frequencies(args.corpus)
    pool = build_pool(freq, args.lexicon, seed_size=args.seed_size)
    seed = _seed(args.out_dir)
    write_seed_file(pool, seed)
    print(f"{len(pool):,} seed pieces -> {seed}")
    print(f"  from corpus substrings   {pool.from_corpus:,}")
    print(f"  lexical pieces offered   {pool.from_lexicon:,}")
    print(f"  added by seeding alone   {pool.lexicon_only:,}")
    return 0


def command_build(args: argparse.Namespace) -> int:
    plain, seed = _require_prepared(args.out_dir, seeded=args.seeded)
    spec = TokenizerSpec(vocab_size=args.vocab_size, seed_file=seed)
    result = train(spec, plain, args.out_dir)
    print(f"{result.spec.name}: {result.pieces:,} pieces -> {result.model}")
    return 0


def command_sweep(args: argparse.Namespace) -> int:
    arms: list[bool] = (
        [False] if args.arm == "base" else ([True] if args.arm == "seeded" else [False, True])
    )
    plain, seed = _require_prepared(args.out_dir, seeded=any(arms))
    for seeded in arms:
        for size in args.sizes:
            spec = TokenizerSpec(vocab_size=size, seed_file=seed if seeded else None)
            result = train(spec, plain, args.out_dir)
            print(f"{result.spec.name}: {result.pieces:,} pieces")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    models = sorted(args.out_dir.glob("*.model"))
    if not models:
        msg = f"no tokenizer models under {args.out_dir}; run `make tokenizer-sweep`"
        raise TokenizerError(msg)

    sentences = sample_sentences(args.corpus, args.sample)
    results = [evaluate(model, sentences) for model in models]

    header = (
        f"{'model':>26} {'vocab':>7} {'fert':>6} {'tok/ch':>7} {'byte%':>7} "
        f"{'whole':>7} {'state':>7} {'clitic':>8} {'emb@384':>9}"
    )
    print(header)
    for item in results:
        print(
            f"{item.name:>26} {item.vocab_size:>7,} {item.fertility:>6.3f} "
            f"{item.tokens_per_char:>7.3f} {item.byte_piece_rate:>6.3%} "
            f"{item.whole_word_share:>6.1%} {item.state_share:>3}/{item.state_trials:<3} "
            f"{item.clitic_atomic:>4}/{item.clitic_trials:<3} "
            f"{item.embedding_params[384] / 1e6:>8.2f}M"
        )

    failures = sum(item.roundtrip_failures for item in results)
    if failures:
        print(f"\nroundtrip failures: {failures}")

    args.reports.mkdir(parents=True, exist_ok=True)
    report = args.reports / "agbalu-tok-v1.eval.json"
    payload: dict[str, object] = {
        "tokenizer_version": TOKENIZER_VERSION,
        "normaliser_version": Normaliser().version,
        "corpus": str(args.corpus),
        "sample_sentences": len(sentences),
        "models": [item.to_dict() for item in results],
    }
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport -> {report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-tokenizer", description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="flat training text and the seed pool")
    prepare.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    prepare.add_argument("--seed-size", type=int, default=DEFAULT_SEED_SIZE)
    prepare.set_defaults(func=command_prepare)

    build = sub.add_parser("build", help="train one vocabulary")
    build.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    build.add_argument("--seeded", action="store_true")
    build.set_defaults(func=command_build)

    sweep = sub.add_parser("sweep", help="train the vocabulary-size grid")
    sweep.add_argument("--sizes", type=int, nargs="+", default=list(SWEEP_SIZES))
    sweep.add_argument("--arm", choices=("base", "seeded", "both"), default="both")
    sweep.set_defaults(func=command_sweep)

    evaluate_parser = sub.add_parser("evaluate", help="score every model in the output directory")
    evaluate_parser.add_argument("--sample", type=int, default=EVAL_SAMPLE)
    evaluate_parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    evaluate_parser.set_defaults(func=command_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except TokenizerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
