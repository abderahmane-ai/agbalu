"""Phase 11 measurements that need no GPU.

python3 -m agbalu.llm.cli fertility
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from agbalu.llm import holdout
from agbalu.llm.bases import BASE
from agbalu.llm.corpus import HOLDOUT_RATE, CorpusError, Source
from agbalu.llm.fertility import Encoder, FertilityError, report, sample
from agbalu.llm.mixture import MixtureError, build

FERTILITY_SEED: Final = 20260810
BASE_MODEL: Final = BASE

CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
VOCABULARY: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
OUTPUT: Final = Path("data/processed/llm/fertility.json")
SAMPLE: Final = 3000

MIXTURE_OUT: Final = Path("data/processed/llm/cpt.jsonl")
MIXTURE_STATS: Final = Path("data/processed/llm/cpt.stats.json")
EPOCHS: Final = 4
"""Four passes is where repetition stops being free (Muennighoff et al., JMLR 2025)."""

SOURCES: Final = (
    Source(name="agbalu-text-v1", path=CORPUS, kind="kabyle", fields=("text",)),
    Source(
        name="mt-train",
        path=Path("data/processed/mt/train.jsonl"),
        kind="aligned",
        fields=("source", "target"),
        # Four directions in one file: `source` is Kabyle on the `kab-*` rows and English
        # or French on the others, so which side is which is a property of the row.
        direction_field="direction",
    ),
    Source(
        name="parallel-kab-fra",
        path=Path("data/interim/parallel/agbalu-parallel-v1.kab-fra.jsonl"),
        kind="aligned",
        fields=("kab", "fra"),
        code="fra_Latn",
    ),
)

HOLDOUT_DIR: Final = Path("data/processed/llm")
HOLDOUT_STATS: Final = Path("data/processed/llm/heldout.stats.json")


class HubEncoder:
    """A `transformers` tokenizer, without its special tokens."""

    def __init__(self, repo: str) -> None:
        from transformers import AutoTokenizer

        self._repo = repo
        self._tokenizer = AutoTokenizer.from_pretrained(repo)

    @property
    def name(self) -> str:
        return self._repo

    @property
    def unknown_id(self) -> int | None:
        unknown = self._tokenizer.unk_token_id
        return int(unknown) if isinstance(unknown, int) else None

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self._tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [[int(i) for i in ids] for ids in encoded]


class SentencePieceEncoder:
    """One of the `Mammeri-Tok` vocabularies."""

    def __init__(self, path: Path) -> None:
        import sentencepiece as spm

        if not path.is_file():
            message = f"vocabulary not found: {path}"
            raise FertilityError(message)
        self._path = path
        self._processor = spm.SentencePieceProcessor()
        self._processor.load(str(path))

    @property
    def name(self) -> str:
        return self._path.stem

    @property
    def unknown_id(self) -> int | None:
        return int(self._processor.unk_id())

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [[int(i) for i in self._processor.encode(text)] for text in texts]


def command_fertility(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        return 1

    sentences = sample(args.corpus, args.sample, seed=args.seed)
    ours = SentencePieceEncoder(args.vocabulary)
    encoders: list[Encoder] = [HubEncoder(args.base), ours]
    measured = report(encoders, sentences, reference=ours.name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base": args.base,
        "corpus": str(args.corpus),
        "seed": args.seed,
        **measured.as_dict(),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{measured.sentences:,} sentences, seed {args.seed}")
    for row in measured.vocabularies:
        ratio = row.ratio_to_reference
        suffix = "" if ratio is None else f"  {ratio}x reference"
        print(
            f"  {row.fertility.name:26} {row.fertility.tokens_per_word:6.3f} tokens/word  "
            f"UNK {100 * row.fertility.unknown_share:.4f}%{suffix}"
        )
    print(f"-> {args.out}")
    return 0


class HubCounter:
    """Token counting for the mixture, by the base model's own tokenizer."""

    def __init__(self, repo: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(repo)

    def count(self, texts: Sequence[str]) -> list[int]:
        encoded = self._tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]


def command_mixture(args: argparse.Namespace) -> int:
    result = build(SOURCES, HubCounter(args.base), args.out, epochs=args.epochs, rate=args.rate)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps({"base": args.base, **result.as_dict()}, indent=2) + "\n", encoding="utf-8"
    )
    for tally in result.tallies:
        print(
            f"  {tally.name:22} {tally.kind:8} {tally.documents:>10,} docs {tally.tokens:>13,} tok"
            f"  {tally.held_out:>6,} held out"
        )
    print(f"  {'per epoch':22} {'':8} {'':>10} {result.total_tokens:>13,} tok")
    print(
        f"  {'x' + str(args.epochs) + ' epochs':22} {'':8} {'':>10} "
        f"{result.total_tokens * args.epochs:>13,} tok"
    )
    print(f"  aligned share {result.aligned_share:.1%}")
    print(f"-> {args.out}  {args.stats}")
    return 0


def command_holdout(args: argparse.Namespace) -> int:
    """Write the evaluation sets the CPT corpus withholds, and what they are drawn from."""
    result = holdout.build(SOURCES, args.out, rate=args.rate, cap=args.cap)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    for tally in result.tallies:
        print(
            f"  {tally.source:22} {tally.language:9} {tally.documents:>6,} docs "
            f"{tally.characters:>10,} chars"
        )
    print(f"  1 in {result.rate}, capped at {result.cap} per source and language")
    print(f"-> {' '.join(str(p) for p in result.paths)}  {args.stats}")
    return 0


def token_budget(stats: Path, given: int | None) -> int:
    """The budget the plan is costed against, read from the built corpus by default.

    A constant here would be a second copy of a number the mixture already owns, and the
    two would part company the first time the corpus is rebuilt.
    """
    if given is not None:
        return given
    if not stats.is_file():
        message = f"{stats} does not exist; run `make llm-mixture`, or pass --tokens"
        raise MixtureError(message)
    payload = json.loads(stats.read_text(encoding="utf-8"))
    total = payload.get("tokens_total")
    if not isinstance(total, int) or total < 1:
        message = f"{stats} carries no usable tokens_total"
        raise MixtureError(message)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-llm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mixture = sub.add_parser("mixture", help="build the CPT corpus: Kabyle plus aligned pairs")
    mixture.add_argument("--base", default=BASE_MODEL)
    mixture.add_argument("--out", type=Path, default=MIXTURE_OUT)
    mixture.add_argument("--stats", type=Path, default=MIXTURE_STATS)
    mixture.add_argument("--epochs", type=int, default=EPOCHS)
    mixture.add_argument("--rate", type=int, default=HOLDOUT_RATE)

    held = sub.add_parser("holdout", help="write the evaluation sets the mixture withholds")
    held.add_argument("--out", type=Path, default=HOLDOUT_DIR)
    held.add_argument("--stats", type=Path, default=HOLDOUT_STATS)
    held.add_argument("--rate", type=int, default=HOLDOUT_RATE)
    held.add_argument("--cap", type=int, default=holdout.CAP)

    fertility = sub.add_parser("fertility", help="tokens per Kabyle word, the base against ours")
    fertility.add_argument("--base", default=BASE)
    fertility.add_argument("--corpus", type=Path, default=CORPUS)
    fertility.add_argument("--vocabulary", type=Path, default=VOCABULARY)
    fertility.add_argument("--sample", type=int, default=SAMPLE)
    fertility.add_argument("--seed", type=int, default=FERTILITY_SEED)
    fertility.add_argument("--out", type=Path, default=OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.command:
            case "fertility":
                return command_fertility(args)
            case "mixture":
                return command_mixture(args)
            case "holdout":
                return command_holdout(args)
    except (CorpusError, FertilityError, MixtureError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
