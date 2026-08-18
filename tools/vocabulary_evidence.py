"""What AƔBALU-Text v1 itself says about tokenizer vocabulary design.

Reproduces every number in `docs/tokenizer_design.md` §11. Four measurements:

1. **types** — is the type inventory real agglutination or a dirty corpus? Heaps' beta
   and the character inventory, split by script.
2. **hyphen** — Kabyle writes clitic clusters on it, so whether it is a boundary or
   part of the token decides a fifth of the vocabulary.
3. **fertility** — how badly adaptable tokenizers fragment Kabyle, against XLM-R.
4. **sweep** — does vocabulary pressure factor the annexed state? It does not (§11.4).

Usage:
    python tools/vocabulary_evidence.py --report types
    python tools/vocabulary_evidence.py --report hyphen
    python tools/vocabulary_evidence.py --report fertility
    python tools/vocabulary_evidence.py --report sweep --work-dir /tmp/tok

`sweep` trains SentencePiece Unigram models and needs the `models` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from agbalu.tokenizer.corpus import DEFAULT_CORPUS, read_texts, sample_sentences
from agbalu.tokenizer.evaluate import CLITIC_HOSTS, CLITICS, STATE_PAIRS
from agbalu.tokenizer.spec import MIN_STEM

DEFAULT_LEXICON: Final = Path("data/processed/lexicon/agbalu-lexicon-v1.jsonl")

KABYLE_LETTERS: Final = "abcčdḍefgǧɣhḥijklmnpqrstṭuvwxyzẓṛṣɛţ"
"""The classification inventory for the §11 corpus statistics, and deliberately *not*
`agbalu.tokenizer.spec.required_chars()`, which is wider. Guaranteeing a character its
own vocabulary slot costs one slot; counting it as Kabyle would inflate the measured
Kabyle share of a 3.09M-type inventory."""
KABYLE_ALPHABET: Final[frozenset[str]] = frozenset(
    set(KABYLE_LETTERS) | {c.upper() for c in KABYLE_LETTERS}
)

COVERAGE_MARKS: Final[tuple[int, ...]] = (
    1_000,
    2_000,
    4_000,
    8_000,
    12_000,
    16_000,
    24_000,
    32_000,
    48_000,
    64_000,
)

SWEEP_SIZES: Final[tuple[int, ...]] = (4_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000)

FOREIGN_LATIN_SHARE: Final = 0.30
"""Above this share of out-of-inventory letters a Latin token is not Kabyle."""


def word_frequencies(corpus: Path) -> Counter[str]:
    freq: Counter[str] = Counter()
    for text in read_texts(corpus):
        freq.update(text.split())
    return freq


def classify(word: str) -> str:
    """Which inventory a type belongs to. Kabyle types are the ones worth counting."""
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return "no-letters"
    if any(c.isdigit() for c in word):
        return "contains-digit"
    scripts: Counter[str] = Counter()
    for char in letters:
        try:
            scripts[unicodedata.name(char).split()[0]] += 1
        except ValueError:
            scripts["UNNAMED"] += 1
    dominant = scripts.most_common(1)[0][0]
    if dominant != "LATIN":
        return f"script:{dominant.lower()}"
    foreign = sum(1 for c in letters if c not in KABYLE_ALPHABET)
    if foreign == 0:
        return "kabyle-alphabet"
    return (
        "mostly-foreign-latin"
        if foreign / len(letters) > FOREIGN_LATIN_SHARE
        else ("some-foreign-latin")
    )


def coverage_curve(freq: Counter[str]) -> Iterator[tuple[int, float]]:
    total = sum(freq.values())
    if not total:
        return
    cumulative = 0
    marks = iter(COVERAGE_MARKS)
    mark = next(marks, None)
    for rank, (_, count) in enumerate(freq.most_common(), start=1):
        cumulative += count
        if mark is not None and rank == mark:
            yield mark, cumulative / total
            mark = next(marks, None)


def report_types(corpus: Path, lexicon: Path) -> None:
    freq = word_frequencies(corpus)
    types = len(freq)
    tokens = sum(freq.values())
    hapax = sum(1 for n in freq.values() if n == 1)

    print(f"types      {types:,}")
    print(f"tokens     {tokens:,}")
    print(f"hapax      {hapax:,} ({hapax / types:.1%} of types)")

    by_class: Counter[str] = Counter()
    class_tokens: Counter[str] = Counter()
    for word, count in freq.items():
        kind = classify(word)
        by_class[kind] += 1
        class_tokens[kind] += count
    print("\ninventory by class:")
    for kind, n in by_class.most_common(8):
        print(
            f"  {kind:22} {n:>10,} types {n / types:>6.1%}  "
            f"{class_tokens[kind]:>12,} tokens {class_tokens[kind] / tokens:>6.2%}"
        )

    kabyle = Counter({w: n for w, n in freq.items() if classify(w) == "kabyle-alphabet"})
    print(f"\npure Kabyle: {len(kabyle):,} types / {sum(kabyle.values()):,} tokens")
    print("token coverage by whole-word vocabulary size:")
    for mark, share in coverage_curve(kabyle):
        print(f"  top {mark:>6,} -> {share:6.2%}")

    if lexicon.is_file():
        forms = {
            str(json.loads(line)["form"]) for line in lexicon.open(encoding="utf-8") if line.strip()
        }
        known = sum(n for w, n in freq.items() if w in forms)
        print(f"\ncorpus tokens whose exact form is in the lexicon: {known / tokens:.1%}")


def report_hyphen(corpus: Path) -> None:
    """Whether the hyphen should be a boundary. It should — see §11.2."""
    freq = Counter(
        {w: n for w, n in word_frequencies(corpus).items() if classify(w) == "kabyle-alphabet"}
    )
    hyphenated = {w: n for w, n in freq.items() if "-" in w}
    total = sum(freq.values())

    print(f"pure-Kabyle types        {len(freq):,}")
    print(f"  containing a hyphen    {len(hyphenated):,} ({len(hyphenated) / len(freq):.1%})")
    print(f"  their token share      {sum(hyphenated.values()) / total:.2%}")
    hapax = sum(1 for n in hyphenated.values() if n == 1)
    print(f"  hyphenated hapax       {hapax:,} ({hapax / max(len(hyphenated), 1):.1%})")

    split: Counter[str] = Counter()
    for word, count in freq.items():
        for piece in word.split("-"):
            if piece:
                split[piece] += count

    print("\nwhole words:")
    for mark, share in coverage_curve(freq):
        print(f"  top {mark:>6,} -> {share:6.2%}")
    print("hyphen-split:")
    for mark, share in coverage_curve(split):
        print(f"  top {mark:>6,} -> {share:6.2%}")
    print(
        f"\ntype inventory {len(freq):,} -> {len(split):,} ({1 - len(split) / len(freq):.1%} fewer)"
    )

    tail: Counter[str] = Counter()
    for word, count in hyphenated.items():
        for piece in word.split("-")[1:]:
            if piece:
                tail[piece] += count
    top = sum(n for _, n in tail.most_common(50))
    print(f"clitic pieces: {len(tail):,} distinct; top 50 cover {top / sum(tail.values()):.1%}")
    print(f"  {[w for w, _ in tail.most_common(20)]}")


def report_fertility(corpus: Path, models: list[str], sample: int) -> None:
    """Fertility of tokenizers anyone would realistically adapt or compare against."""
    from transformers import AutoTokenizer

    sentences = sample_sentences(corpus, sample)
    words = sum(len(s.split()) for s in sentences)
    print(f"sample {len(sentences):,} sentences / {words:,} words\n")
    for path in models:
        if not Path(path).is_dir():
            print(f"  {path:46} not on disk")
            continue
        tok = AutoTokenizer.from_pretrained(path)
        total = 0
        unk = 0
        for text in sentences:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            total += len(ids)
            if tok.unk_token_id is not None:
                unk += sum(1 for i in ids if i == tok.unk_token_id)
        print(
            f"  {Path(path).name:46} vocab={tok.vocab_size:>7,} "
            f"fertility={total / words:5.3f} unk={unk:,}"
        )


def report_sweep(corpus: Path, work: Path, sample: int) -> None:
    """Train the Unigram sweep and score it on Kabyle-specific criteria."""
    import sentencepiece as spm

    work.mkdir(parents=True, exist_ok=True)
    plain = work / "corpus.txt"
    if not plain.is_file():
        with plain.open("w", encoding="utf-8") as out:
            for text in read_texts(corpus):
                flat = text.replace("\n", " ").strip()
                if flat:
                    out.write(flat + "\n")

    sentences = sample_sentences(corpus, sample)
    words = sum(len(s.split()) for s in sentences)
    required = "".join(sorted(KABYLE_ALPHABET | set("-'")))

    print(
        f"{'vocab':>7} {'fertility':>10} {'whole-word':>11} {'state-share':>12} "
        f"{'clitic-split':>13}"
    )
    for size in SWEEP_SIZES:
        prefix = work / f"kab-uni-{size}"
        model = prefix.with_suffix(".model")
        if not model.is_file():
            spm.SentencePieceTrainer.train(
                input=str(plain),
                model_prefix=str(prefix),
                vocab_size=size,
                model_type="unigram",
                character_coverage=0.9995,
                byte_fallback=True,
                required_chars=required,
                split_digits=True,
                # The corpus is already normaliser output; SentencePiece's own NMT
                # normalisation would re-fold characters we deliberately preserve.
                normalization_rule_name="identity",
                input_sentence_size=2_000_000,
                shuffle_input_sentence=True,
                num_threads=8,
            )
        sp = spm.SentencePieceProcessor()
        sp.load(str(model))

        total = sum(len(sp.encode(s, out_type=int)) for s in sentences)
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        whole = sum(1 for p in pieces if p.startswith("▁"))

        shared = 0
        for free, annexed in STATE_PAIRS:
            a = {p.lstrip("▁") for p in sp.encode(free, out_type=str)}
            b = {p.lstrip("▁") for p in sp.encode(annexed, out_type=str)}
            if {p for p in a if len(p) >= MIN_STEM} & {p for p in b if len(p) >= MIN_STEM}:
                shared += 1

        atomic = sum(
            1
            for host in CLITIC_HOSTS
            for clitic in CLITICS
            if any(p.lstrip("▁") == clitic for p in sp.encode(f"{host}-{clitic}", out_type=str))
        )
        trials = len(CLITIC_HOSTS) * len(CLITICS)
        print(
            f"{size:>7,} {total / words:>10.3f} {whole / len(pieces):>10.1%} "
            f"{shared:>7}/{len(STATE_PAIRS):<4} {atomic:>8}/{trials:<4}"
        )

    print("\nstate-share  = free/annexed pairs sharing a >=3-char stem piece.")
    print("clitic-split = host-clitic pairs where the clitic survives as one piece.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocabulary-evidence", description=__doc__)
    parser.add_argument(
        "--report", choices=("types", "hyphen", "fertility", "sweep"), default="types"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/tokenizer-sweep"))
    parser.add_argument("--sample", type=int, default=30_000)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "data/raw/hf.boffire.kabyle-pos-v2",
            "data/raw/hf.boffire.bpe-tokenizer-for-kabyle",
            "data/raw/hf.boffire.kabyle-bpe-tokenizer-v2",
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.corpus.is_file():
        print(f"corpus not found: {args.corpus}; run `make extract`", file=sys.stderr)
        return 1
    if args.report == "types":
        report_types(args.corpus, args.lexicon)
    elif args.report == "hyphen":
        report_hyphen(args.corpus)
    elif args.report == "fertility":
        report_fertility(args.corpus, args.models, args.sample)
    else:
        report_sweep(args.corpus, args.work_dir, args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
