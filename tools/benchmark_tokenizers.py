"""Benchmark published Kabyle tokenizers on a real corpus sample.

Reproduces the measurements in docs/prior_art.md §3.2-3.4: fertility, UNK rate,
and the token cost of homoglyph corruption. Existing published benchmarks use ten
hand-picked sentences; this uses tens of thousands drawn from the corpus.

Usage:
    python tools/benchmark_tokenizers.py data/raw/tatoeba/tatoeba_kab_mono_2026-08-05.tsv

Requires network access on first run (downloads tokenizers from the Hub) and the
optional `tokenizers` package.
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# Greek/Cyrillic lookalikes for Kabyle Latin letters, with their repairs.
# See docs/prior_art.md; `ε` U+03B5 vs `ɛ` U+025B are distinct characters, so NFC
# does not merge them.
HOMOGLYPH_REPAIRS: Final[dict[str, str]] = {
    "ε": "ɛ",
    "Σ": "Ɛ",
    "Ԑ": "Ɛ",
    "γ": "ɣ",
    "Γ": "Ɣ",
    "ğ": "ǧ",
    "Ğ": "Ǧ",
    "ț": "ţ",
}

RAW: Final = Path("data/raw")

TOKENIZERS: Final[tuple[str, ...]] = (
    "boffire/bpe-tokenizer-for-kabyle",
    "boffire/Kabyle-BPE-Tokenizer-v2",
    "boffire/kabyle-gpt2-tokenizer",
    "facebook/nllb-200-distilled-600M",
)


class Result(NamedTuple):
    name: str
    vocab: int
    fertility: float
    tokens_per_sentence: float
    unk_rate: float
    corrupted_tokens: int
    repaired_tokens: int
    atomic_corrupt: tuple[str, ...]
    split_correct: tuple[str, ...]

    @property
    def waste(self) -> float:
        if self.repaired_tokens == 0:
            return 0.0
        return 100.0 * (self.corrupted_tokens - self.repaired_tokens) / self.repaired_tokens


def repair(text: str) -> str:
    for wrong, right in HOMOGLYPH_REPAIRS.items():
        text = text.replace(wrong, right)
    return text


def read_sentences(path: Path, column: int) -> list[str]:
    """Read one column of a TSV. Tolerates a BOM; skips malformed rows."""
    sentences: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) > column:
                sentences.append(fields[column])
    return sentences


def locate(repo: str, cache: Path) -> Path | None:
    """The checksummed copy under `data/raw` if we hold it, else the Hub.

    The Hub is not a durable source for this measurement: on 2026-08-08 every `boffire`
    tokenizer answered HTTP 401, having been public when the table was first measured.
    Returns None when the repo can be reached from neither, so one missing row does not
    cost the others.
    """
    local = RAW / f"hf.{repo.replace('/', '.').lower()}" / "tokenizer.json"
    if local.is_file():
        return local

    target = cache / f"{repo.replace('/', '__')}.json"
    if target.exists():
        return target

    url = f"https://huggingface.co/{repo}/resolve/main/tokenizer.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 — fixed https host
            target.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        print(f"{repo:<34} unavailable — HTTP {error.code}", file=sys.stderr)
        return None
    return target


def _is_one_piece(tokenizer: Tokenizer, char: str) -> bool:
    """Whether the letter survives as a single piece, tested by behaviour rather than by
    vocabulary membership — byte-level and SentencePiece vocabularies do not store the
    literal character, so looking it up would answer a different question."""
    tokens = [t for t in tokenizer.encode(char, add_special_tokens=False).tokens if t != "▁"]
    return len(tokens) == 1 and "<unk>" not in tokens


def evaluate(repo: str, path: Path, sample: list[str], corrupted: list[str]) -> Result:
    """Special tokens are excluded from every count. NLLB's post-processor puts `<unk>` in
    the source-language slot when no language is set, which reads as 7.404% UNK on text
    that contains none.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    words = sum(len(s.split()) for s in sample)
    encodings = tokenizer.encode_batch(sample, add_special_tokens=False)
    total = sum(len(e.ids) for e in encodings)

    unk_id = tokenizer.token_to_id("<unk>")
    unks = sum(e.ids.count(unk_id) for e in encodings) if unk_id is not None else 0

    as_is = sum(len(e.ids) for e in tokenizer.encode_batch(corrupted, add_special_tokens=False))
    repaired = [repair(s) for s in corrupted]
    fixed = sum(len(e.ids) for e in tokenizer.encode_batch(repaired, add_special_tokens=False))

    atomic = tuple(wrong for wrong in HOMOGLYPH_REPAIRS if _is_one_piece(tokenizer, wrong))
    correct = dict.fromkeys(HOMOGLYPH_REPAIRS.values())
    split = tuple(right for right in correct if not _is_one_piece(tokenizer, right))

    return Result(
        name=repo,
        vocab=tokenizer.get_vocab_size(),
        fertility=total / words if words else 0.0,
        tokens_per_sentence=total / len(sample) if sample else 0.0,
        unk_rate=100.0 * unks / total if total else 0.0,
        corrupted_tokens=as_is,
        repaired_tokens=fixed,
        atomic_corrupt=atomic,
        split_correct=split,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="TSV corpus to sample from")
    parser.add_argument("--column", type=int, default=2, help="0-indexed sentence column")
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=Path, default=Path(".tokenizer-cache"))
    args = parser.parse_args(argv)

    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("error: pip install tokenizers", file=sys.stderr)
        return 1

    sentences = read_sentences(args.corpus, args.column)
    if not sentences:
        print(f"error: no sentences read from {args.corpus}", file=sys.stderr)
        return 1

    corrupted = [s for s in sentences if any(c in s for c in HOMOGLYPH_REPAIRS)]
    random.seed(args.seed)
    sample = random.sample(sentences, min(args.sample, len(sentences)))

    print(
        f"corpus {len(sentences):,} sentences | sample {len(sample):,} "
        f"| corrupted {len(corrupted):,}\n"
    )
    header = (
        f"{'tokenizer':<34} {'vocab':>7} {'fertility':>10} {'tok/sent':>9} {'unk%':>7} {'waste':>8}"
    )
    print(header)
    results = []
    for repo in TOKENIZERS:
        path = locate(repo, args.cache)
        if path is None:
            continue
        r = evaluate(repo, path, sample, corrupted)
        results.append(r)
        print(
            f"{r.name:<34} {r.vocab:>7,} {r.fertility:>10.3f} "
            f"{r.tokens_per_sentence:>9.2f} {r.unk_rate:>7.3f} {r.waste:>7.1f}%"
        )
    print("\nwaste = extra tokens spent on corrupted text vs the same text repaired\n")

    print("letters that survive as a single piece:")
    for r in results:
        atomic = " ".join(r.atomic_corrupt) or "—"
        split = " ".join(r.split_correct) or "—"
        print(f"{r.name:<34} corrupt kept {atomic:<20} correct split {split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
