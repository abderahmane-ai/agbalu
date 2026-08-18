"""Seeded Unigram initialization from the lexicon.

Land & Pinter (arXiv:2512.12641) find initialization dominates Unigram training and the
EM refinement contributes little. §11.4 found the default initialization never
represents the annexed state: `axxam` and `wexxam` are memorised whole at every size
from 4k to 48k, so the stem is never shared.

`seed_sentencepieces_file` replaces SentencePiece's own seed extraction rather than
extending it. Measured: a model seeded with 11 pieces cannot exceed 28 total where the
default reaches 34 on the same corpus. The pool built here is therefore complete —
corpus substrings scored the way SentencePiece scores its own, plus the lexical
material — not just the additions under test.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.lexicon.state import ANNEXED_TO_FREE
from agbalu.tokenizer.spec import (
    MAX_PIECE_LENGTH,
    METASPACE,
    MIN_STEM,
    TokenizerError,
    required_chars,
)

DEFAULT_LEXICON: Final = Path("data/processed/lexicon/agbalu-lexicon-v1.jsonl")

MIN_TYPE_COUNT: Final = 2
"""A substring attested only inside a single hapax type is not evidence of a piece."""

DEFAULT_SEED_SIZE: Final = 1_000_000
"""SentencePiece's own `seed_sentencepiece_size` default, kept so the seeded and default
runs start from pools of the same order."""

TOP_CLITICS: Final = 200
"""§11.2 measured the clitic class as closed: the top 50 post-hyphen pieces are 60.5% of
all occurrences. 200 covers the tail without admitting hapax noise."""

LEXICON_FLOOR: Final = 1
"""Frequency credited to a lexical piece the corpus never shows as a substring. It still
enters the pool — inclusion is the intervention; inflating its score would not be."""

FREE_PREFIXES: Final[tuple[str, ...]] = ("ta", "ti", "a", "i", "u")
"""Free-state noun prefixes, longest first, so `ta-` is stripped before `a-`."""


@dataclass(frozen=True, slots=True)
class SeedPool:
    pieces: tuple[tuple[str, float], ...]
    from_corpus: int
    from_lexicon: int
    lexicon_only: int

    def __len__(self) -> int:
        return len(self.pieces)


def substring_counts(
    freq: Counter[str],
    *,
    min_type_count: int = MIN_TYPE_COUNT,
    max_length: int = MAX_PIECE_LENGTH,
) -> Counter[str]:
    """Frequency-weighted substrings of the word types, over metaspaced text.

    The metaspace prefix is part of the string SentencePiece sees, so `▁axxam` and
    `xxam` are different candidates and both belong in the pool.
    """
    counts: Counter[str] = Counter()
    for word, occurrences in freq.items():
        if occurrences < min_type_count:
            continue
        marked = METASPACE + word
        length = len(marked)
        for start in range(length):
            for end in range(start + 1, min(start + max_length + 1, length + 1)):
                counts[marked[start:end]] += occurrences
    return counts


def state_pieces() -> set[str]:
    """The word-initial alternation, as pieces rather than as whole forms.

    Every annexed prefix and every free prefix it maps back to. With the stems from
    `lexicon_pieces` in the same pool, `▁we` + `xxam` and `▁a` + `xxam` become available
    as a factorisation — which is the whole point of §11.5.
    """
    pieces = {METASPACE + annexed for annexed, _ in ANNEXED_TO_FREE}
    for _, frees in ANNEXED_TO_FREE:
        pieces.update(METASPACE + free for free in frees)
    return pieces


def lexicon_pieces(lexicon: Path) -> set[str]:
    """Single-word forms as whole pieces, and their stems as non-initial pieces."""
    if not lexicon.is_file():
        msg = f"lexicon not found: {lexicon}; run `make lexicon`"
        raise TokenizerError(msg)
    pieces: set[str] = set()
    with lexicon.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            form = str(json.loads(line)["form"])
            if not form or " " in form or len(form) > MAX_PIECE_LENGTH - 1:
                continue
            pieces.add(METASPACE + form)
            for prefix in FREE_PREFIXES:
                if form.startswith(prefix):
                    stem = form[len(prefix) :]
                    if len(stem) >= MIN_STEM and len(stem) <= MAX_PIECE_LENGTH:
                        pieces.add(stem)
                    break
    return pieces


def clitic_pieces(freq: Counter[str], top: int = TOP_CLITICS) -> set[str]:
    """The post-hyphen inventory, measured rather than listed.

    §11.2 established the class is small and closed; taking it from the corpus keeps it
    that way without hard-coding a list that the next corpus build would invalidate.
    """
    tail: Counter[str] = Counter()
    for word, occurrences in freq.items():
        if "-" not in word:
            continue
        for piece in word.split("-")[1:]:
            if piece and len(piece) <= MAX_PIECE_LENGTH:
                tail[piece] += occurrences
    return {piece for piece, _ in tail.most_common(top)}


def build_pool(
    freq: Counter[str],
    lexicon: Path,
    *,
    seed_size: int = DEFAULT_SEED_SIZE,
) -> SeedPool:
    """Corpus substrings capped at `seed_size`, then the lexical material forced in.

    The cap is applied before the lexical union so that seeding can only ever *add*
    candidates. A run where the lexical pieces were silently truncated away would look
    exactly like a run where they made no difference.
    """
    if seed_size < 1:
        msg = f"seed_size {seed_size} must be positive"
        raise TokenizerError(msg)

    counts = substring_counts(freq)
    ranked = sorted(counts.items(), key=lambda item: (-item[1] * len(item[0]), item[0]))
    kept: dict[str, int] = dict(ranked[:seed_size])
    from_corpus = len(kept)

    lexical = state_pieces() | lexicon_pieces(lexicon) | clitic_pieces(freq)
    lexical.update(required_chars())
    lexicon_only = 0
    for piece in sorted(lexical):
        if piece in kept:
            continue
        kept[piece] = counts.get(piece, LEXICON_FLOOR)
        lexicon_only += 1

    pieces = tuple(
        (piece, float(count * len(piece)))
        for piece, count in sorted(
            kept.items(), key=lambda item: (-item[1] * len(item[0]), item[0])
        )
    )
    return SeedPool(
        pieces=pieces,
        from_corpus=from_corpus,
        from_lexicon=len(lexical),
        lexicon_only=lexicon_only,
    )


def write_seed_file(pool: SeedPool, dest: Path) -> None:
    """`piece<TAB>score`, the format `seed_sentencepieces_file` reads.

    Scores are SentencePiece's own seeding gain, frequency times length, so a seeded run
    and a default run start on the same scale and only the membership differs.
    """
    if not pool.pieces:
        msg = "refusing to write an empty seed pool"
        raise TokenizerError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_suffix(dest.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as out:
        for piece, score in pool.pieces:
            out.write(f"{piece}\t{score:.6f}\n")
    staging.replace(dest)
