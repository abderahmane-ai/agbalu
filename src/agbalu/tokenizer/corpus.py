"""Corpus access for the tokenizer.

Three views of AƔBALU-Text v1: the flat text SentencePiece trains on, the
word-frequency table the seed pool is extracted from, and a reservoir sample for
evaluation. `tools/vocabulary_evidence.py` reads through here too, so the §11 numbers
and the built vocabulary cannot drift apart.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from agbalu.tokenizer.spec import RANDOM_SEED, TokenizerError

DEFAULT_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")


def read_texts(corpus: Path) -> Iterator[str]:
    """Sentences from a corpus JSONL, with the line number on any failure.

    A build killed mid-write leaves a truncated final line; without the position that
    surfaces as a bare `JSONDecodeError` from somewhere inside a 3M-line file.
    """
    if not corpus.is_file():
        msg = f"corpus not found: {corpus}; run `make extract`"
        raise TokenizerError(msg)
    with corpus.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                msg = f"{corpus}:{number} is not JSON: {error}"
                raise TokenizerError(msg) from error
            if not isinstance(record, dict) or "text" not in record:
                msg = f"{corpus}:{number} has no `text` field"
                raise TokenizerError(msg)
            yield str(record["text"])


def write_plain(corpus: Path, dest: Path) -> int:
    """Flatten to one sentence per line, which is the only format the trainer reads.

    Written to a sibling temporary first: a run interrupted midway would otherwise leave
    a truncated file that looks complete and silently trains on a prefix of the corpus.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_suffix(dest.suffix + ".partial")
    written = 0
    with staging.open("w", encoding="utf-8") as out:
        for text in read_texts(corpus):
            flat = " ".join(text.split())
            if flat:
                out.write(flat + "\n")
                written += 1
    staging.replace(dest)
    return written


def word_frequencies(corpus: Path) -> Counter[str]:
    freq: Counter[str] = Counter()
    for text in read_texts(corpus):
        freq.update(text.split())
    return freq


def sample_sentences(corpus: Path, size: int, seed: int = RANDOM_SEED) -> list[str]:
    """Reservoir sample, so the whole corpus is represented without loading it."""
    if size < 1:
        msg = f"sample size {size} must be positive"
        raise TokenizerError(msg)
    rng = random.Random(seed)  # noqa: S311 — a fixed-seed sample, not a secret
    kept: list[str] = []
    for index, text in enumerate(read_texts(corpus)):
        if len(kept) < size:
            kept.append(text)
        else:
            slot = rng.randint(0, index)
            if slot < size:
                kept[slot] = text
    return kept
