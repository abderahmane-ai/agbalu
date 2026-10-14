"""Training text that overlaps an evaluation set (tasks 11.16 and 11.17).

A text is contaminated when it shares a 13-word window with an evaluation sentence, the
window the GPT-3 and Llama decontamination reports use, or contains a whole evaluation
sentence of `MIN_WORDS` to 12 words. Shorter sentences are too generic to decide anything:
they are counted and left out of the index rather than allowed to drop half the corpus.

Both sides are normalised and case-folded before matching, so a Greek `ε` in FLORES+ and a
Latin `ɛ` in the corpus are the same word here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.normalise import normalise

WINDOW: Final = 13
MIN_WORDS: Final = 8
MIN_WHOLE_WORDS: Final = 4
"""8 of 2,000 held-out Kabyle sentences, all under `MIN_WORDS`, were found inside kept
documents on 2026-10-14; a held-out sentence is indexed whole from this length."""
WORD: Final = re.compile(r"\w+")

HELDOUT_DIR: Final = Path("data/processed/llm")
FLORES_ROOT: Final = Path("data/raw/hf.flores-plus-kab")
FLORES_CORRECTED: Final = Path("data/processed/bench/flores-plus-kab_Latn-corrected.jsonl")
SIB_ROOT: Final = Path("data/raw/hf.sib200-kab/kab_Latn")
FLORES_LANGUAGES: Final = ("kab_Latn", "eng_Latn", "fra_Latn")
HELDOUT_LANGUAGES: Final = ("kab", "eng", "fra")


class DecontaminationError(Exception):
    """An evaluation set is missing, or the index cannot match what it was built from."""


def words(text: str) -> list[str]:
    return WORD.findall(normalise(text).casefold())


@dataclass(slots=True)
class Decontaminator:
    """An n-gram index over every evaluation sentence."""

    grams: dict[int, set[tuple[str, ...]]] = field(default_factory=dict)
    indexed: int = 0
    too_short: int = 0

    @classmethod
    def build(cls, sentences: Iterable[str], whole: Iterable[str] = ()) -> Decontaminator:
        """`whole` sentences are indexed entire from `MIN_WHOLE_WORDS` up: the held-out sets,
        whose short sentences otherwise survive inside training documents."""
        index = cls()
        for sentence in whole:
            tokens = words(sentence)
            if MIN_WHOLE_WORDS <= len(tokens) < WINDOW:
                index.grams.setdefault(len(tokens), set()).add(tuple(tokens))
        for sentence in sentences:
            tokens = words(sentence)
            if len(tokens) < MIN_WORDS:
                index.too_short += 1
                continue
            index.indexed += 1
            if len(tokens) < WINDOW:
                index.grams.setdefault(len(tokens), set()).add(tuple(tokens))
                continue
            bucket = index.grams.setdefault(WINDOW, set())
            for start in range(len(tokens) - WINDOW + 1):
                bucket.add(tuple(tokens[start : start + WINDOW]))
        if index.indexed == 0:
            message = "no evaluation sentence is long enough to index; the check could never fire"
            raise DecontaminationError(message)
        return index

    def matches(self, text: str) -> bool:
        tokens = words(text)
        for length, bucket in self.grams.items():
            for start in range(len(tokens) - length + 1):
                if tuple(tokens[start : start + length]) in bucket:
                    return True
        return False

    def require_positive(self, sentence: str) -> None:
        """Refuse to run unless the index fires on a sentence it was built from."""
        if not self.matches(f"Prefix words that are not evaluated. {sentence} And a suffix."):
            message = f"the index does not match its own evaluation sentence: {sentence[:80]!r}"
            raise DecontaminationError(message)


def _jsonl_texts(path: Path, key: str) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(key) if isinstance(row, dict) else None
            if not isinstance(value, str):
                message = f"{path}:{number} has no string {key!r}"
                raise DecontaminationError(message)
            yield value


def evaluation_sentences(
    heldout_dir: Path = HELDOUT_DIR,
    flores_root: Path = FLORES_ROOT,
    corrected: Path = FLORES_CORRECTED,
    sib_root: Path = SIB_ROOT,
) -> list[str]:
    """Every sentence any Phase 11 evaluation scores, from the files on disk."""
    import pyarrow.parquet as pq

    paths: list[tuple[Path, str]] = [
        (heldout_dir / f"heldout-{language}.jsonl", "text") for language in HELDOUT_LANGUAGES
    ]
    paths += [
        (flores_root / split / f"{language}.jsonl", "text")
        for split in ("dev", "devtest")
        for language in FLORES_LANGUAGES
    ]
    paths.append((corrected, "text"))
    missing = [str(path) for path, _ in paths if not path.is_file()]
    sib = sorted(sib_root.glob("*.parquet"))
    if not sib:
        missing.append(str(sib_root / "*.parquet"))
    if missing:
        message = f"evaluation sets missing, so nothing can be decontaminated: {missing}"
        raise DecontaminationError(message)

    sentences: list[str] = []
    for path, key in paths:
        sentences.extend(_jsonl_texts(path, key))
    for part in sib:
        sentences.extend(str(text) for text in pq.read_table(part, columns=["text"])["text"])
    return sentences


def heldout_sentences(heldout_dir: Path = HELDOUT_DIR) -> list[str]:
    """Every held-out set's text, the sentences indexed whole."""
    return [
        text
        for language in HELDOUT_LANGUAGES
        for text in _jsonl_texts(heldout_dir / f"heldout-{language}.jsonl", "text")
    ]


def first_indexable(sentences: Sequence[str]) -> str:
    """The positive control: the first sentence long enough to be indexed whole."""
    for sentence in sentences:
        if len(words(sentence)) >= WINDOW:
            return sentence
    message = f"none of {len(sentences)} evaluation sentences has {WINDOW} words"
    raise DecontaminationError(message)
