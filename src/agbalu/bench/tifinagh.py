"""Scoring for script conversion, and the deterministic table that is its baseline.

Kabyle Neo-Tifinagh writes no `e`. Every other letter maps one-to-one, so the mapping
back to Latin is a lookup for everything except the vowel — which means a character
table can be built in an afternoon and reaches 0% at sentence level. That table is the
comparison this task has, and it is kept here rather than described, so the number is
produced rather than quoted.

No torch. The metrics are scored the same way whether the hypotheses came from the
model, from the table, or from a file, and the module a benchmark lives in must not
require the training stack to import.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.bench.strings import StringScore, score_strings

SPLIT_DIR: Final = Path("data/tasks/tifinagh/script_conversion")

SCHWA: Final = "e"

LATN_TO_TFNG: Final[dict[str, str]] = {
    "a": "ⴰ",
    "b": "ⴱ",
    "g": "ⴳ",
    "d": "ⴷ",
    "ḍ": "ⴹ",
    "e": "ⴻ",
    "f": "ⴼ",
    "k": "ⴽ",
    "h": "ⵀ",
    "ḥ": "ⵃ",
    "ɛ": "ⵄ",
    "x": "ⵅ",
    "q": "ⵇ",
    "i": "ⵉ",
    "j": "ⵊ",
    "l": "ⵍ",
    "m": "ⵎ",
    "n": "ⵏ",
    "u": "ⵓ",
    "r": "ⵔ",
    "ṛ": "ⵕ",
    "ɣ": "ⵖ",
    "s": "ⵙ",
    "ṣ": "ⵚ",
    "c": "ⵛ",
    "t": "ⵜ",
    "ṭ": "ⵟ",
    "w": "ⵡ",
    "y": "ⵢ",
    "z": "ⵣ",
    "ẓ": "ⵥ",
}
"""`docs/orthography.md`'s Latin inventory against the Neo-Tifinagh repertoire. Not
injective in the direction that matters: the corpus writes no `ⴻ`, so `e` has no
Tifinagh source to be recovered from."""

TFNG_TO_LATN: Final[dict[str, str]] = {tifinagh: latin for latin, tifinagh in LATN_TO_TFNG.items()}


class BenchmarkError(Exception):
    """A scoring request that cannot produce a number."""


@dataclass(frozen=True, slots=True)
class Pair:
    """One sentence in both scripts."""

    latin: str
    tifinagh: str


def read_split(split: str, directory: Path = SPLIT_DIR, limit: int | None = None) -> list[Pair]:
    """A built split of `agbalu/KabTifinagh`, in file order.

    `limit` truncates rather than samples, which is honest only because the split was
    written under a seeded shuffle: a prefix of a source-ordered file is a different
    corpus, and the builder is what makes this one safe to cut.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    path = directory / f"{split}.parquet"
    if not path.is_file():
        message = f"no {split} split at {path}; build it with tools/build_kabtifinagh.py"
        raise BenchmarkError(message)
    table = pq.read_table(path, columns=["text_latn", "text_tfng"])
    if limit is not None:
        table = table.slice(0, limit)
    return [Pair(latin=row["text_latn"], tifinagh=row["text_tfng"]) for row in table.to_pylist()]


def to_tifinagh(text: str) -> str:
    """Latin to Tifinagh by table, case-folded."""
    return "".join(LATN_TO_TFNG.get(char, char) for char in text.lower())


def to_latin(text: str) -> str:
    """Tifinagh to Latin by table. Restores no `e`, which is the point of the baseline."""
    return "".join(TFNG_TO_LATN.get(char, char) for char in text)


@dataclass(frozen=True, slots=True)
class Schwa:
    """Precision, recall and F1 over *where* the vowel was placed.

    Counting how many `e` each side contains is not this measurement, and the difference
    is not academic: a hypothesis with the right number of them in the wrong places
    scores 100% under a count, which is what the first evaluation of this model reported.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    undefined: int
    """Sentences whose consonant skeleton the hypothesis got wrong, so no schwa in them
    has a position to be right or wrong at. Their vowels are charged to both error
    columns rather than dropped, because dropping them would score a model that fails the
    sentence outright as if it had abstained."""

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "undefined_sentences": self.undefined,
        }


def skeleton(text: str) -> tuple[str, tuple[int, ...]]:
    """`text` with every `e` removed, and the surviving positions they sat at.

    A schwa is located by how many non-`e` characters precede it, so two strings that
    agree on their consonants can have their vowel placements compared exactly, with no
    alignment step and no tolerance to tune. `tudert` gives `tudrt` and `(4,)`: one `e`
    after four characters.
    """
    consonants: list[str] = []
    positions: list[int] = []
    for char in text.lower():
        if char == SCHWA:
            positions.append(len(consonants))
        else:
            consonants.append(char)
    return "".join(consonants), tuple(positions)


def score_conversion(references: Iterable[str], hypotheses: Iterable[str]) -> StringScore:
    """Exact match and CER, both computed over the whole set rather than per sentence."""
    return score_strings(references, hypotheses)


def score_schwa(references: Iterable[str], hypotheses: Iterable[str]) -> Schwa:
    """Whether each `e` was placed where the reference places it."""
    true_positives = false_positives = false_negatives = undefined = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        reference_skeleton, reference_positions = skeleton(reference)
        hypothesis_skeleton, hypothesis_positions = skeleton(hypothesis)
        if reference_skeleton != hypothesis_skeleton:
            undefined += 1
            false_negatives += len(reference_positions)
            false_positives += len(hypothesis_positions)
            continue
        matched = _overlap(reference_positions, hypothesis_positions)
        true_positives += matched
        false_negatives += len(reference_positions) - matched
        false_positives += len(hypothesis_positions) - matched
    return Schwa(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        undefined=undefined,
    )


def _overlap(left: Sequence[int], right: Sequence[int]) -> int:
    """Positions present in both, counting a repeated position once per occurrence.

    `ttedu` places two schwas at the same skeleton index, so a set intersection would
    score one of them as missing whatever the hypothesis did.
    """
    remaining = list(right)
    matched = 0
    for position in left:
        if position in remaining:
            remaining.remove(position)
            matched += 1
    return matched
