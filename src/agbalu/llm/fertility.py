"""How many tokens a vocabulary spends on Kabyle (task 11.1).

Fertility decides three separate things, which is why it is measured before anything is
trained: the sequence length continued pretraining pays for, the inference cost every user
pays forever, and — per Nag et al. (NAACL 2025 Industry) — whether vocabulary expansion is
worth attempting at all, since it helped only at high fragment ratios and cost accuracy at
medium ones.

Tokenizers arrive through `Encoder` rather than being constructed here, so the suite
exercises the arithmetic without downloading a 10 GB checkpoint.

Special tokens are excluded. Counting them put a prior Kabyle UNK measurement at 7.404%
against a true 0.011% (CLAUDE.md 6.3).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

MIN_CHARS: Final = 40
MAX_CHARS: Final = 300
"""The sampled population, stated because a fertility number is meaningless without it.
Short rows are disproportionately titles and fragments; long rows are concatenated
documents. Both distort tokens-per-word against ordinary prose."""


class FertilityError(Exception):
    """The corpus cannot be sampled, or a tokenizer disagrees with its own output."""


class Encoder(Protocol):
    """The slice of a tokenizer this module drives."""

    @property
    def name(self) -> str: ...

    @property
    def unknown_id(self) -> int | None: ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...


@dataclass(frozen=True, slots=True)
class Fertility:
    """One vocabulary measured over one population."""

    name: str
    sentences: int
    words: int
    tokens: int
    unknown: int

    def __post_init__(self) -> None:
        if self.words <= 0 or self.tokens <= 0:
            message = f"{self.name}: {self.words} words and {self.tokens} tokens"
            raise FertilityError(message)

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words

    @property
    def unknown_share(self) -> float:
        return self.unknown / self.tokens

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sentences": self.sentences,
            "words": self.words,
            "tokens": self.tokens,
            "unknown": self.unknown,
            "tokens_per_word": round(self.tokens_per_word, 4),
            "unknown_share": round(self.unknown_share, 6),
        }


def texts(path: Path, field: str = "text") -> Iterator[str]:
    """Every value of `field`, skipping rows that do not carry one."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"{path}:{number} is not JSON"
                raise FertilityError(message) from exc
            value = row.get(field)
            if isinstance(value, str):
                yield value


def sample(path: Path, count: int, *, seed: int, field: str = "text") -> list[str]:
    """`count` sentences drawn uniformly from the whole file, or every eligible one.

    Reservoir sampling: the corpus is 3.04M rows and the population must be the file, not
    its first n lines. A prefix is ordered by source id, so it is a different corpus.

    The length bounds are the module constants rather than parameters. A fertility figure
    is only comparable across vocabularies and across runs if its population is fixed.
    """
    if count < 1:
        message = f"count must be positive, got {count}"
        raise FertilityError(message)

    rng = random.Random(seed)  # noqa: S311 — a fixed-seed sample, not a secret
    reservoir: list[str] = []
    seen = 0
    for text in texts(path, field):
        if not MIN_CHARS < len(text) < MAX_CHARS:
            continue
        seen += 1
        if len(reservoir) < count:
            reservoir.append(text)
            continue
        index = rng.randrange(seen)
        if index < count:
            reservoir[index] = text

    if not reservoir:
        message = f"no rows in {path} with {MIN_CHARS} < length < {MAX_CHARS}"
        raise FertilityError(message)
    return reservoir


def measure(encoder: Encoder, sentences: Sequence[str]) -> Fertility:
    """One vocabulary over one sample."""
    if not sentences:
        message = f"{encoder.name}: nothing to measure"
        raise FertilityError(message)

    encoded = encoder.encode_batch(sentences)
    if len(encoded) != len(sentences):
        message = f"{encoder.name} returned {len(encoded)} rows for {len(sentences)}"
        raise FertilityError(message)

    unknown_id = encoder.unknown_id
    return Fertility(
        name=encoder.name,
        sentences=len(sentences),
        words=sum(len(s.split()) for s in sentences),
        tokens=sum(len(ids) for ids in encoded),
        unknown=0 if unknown_id is None else sum(ids.count(unknown_id) for ids in encoded),
    )


@dataclass(frozen=True, slots=True)
class Measured:
    """One vocabulary's fertility, and its cost against the reference vocabulary."""

    fertility: Fertility
    ratio_to_reference: float | None

    def as_dict(self) -> dict[str, object]:
        row = self.fertility.as_dict()
        if self.ratio_to_reference is not None:
            row["ratio_to_reference"] = self.ratio_to_reference
        return row


@dataclass(frozen=True, slots=True)
class Report:
    """Every vocabulary over one population."""

    sentences: int
    reference: str | None
    vocabularies: tuple[Measured, ...]

    def by_name(self) -> dict[str, Measured]:
        return {m.fertility.name: m for m in self.vocabularies}

    def as_dict(self) -> dict[str, object]:
        return {
            "population": {
                "sentences": self.sentences,
                "min_chars": MIN_CHARS,
                "max_chars": MAX_CHARS,
            },
            "reference": self.reference,
            "vocabularies": [m.as_dict() for m in self.vocabularies],
        }


def report(
    encoders: Sequence[Encoder], sentences: Sequence[str], *, reference: str | None = None
) -> Report:
    """Every vocabulary over one sample, with each one's cost against `reference`."""
    measured = [measure(encoder, sentences) for encoder in encoders]
    baseline = {f.name: f for f in measured}
    if reference is not None and reference not in baseline:
        message = f"reference {reference!r} is not among {sorted(baseline)}"
        raise FertilityError(message)

    divisor = None if reference is None else baseline[reference].tokens_per_word
    rows = tuple(
        Measured(
            fertility=fertility,
            ratio_to_reference=(
                None if divisor is None else round(fertility.tokens_per_word / divisor, 4)
            ),
        )
        for fertility in measured
    )
    return Report(sentences=len(sentences), reference=reference, vocabularies=rows)
