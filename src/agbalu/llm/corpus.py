"""The row model behind the CPT corpus, and the split that keeps an evaluation set out of it.

A held-out set is only worth measuring if the training corpus provably never saw it. That
guarantee is made by construction here rather than by bookkeeping: `is_held_out` is a pure
function of one *string*, `mixture` skips every record `withheld` selects and `holdout`
writes the sides `is_held_out` selects, so the two files cannot drift and neither has to be
built first.

The two predicates are deliberately different, and the difference is the guarantee.
Selection is per string; **exclusion is per record, on either side**. A sentence does not
appear in one pair only — 932 English evaluation sentences drawn on the Kabyle side alone
put 274 of themselves back into training through a *different* Kabyle partner, which is
`CLAUDE.md` §6.3's "match a filter to a pool on the pair, never the sentence" arriving from
the other direction. Excluding a record when either of its sides is selected makes the
invariant unconditional: if a string is in an evaluation set, every record carrying it is
out of the corpus, however many partners it has.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

Kind = Literal["kabyle", "aligned"]

KABYLE: Final = "kab"
LANGUAGE_TAG: Final[dict[str, str]] = {
    "kab": "kab_Latn",
    "eng": "eng_Latn",
    "fra": "fra_Latn",
}
"""The languages the CPT corpus carries, in the NLLB codes the documents are tagged with."""

HOLDOUT_RATE: Final = 100
"""One record in 100, set by the *smallest* evaluation set rather than the largest. English
is the binding one: most of its sentences are shorter than `holdout.MIN_CHARS`, so 1 in 500
yielded 214 documents — too few to detect the modest regression the retention gate exists to
catch. At 1 in 100 it yields 832, beside 2,000 Kabyle and 2,691 French, and costs training
75,024 records of 5.33M."""

HOLDOUT_SALT: Final = b"agbalu-llm-holdout-1"
"""Changing it re-draws the split, which invalidates every baseline measured against the
old one. Bump the suffix deliberately, never silently."""


class CorpusError(Exception):
    """A source is unreadable, or a row does not carry the sides it claims to."""


@dataclass(frozen=True, slots=True)
class Source:
    """One input file and how to read a Kabyle-bearing row out of it."""

    name: str
    path: Path
    kind: Kind
    fields: tuple[str, ...]
    """One field for `kabyle`. Two for `aligned`, in file order, not language order."""

    code: str = ""
    """NLLB code of the non-Kabyle side when every row has the same layout."""

    direction_field: str = ""
    """Field naming the row's direction, `src-tgt`, when the layout varies by row."""

    def __post_init__(self) -> None:
        expected = 1 if self.kind == "kabyle" else 2
        if len(self.fields) != expected:
            message = f"{self.name}: {self.kind} needs {expected} fields, got {len(self.fields)}"
            raise CorpusError(message)
        if self.kind == "kabyle" and (self.code or self.direction_field):
            message = f"{self.name}: a monolingual source has no other side to name"
            raise CorpusError(message)
        if self.kind == "aligned" and bool(self.code) == bool(self.direction_field):
            message = (
                f"{self.name}: an aligned source needs either a fixed language code or a "
                f"direction field, not both and not neither"
            )
            raise CorpusError(message)


@dataclass(frozen=True, slots=True)
class Record:
    """One usable row: the Kabyle side, and the other language if there is one."""

    kabyle: str
    index: int
    """Row number in the file. Decides which side leads, so the order is reproducible."""

    other: str = ""
    code: str = ""

    def __post_init__(self) -> None:
        if bool(self.other) != bool(self.code):
            message = f"row {self.index}: an aligned record needs both a side and its code"
            raise CorpusError(message)

    @property
    def aligned(self) -> bool:
        return bool(self.other)


def sides(source: Source, row: dict[str, object], line: int) -> tuple[str, str, str] | None:
    """The Kabyle side, the other side and its code, for one aligned row.

    `None` when the row does not carry both sides. The direction is what identifies the
    Kabyle side: `train.jsonl` holds `kab-eng`, `eng-kab`, `kab-fra` and `fra-kab`, so the
    field named `source` is Kabyle in half of them and French or English in the other half.
    """
    values = [row.get(field) for field in source.fields]
    if not all(isinstance(v, str) and v.strip() for v in values):
        return None
    first, second = str(values[0]), str(values[1])

    if source.code:
        return first, second, source.code

    direction = row.get(source.direction_field)
    if not isinstance(direction, str) or direction.count("-") != 1:
        message = f"{source.path}:{line}: {source.direction_field}={direction!r} is not `src-tgt`"
        raise CorpusError(message)
    left, right = direction.split("-")
    if left == KABYLE:
        kabyle, other, language = first, second, right
    elif right == KABYLE:
        kabyle, other, language = second, first, left
    else:
        message = f"{source.path}:{line}: direction {direction!r} has no Kabyle side"
        raise CorpusError(message)
    if language not in LANGUAGE_TAG:
        message = f"{source.path}:{line}: no NLLB code registered for {language!r}"
        raise CorpusError(message)
    return kabyle, other, LANGUAGE_TAG[language]


def records(source: Source) -> Iterator[Record]:
    """Every usable row of one source, in file order."""
    if not source.path.is_file():
        message = f"{source.name}: not found at {source.path}"
        raise CorpusError(message)
    with source.path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"{source.path}:{number + 1} is not JSON"
                raise CorpusError(message) from exc
            if not isinstance(row, dict):
                message = f"{source.path}:{number + 1} is not a JSON object"
                raise CorpusError(message)

            if source.kind == "kabyle":
                value = row.get(source.fields[0])
                if isinstance(value, str) and value.strip():
                    yield Record(kabyle=value, index=number)
                continue

            aligned = sides(source, row, number + 1)
            if aligned is not None:
                kabyle, other, code = aligned
                yield Record(kabyle=kabyle, index=number, other=other, code=code)


def is_held_out(text: str, *, rate: int = HOLDOUT_RATE) -> bool:
    """Whether this sentence belongs to the evaluation split.

    A keyed digest rather than a shuffled index: the decision must not depend on how many
    rows precede it, so a source growing by one row does not re-draw the whole split, and
    the same sentence decides the same way wherever in the corpus it turns up.
    """
    if rate < 1:
        message = f"rate must be positive, got {rate}"
        raise CorpusError(message)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=HOLDOUT_SALT).digest()
    return int.from_bytes(digest, "big") % rate == 0


def withheld(record: Record, *, rate: int = HOLDOUT_RATE) -> bool:
    """Whether this record must stay out of training.

    Either side is enough. An aligned record whose Kabyle side is evaluated must go, and so
    must one whose English or French side is evaluated — including when the two sides were
    selected in different rows, which is the case the pair-level rule exists for.
    """
    if is_held_out(record.kabyle, rate=rate):
        return True
    return record.aligned and is_held_out(record.other, rate=rate)
