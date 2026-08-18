"""Typed records describing what normalisation did, and what it refused to do."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChangeKind = Literal[
    "nfc",
    "homoglyph",
    "diacritic-fold",
    "whitespace",
    "punctuation",
    "invisible-removed",
]
"""Why a character changed. Every applied edit carries one."""

FlagKind = Literal[
    "legacy-t-cedilla",
    "rejected-character",
    "ascii-digraph",
    "out-of-inventory",
    "mixed-script",
    "foreign-proper-noun",
]
"""Why a span needs human judgement. Flags never modify the text.

`legacy-t-cedilla` is `ţ`: real Kabyle orthography from the Dallet tradition whose
modern continuation is ambiguous between `t` and `tt`, so it is never rewritten.
`ascii-digraph` is `ch`/`gh`/`dj`: collides with French and English loanwords.
`foreign-proper-noun` is a capitalised token in another language's orthography,
where a homoglyph rule would invent a word rather than repair one.
"""


@dataclass(frozen=True, slots=True)
class Change:
    """One applied edit, positioned in the *input* string."""

    kind: ChangeKind
    position: int
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class Flag:
    """One span the normaliser deliberately left alone."""

    kind: FlagKind
    position: int
    text: str
    detail: str


@dataclass(frozen=True, slots=True)
class NormalisationResult:
    """The normalised string plus a full account of how it got there."""

    text: str
    original: str
    version: str
    changes: tuple[Change, ...] = field(default=())
    flags: tuple[Flag, ...] = field(default=())

    @property
    def changed(self) -> bool:
        return self.text != self.original

    def count(self, kind: ChangeKind) -> int:
        return sum(1 for c in self.changes if c.kind == kind)
