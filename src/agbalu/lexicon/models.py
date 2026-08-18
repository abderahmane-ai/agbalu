"""The unified lexical schema.

One entry type over six source formats, each entry carrying its source and licence so a
permissive-only cut is a filter. Features use UD FEATS naming (`Number=Plur`) so the
lexicon scores against `UD_Kabyle-ADPT` without a translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Upos = Literal[
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

UPOS_TAGS: Final[frozenset[str]] = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)


class LexiconError(Exception):
    """A lexical source could not be read or reconciled."""


@dataclass(frozen=True, slots=True)
class Gloss:
    """A translation, tagged with its language."""

    language: str
    text: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One form of one word, with whatever its source knew about it.

    `lemma` and `upos` are optional because the sources are: Hunspell records a stem for
    11,940 of 21,823 entries and a part of speech for 9,474.
    """

    form: str
    lemma: str | None
    upos: Upos | None
    features: tuple[tuple[str, str], ...]
    glosses: tuple[Gloss, ...]
    source: str
    licence: str
    redistribution: str

    def feats(self) -> str:
        """UD FEATS column: `Gender=Fem|Number=Sing`, or `_` when empty."""
        if not self.features:
            return "_"
        return "|".join(f"{k}={v}" for k, v in self.features)


def features_of(**pairs: str | None) -> tuple[tuple[str, str], ...]:
    """Sorted feature tuple with unset values dropped, so equal features hash equal."""
    return tuple(sorted((k, v) for k, v in pairs.items() if v))
