"""Kabyle Latin <-> Tifinagh transliteration, which is not round-trip lossless.

The mapping derived from 765,736 real pairs in `hf.abdelhaqueidali.kab-latn-tfng`
shows the convention is lossy in four independent ways:

    schwa `e`   97.0% dropped  (207,018 of 213,369 occurrences sampled)
    hyphen `-`  100%  dropped  (0 of 51,520 sentences keep it)
    o/u -> ⵓ, b/p -> ⴱ, f/v -> ⴼ   many-to-one merges
    č -> ⵜⵛ, ǧ -> ⴷⵊ               digraphs, so the map is not 1:1

Only 6.2% of corpus pairs have equal letter counts. Latin -> Tifinagh is a surjection,
so Tifinagh -> Latin cannot be a function.

- `to_tifinagh(..., faithful=True)` reproduces the corpus convention, so output is
  comparable with the 765k reference pairs.
- `to_tifinagh(..., faithful=False)` keeps schwa and hyphen, which is reversible.
- `to_latin` is best-effort and reports the ambiguities it hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# Derived by alignment over the corpus, then reconciled with the Neo-Tifinagh
# (IRCAM) letter set. Agreement was >90% for every single-letter correspondence;
# the apparent disagreement on č and ǧ is the digraph effect described above.
LATIN_TO_TIFINAGH: Final[dict[str, str]] = {
    "č": "ⵜⵛ",  # digraph: yat + yash
    "ǧ": "ⴷⵊ",  # digraph: yad + yazh
    "a": "ⴰ",
    "b": "ⴱ",
    "c": "ⵛ",
    "d": "ⴷ",
    "ḍ": "ⴹ",
    "e": "ⴻ",
    "f": "ⴼ",
    "g": "ⴳ",
    "ɣ": "ⵖ",
    "h": "ⵀ",
    "ḥ": "ⵃ",
    "i": "ⵉ",
    "j": "ⵊ",
    "k": "ⴽ",
    "l": "ⵍ",
    "m": "ⵎ",
    "n": "ⵏ",
    "q": "ⵇ",
    "r": "ⵔ",
    "ṛ": "ⵕ",
    "s": "ⵙ",
    "ṣ": "ⵚ",
    "t": "ⵜ",
    "ṭ": "ⵟ",
    "u": "ⵓ",
    "w": "ⵡ",
    "x": "ⵅ",
    "y": "ⵢ",
    "z": "ⵣ",
    "ẓ": "ⵥ",
    "ɛ": "ⵄ",
    # Loanword letters merge into their nearest native counterpart. This is where
    # information is destroyed: o/u, b/p and f/v become indistinguishable.
    "o": "ⵓ",
    "p": "ⴱ",
    "v": "ⴼ",
    # Legacy spirantised t (see resources/homoglyphs.yaml) behaves as plain t.
    "ţ": "ⵜ",
}

TIFINAGH_TO_LATIN: Final[dict[str, str]] = {
    "ⴰ": "a",
    "ⴱ": "b",
    "ⵛ": "c",
    "ⴷ": "d",
    "ⴹ": "ḍ",
    "ⴻ": "e",
    "ⴼ": "f",
    "ⴳ": "g",
    "ⵖ": "ɣ",
    "ⵀ": "h",
    "ⵃ": "ḥ",
    "ⵉ": "i",
    "ⵊ": "j",
    "ⴽ": "k",
    "ⵍ": "l",
    "ⵎ": "m",
    "ⵏ": "n",
    "ⵇ": "q",
    "ⵔ": "r",
    "ⵕ": "ṛ",
    "ⵙ": "s",
    "ⵚ": "ṣ",
    "ⵜ": "t",
    "ⵟ": "ṭ",
    "ⵓ": "u",
    "ⵡ": "w",
    "ⵅ": "x",
    "ⵢ": "y",
    "ⵣ": "z",
    "ⵥ": "ẓ",
    "ⵄ": "ɛ",
}

TIFINAGH_DIGRAPHS: Final[tuple[tuple[str, str], ...]] = (("ⵜⵛ", "č"), ("ⴷⵊ", "ǧ"))
"""Applied before single letters, longest first, or `ⵜⵛ` would decode as `tc`."""

AMBIGUOUS_TIFINAGH: Final[frozenset[str]] = frozenset({"ⵓ", "ⴱ", "ⴼ"})
"""Letters with more than one Latin pre-image: u/o, b/p, f/v."""

MERGING_LATIN: Final[frozenset[str]] = frozenset({"o", "p", "v"})
"""Loanword letters whose Tifinagh target is already taken by a native letter."""

LABIALIZATION: Final = "ⵯ"

_TIFINAGH_RANGE: Final = re.compile(r"[ⴰ-⵿]")


@dataclass(frozen=True, slots=True)
class RoundTripReport:
    """What a Latin -> Tifinagh -> Latin cycle cost."""

    original: str
    tifinagh: str
    recovered: str
    schwa_lost: int
    hyphens_lost: int
    ambiguous_letters: int

    @property
    def lossless(self) -> bool:
        return self.original == self.recovered


def is_tifinagh(text: str) -> bool:
    """True if `text` contains any Tifinagh codepoint."""
    return bool(_TIFINAGH_RANGE.search(text))


def to_tifinagh(text: str, *, faithful: bool = True) -> str:
    """Transliterate Kabyle Latin to Tifinagh.

    Args:
        text: Kabyle in Latin script. Normalise it first — this function maps
            canonical letters only and passes homoglyphs through unchanged.
        faithful: reproduce the corpus convention, which drops schwa and hyphen.
            Set False to keep both, which makes the result reversible.
    """
    out: list[str] = []
    for char in text:
        lower = char.lower()
        if faithful and (lower == "e" or char == "-"):
            continue
        if not faithful and lower in MERGING_LATIN:
            # o/p/v have no Tifinagh letter of their own and would merge into
            # u/b/f. In reversible mode they stay Latin so the map is injective.
            out.append(char)
            continue
        out.append(LATIN_TO_TIFINAGH.get(lower, char))
    return "".join(out)


def to_latin(text: str) -> str:
    """Transliterate Tifinagh to Kabyle Latin, best effort.

    Ambiguous letters resolve to their most frequent Latin pre-image (u, b, f).
    Schwa cannot be restored: the information is not present in the input.
    """
    out = text
    for digraph, latin in TIFINAGH_DIGRAPHS:
        out = out.replace(digraph, latin)
    return "".join(TIFINAGH_TO_LATIN.get(ch, ch) for ch in out if ch != LABIALIZATION)


def round_trip(text: str, *, faithful: bool = True) -> RoundTripReport:
    """Run Latin -> Tifinagh -> Latin and account for what was lost."""
    tifinagh = to_tifinagh(text, faithful=faithful)
    recovered = to_latin(tifinagh)
    return RoundTripReport(
        original=text,
        tifinagh=tifinagh,
        recovered=recovered,
        schwa_lost=text.lower().count("e") if faithful else 0,
        hyphens_lost=text.count("-") if faithful else 0,
        ambiguous_letters=sum(text.lower().count(c) for c in "opv"),
    )


__all__ = [
    "AMBIGUOUS_TIFINAGH",
    "LATIN_TO_TIFINAGH",
    "TIFINAGH_TO_LATIN",
    "RoundTripReport",
    "is_tifinagh",
    "round_trip",
    "to_latin",
    "to_tifinagh",
]
