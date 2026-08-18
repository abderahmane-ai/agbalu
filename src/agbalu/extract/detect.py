"""Score a string for being Kabyle.

Two signals, because neither works alone. Kabyle's alphabet is a superset of
ASCII, so letter membership cannot tell Kabyle from English; and short sentences
often carry no Kabyle-specific letter at all, so the letter signal alone rejects
them.

**Aggregate use only.** Pooled over 3,000 seed sentences the columns separate
completely — Kabyle 0.962, English 0.000, French 0.000, and the gap holds down to
two sentences. Per sentence it does not: 10% of genuine Kabyle sentences carry no
Kabyle-specific letter and score 0.0. Filtering individual rows on this score
would delete real data, so it selects columns and never gates rows.
"""

from __future__ import annotations

import re
from typing import Final

SPECIFIC: Final = frozenset("čḍɛǧɣḥṛṣṭẓČḌƐǦƔḤṚṢṬẒţŢ")
"""Letters that occur in Kabyle and in no major contact language of the corpus."""

FUNCTION_WORDS: Final = frozenset(
    (
        "d", "n", "i", "s", "ur", "ara", "ɣer", "deg", "am", "ay", "ass", "aql",
        "mačči", "acu", "anwa", "anta", "ɣef", "akken", "imi", "war", "neɣ", "yal",
        "kra", "ula", "ma", "ala", "yid", "nnig", "ddaw", "gar", "seg", "alamma",
        "imir", "mi", "ideg", "anida", "amek", "ayen", "wa", "ta", "wi", "ti", "nni",
        "agi", "nne", "ines", "nsen", "yiwen", "sin", "kan", "daɣen", "dɣa", "ihi",
        "maca", "acḥal",
    )
)  # fmt: skip
"""Closed-class Kabyle items. Closed classes cannot be borrowed away, so their
rate is stable across register in a way content words are not."""

_TOKEN: Final = re.compile(r"[^\W\d_]+", re.UNICODE)
_LETTER: Final = re.compile(r"[^\W\d_]", re.UNICODE)

LATIN_MIN: Final = 0.55
"""Below this the string is predominantly another script and is not Kabyle Latin."""

LATIN_BLOCK_END: Final = 0x0250
"""End of Latin Extended-B. Above it lie IPA, Greek, Cyrillic and everything else;
the Kabyle letters that live up there are enumerated in `SPECIFIC`."""


def latin_ratio(text: str) -> float:
    letters = _LETTER.findall(text)
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if c.isascii() or c in SPECIFIC or ord(c) < LATIN_BLOCK_END)
    return latin / len(letters)


def specific_rate(text: str) -> float:
    """Share of letters that are Kabyle-specific."""
    letters = _LETTER.findall(text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in SPECIFIC) / len(letters)


def function_rate(text: str) -> float:
    """Share of tokens that are closed-class Kabyle items."""
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in FUNCTION_WORDS) / len(tokens)


SPECIFIC_SATURATION: Final = 0.08
FUNCTION_SATURATION: Final = 0.30
SPECIFIC_WEIGHT: Final = 0.6


def _blend(specific: float, function: float) -> float:
    """Weighted evidence, gated on the letter signal being present at all.

    The gate is what stops English scoring as Kabyle. `i`, `am`, `d`, `n` and `s`
    are all Kabyle closed-class items *and* ordinary English words, so the function
    signal alone rates "I am at home and will not go" at 0.42 — above any usable
    threshold. No Kabyle-specific letter anywhere in the sample means not Kabyle.
    """
    if specific <= 0.0:
        return 0.0
    return SPECIFIC_WEIGHT * min(specific / SPECIFIC_SATURATION, 1.0) + (
        1.0 - SPECIFIC_WEIGHT
    ) * min(function / FUNCTION_SATURATION, 1.0)


def kabyle_score(text: str) -> float:
    """Combined evidence that `text` is Kabyle, in [0, 1]."""
    if latin_ratio(text) < LATIN_MIN:
        return 0.0
    return _blend(specific_rate(text), function_rate(text))


def column_score(values: list[str]) -> float:
    """Kabyle score for a whole column.

    The signals are pooled over the column rather than averaged per value: a single
    short sentence carries too little evidence, and averaging lets a column of many
    such sentences look like noise.
    """
    kept = [v for v in values if v.strip() and latin_ratio(v) >= LATIN_MIN]
    if not kept:
        return 0.0
    joined = "\n".join(kept)
    return _blend(specific_rate(joined), function_rate(joined))
