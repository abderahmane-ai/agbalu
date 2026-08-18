"""Mechanically checkable defects in a sentence pair.

Not an adequacy judgement: nothing here distinguishes a good translation from a
plausible-looking wrong one, which needs a human and is task 4.2. What these signals
give is the other half — pairs defective on evidence, such as two byte-identical sides
or disagreeing numbers.

The rate of these defects is a lower bound on a mined corpus's error rate, and is
reported as one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Final

from agbalu.extract.detect import LATIN_MIN, latin_ratio
from agbalu.parallel.langid import ForeignLang, rates

DefectKind = str

MIN_CHARS: Final = 2
MAX_CHARS: Final = 1000
MAX_LENGTH_RATIO: Final = 4.0
"""Beyond this the sides cannot be translations of each other.

Kabyle is agglutinative and writes clitics with hyphens, so it runs shorter than
English in tokens and comparable in characters; 4x either way is far outside the
spread of any human-authored source in the registry.
"""

MIN_TOKENS_FOR_RATIO: Final = 3
"""Below this, length ratio is noise: 'Yes.' / 'Ih.' is a 1:1 pair either way."""

_NUMBER: Final = re.compile(r"\d[\d.,   ]*\d|\d")
_URL: Final = re.compile(r"https?://\S+|www\.\S+")

_PLACEHOLDER: Final = re.compile(
    r"""
    %\(\w+\)[a-zA-Z]      # %(name)s
  | %\d+\$[\d.]*[a-zA-Z]  # %1$S, %1$.02f
  | %[\d.]*[a-zA-Z]       # %s, %d, %.2f
  | \{[^{}]*\}            # {0}, {name}
  | \$\{[^{}]*\}          # ${name}
  | &[a-zA-Z]+;           # &amp;
  | &\#\d+;               # &#160;
""",
    re.VERBOSE,
)
"""Format placeholders, stripped before numbers are compared.

`%1$S` and `%2$S` carry digits that are argument *positions*, not quantities, and
translation legitimately reorders them. Counting those digits made
`%1$S n %2$S` / `%2$S of %1$S` look like a number mismatch and put 524,399 false
positives into the first defect count — 94% of every defect reported.
"""

_DIGIT_GROUPING: Final = str.maketrans("", "", ",.   ")


HARD_DEFECTS: Final = frozenset(
    {
        "empty",
        "too-short",
        "untranslated-copy",
        "kab-wrong-script",
        "foreign-language-mismatch",
        "length-ratio",
    }
)
"""Defects that disqualify a pair on their own, with no judgement needed."""

SOFT_DEFECTS: Final = frozenset({"number-mismatch", "url-mismatch", "too-long"})
"""Inconsistencies that may be a translation error or may be formatting."""


@dataclass(frozen=True, slots=True)
class PairDefects:
    """Every mechanical defect found in one pair."""

    kinds: tuple[DefectKind, ...]
    length_ratio: float

    @property
    def defective(self) -> bool:
        return bool(self.kinds)

    @property
    def hard(self) -> bool:
        """Unusable regardless of adequacy. Only these enter the error bound."""
        return any(k in HARD_DEFECTS for k in self.kinds)

    @property
    def soft(self) -> bool:
        return any(k in SOFT_DEFECTS for k in self.kinds)


def numbers(text: str) -> Counter[str]:
    """Quantities in `text`, as a multiset.

    A multiset, not a sequence: translation reorders freely, so `3 of 5` and `5 seg 3`
    agree on their numbers. Digit-grouping is stripped so `1,000`, `1.000` and
    `1 000` compare equal across English, French and Kabyle conventions.
    """
    stripped = _PLACEHOLDER.sub(" ", text)
    found: Counter[str] = Counter()
    for raw in _NUMBER.findall(stripped):
        digits = raw.translate(_DIGIT_GROUPING)
        if digits:
            found[digits.lstrip("0") or "0"] += 1
    return found


def length_ratio(kab: str, foreign: str) -> float:
    """Longer side over shorter, in characters. 1.0 when either side is empty."""
    shorter, longer = sorted((len(kab), len(foreign)))
    if shorter == 0:
        return 1.0
    return longer / shorter


def inspect(kab: str, foreign: str, expected: ForeignLang) -> PairDefects:
    """Every mechanical reason this pair cannot be a good translation."""
    kinds: list[DefectKind] = []
    ratio = length_ratio(kab, foreign)

    if not kab.strip() or not foreign.strip():
        return PairDefects(kinds=("empty",), length_ratio=ratio)
    if len(kab) < MIN_CHARS or len(foreign) < MIN_CHARS:
        kinds.append("too-short")
    if len(kab) > MAX_CHARS or len(foreign) > MAX_CHARS:
        kinds.append("too-long")
    if kab.strip().casefold() == foreign.strip().casefold():
        kinds.append("untranslated-copy")

    tokens = min(len(kab.split()), len(foreign.split()))
    if tokens >= MIN_TOKENS_FOR_RATIO and ratio > MAX_LENGTH_RATIO:
        kinds.append("length-ratio")

    if numbers(kab) != numbers(foreign):
        kinds.append("number-mismatch")
    if _URL.findall(kab) != _URL.findall(foreign):
        kinds.append("url-mismatch")

    if latin_ratio(kab) < LATIN_MIN:
        kinds.append("kab-wrong-script")

    if expected in ("eng", "fra"):
        english, french = rates(foreign)
        best = "eng" if english >= french else "fra"
        if max(english, french) > 0.0 and best != expected:
            kinds.append("foreign-language-mismatch")

    return PairDefects(kinds=tuple(kinds), length_ratio=ratio)
