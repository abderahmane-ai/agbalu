"""Identify the non-Kabyle side of a parallel corpus.

Only the languages the registry actually pairs with Kabyle are distinguished:
English and French. Everything else resolves to `other` rather than being guessed,
because a wrong label is worse than an absent one — it would put Spanish pairs into
an `eng` release.

Closed-class function words again, for the reason given in `extract.detect`: they
are frequent, short, and cannot be borrowed away. Scores are pooled over a column,
never trusted per sentence.
"""

from __future__ import annotations

import re
from typing import Final, Literal

ForeignLang = Literal["eng", "fra", "other"]

ENGLISH: Final = frozenset(
    (
        "the", "of", "and", "to", "in", "is", "it", "you", "that", "he", "was",
        "for", "on", "are", "with", "as", "his", "they", "at", "be", "this",
        "have", "from", "or", "had", "by", "not", "but", "what", "were", "when",
        "there", "can", "an", "your", "which", "their", "said", "will", "would",
        "about", "if", "has", "been", "who", "its", "did", "she", "her", "him",
    )
)  # fmt: skip

FRENCH: Final = frozenset(
    (
        "le", "la", "les", "de", "des", "du", "et", "un", "une", "est", "que",
        "qui", "dans", "pour", "pas", "sur", "au", "aux", "il", "elle", "ils",
        "nous", "vous", "ce", "cette", "se", "sont", "avec", "plus", "par",
        "ne", "son", "sa", "ses", "mais", "comme", "tout", "être", "avoir",
        "faire", "leur", "nos", "vos", "où", "donc", "ainsi", "chez", "sans",
    )
)  # fmt: skip

_TOKEN: Final = re.compile(r"[^\W\d_]+", re.UNICODE)

MIN_RATE: Final = 0.08
"""Below this no language is claimed.

Pooled over 3,000 rows of the seed corpora: the English column of `tatoeba_kab_eng`
scores 0.311 English / 0.000 French, the French side of OPUS Tatoeba 0.325 French /
0.005 English, and the Kabyle column 0.013 / 0.001. The threshold sits in the empty
band between Kabyle and either target.
"""

MARGIN: Final = 1.5
"""How far ahead the winner must be. English and French share function words
(`son`, `plus`, `a`, `on`), so a near-tie is not evidence."""


def rates(text: str) -> tuple[float, float]:
    """(English rate, French rate) over the tokens of `text`."""
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return 0.0, 0.0
    total = len(tokens)
    return (
        sum(1 for t in tokens if t in ENGLISH) / total,
        sum(1 for t in tokens if t in FRENCH) / total,
    )


def identify(values: list[str]) -> ForeignLang:
    """Label a whole column. Pooled, for the reason given in `extract.detect`."""
    joined = "\n".join(v for v in values if v.strip())
    if not joined:
        return "other"
    english, french = rates(joined)
    if english < MIN_RATE and french < MIN_RATE:
        return "other"
    if english >= french * MARGIN:
        return "eng"
    if french >= english * MARGIN:
        return "fra"
    return "other"
