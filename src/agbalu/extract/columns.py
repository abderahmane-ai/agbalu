"""Decide which field of a record holds Kabyle text.

Most sources do not name their Kabyle column `kab`; they name it `text`,
`translation`, `sentence` or `col1`. Names are therefore a fast path, not the
mechanism: when no name matches, columns are scored on their content.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from agbalu.extract.detect import column_score

EXPLICIT_KAB: Final = frozenset(
    {"kab", "kabyle", "kab_latn", "taqbaylit", "kabyle_sentence", "kab_sentence"}
)
"""Field names that assert Kabyle Latin.

`kab_tfng` is deliberately absent: this corpus is `kab-Latn` (CLAUDE.md 1), and
matching it pulled 758,579 Tifinagh-script lines into the first build.
"""

METADATA: Final = frozenset(
    {
        "id", "ids", "url", "uri", "source", "licence", "license", "score", "scores",
        "lang", "language", "lang_id", "langid", "split", "index", "idx", "date",
        "timestamp", "author", "title_id", "doc_id", "glot_lang", "berber_label",
        "berber_status", "dominant_lang", "lang_distribution", "status", "label",
    }
)  # fmt: skip

SAMPLE_ROWS: Final = 400
MIN_COLUMN_SCORE: Final = 0.18
"""A column must beat this to be accepted as Kabyle when no name identifies it.

Pooled over 3,000 rows of the seed corpus: Kabyle 0.962, English 0.000, French
0.000. The threshold sits in an empty gap and is not delicate.
"""

_NUMERIC: Final = re.compile(r"^[\d\s.,:%+-]*$")


def _leaf(name: str) -> str:
    return name.rsplit(".", maxsplit=1)[-1].strip().lower()


def _is_metadata(name: str) -> bool:
    leaf = _leaf(name)
    return leaf in METADATA or leaf.endswith("_id") or leaf.endswith("_score")


def explicit_kab_field(names: Iterable[str]) -> str | None:
    """A field whose name states it is Kabyle, if one exists."""
    for name in names:
        if _leaf(name) in EXPLICIT_KAB:
            return name
    return None


def choose_field(records: Sequence[Mapping[str, str]]) -> tuple[str | None, float]:
    """Return the Kabyle field of these records and the score that justified it."""
    if not records:
        return None, 0.0

    names: list[str] = []
    for record in records:
        for name in record:
            if name not in names:
                names.append(name)

    explicit = explicit_kab_field(names)
    if explicit is not None:
        # A name is a claim, not evidence. Flattening puts a leaf `kab` on metadata
        # containers too: `lang_distribution.kab` holds the count "2", not a
        # sentence. Confirm the claim against the content before trusting it.
        score = column_score([r[explicit] for r in records if explicit in r])
        if score >= MIN_COLUMN_SCORE:
            return explicit, score

    best: str | None = None
    best_score = 0.0
    for name in names:
        if _is_metadata(name):
            continue
        values = [r[name] for r in records if name in r]
        if not values or all(_NUMERIC.match(v) for v in values):
            continue
        score = column_score(values)
        if score > best_score:
            best, best_score = name, score

    if best is None or best_score < MIN_COLUMN_SCORE:
        return None, best_score
    return best, best_score


def sample(
    records: Iterable[Mapping[str, str]], limit: int = SAMPLE_ROWS
) -> list[Mapping[str, str]]:
    """Take the first `limit` records, for column scoring."""
    collected: list[Mapping[str, str]] = []
    for record in records:
        collected.append(record)
        if len(collected) >= limit:
            break
    return collected
