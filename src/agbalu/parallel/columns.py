"""Pick the Kabyle field and its aligned translation from a record schema.

Choosing two fields is not choosing one twice. The Kabyle side is found the way
`extract.columns` finds it; the foreign side then has to be the field *aligned* to
it, which no content score can establish — a file may carry three translations, or
a Kabyle field plus unrelated metadata prose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from agbalu.extract.columns import METADATA, choose_field
from agbalu.parallel.langid import ForeignLang, identify

EXPLICIT_FOREIGN: Final[dict[str, ForeignLang]] = {
    "en": "eng", "eng": "eng", "english": "eng", "en_us": "eng", "eng_latn": "eng",
    "source": "eng", "source_string": "eng", "en_sentence": "eng",
    "fr": "fra", "fra": "fra", "french": "fra", "fre": "fra", "fra_latn": "fra",
    "francais": "fra",
}  # fmt: skip

MIN_COVERAGE: Final = 0.5
"""A candidate must be present and non-empty on at least this share of sampled rows.

A field that is mostly absent cannot be the aligned translation, however well its
few values score.
"""


def _leaf(name: str) -> str:
    return name.rsplit(".", maxsplit=1)[-1].strip().lower()


def _coverage(records: Sequence[Mapping[str, str]], name: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get(name, "").strip()) / len(records)


def choose_pair(
    records: Sequence[Mapping[str, str]],
) -> tuple[str | None, str | None, ForeignLang]:
    """`(kabyle_field, foreign_field, foreign_language)`.

    The foreign field is preferred by name, because a name states the alignment and
    content cannot. Only when no name matches is the best-covered non-metadata field
    identified by content, and a field whose language cannot be established is
    rejected rather than guessed.
    """
    kab_field, _ = choose_field(list(records))
    if kab_field is None:
        return None, None, "other"

    candidates = [
        name
        for name in {n for r in records for n in r}
        if name != kab_field
        and _leaf(name) not in METADATA
        and _coverage(records, name) >= MIN_COVERAGE
    ]
    if not candidates:
        return kab_field, None, "other"

    for name in sorted(candidates):
        declared = EXPLICIT_FOREIGN.get(_leaf(name))
        if declared is None:
            continue
        found = identify([r.get(name, "") for r in records])
        if found in (declared, "other"):
            return kab_field, name, declared

    best: tuple[str, ForeignLang] | None = None
    for name in sorted(candidates):
        found = identify([r.get(name, "") for r in records])
        if found != "other":
            best = (name, found)
            break
    if best is None:
        return kab_field, None, "other"
    return kab_field, best[0], best[1]
