"""The Kabyle annexed state, reversed.

A Kabyle noun is obligatorily prefixed after a preposition (`axxam` -> `deg wexxam`) and
the dictionaries list only the free form. The rules below are read off the 4,401 pairs
Amawal attests, and explain 95.7% of them; counts are in
`docs/phases/phase-06-lexicon.md` finding 1.

Analysis only. The alternation is many-to-one that way — `we-` can only come from `a-` —
but one-to-many in generation, where `a-` may become `u-`, `we-` or `wa-`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

ANNEXED_TO_FREE: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("we", ("a",)),
    ("wa", ("a",)),
    ("wu", ("u",)),
    ("ye", ("i",)),
    ("yi", ("i",)),
    ("te", ("ta", "ti")),
    ("t", ("ta", "ti")),
    ("u", ("a",)),
)
"""Longest prefix first, so `te-` is tried before the bare `t-`."""


def free_candidates(form: str) -> Iterator[str]:
    """Free-state forms `form` could be the annexed state of."""
    if not form:
        return
    seen: set[str] = set()
    for annexed, frees in ANNEXED_TO_FREE:
        if not form.startswith(annexed):
            continue
        stem = form[len(annexed) :]
        if not stem:
            continue
        for free in frees:
            candidate = free + stem
            if candidate != form and candidate not in seen:
                seen.add(candidate)
                yield candidate
