"""The biblical exclusion index every Phase 12 prompt set is filtered through.

`facebook/mms-tts-kab` was trained on New Testament recordings, so a biblical prompt
measures how much of its training text it remembers rather than how it synthesises
Kabyle. The King James Bible is also in this project's own corpus on both sides —
559 of 800 sampled verses verbatim (CLAUDE.md §6.2) — so the exclusion protects both
systems in the comparison, not just the baseline.

`opus.bible-uedin-kab` is the Kabyle side of that text as this project holds it:
15,857 verses, indexed in both raw and normalised form because the corpus is
normalised and the bundle is not.

`load` refuses to return an index that cannot match. A join with a wrong field name
or an unparsed side returns zero hits and reads as a clean result — it has already
happened here once, on the question of whether the KJV was in the corpus — so a verse
drawn from the source itself is put through the same predicate the prompts are.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.extract import fingerprint
from agbalu.extract.readers import read_zip
from agbalu.normalise import Normaliser

BIBLE: Final = Path("data/raw/opus.bible-uedin-kab/en-kab.txt.zip")

NGRAM: Final = 5
"""Tokens in a shared run that condemns a prompt.

Shorter than `bench.contamination`'s 8 because these prompts are short: the speech
corpus averages 4.58 words and 8 tokens would exempt almost every one of them.
Measured over the 9,494 test-split clips of at least `prompts.MIN_WORDS` words, 5
catches 15 and 4 catches 209 — and the 4-gram hits are ordinary Kabyle phrasing
(`d acu ara d-tiniḍ deg wa`), not scripture. 5 is where the trade stops being free.
"""

EXACT: Final = "biblical-exact"
NGRAM_HIT: Final = "biblical-ngram"


class ScriptureError(Exception):
    """An index that cannot be built, or one that cannot match its own source."""


def _grams(text: str, size: int = NGRAM) -> set[tuple[str, ...]]:
    tokens = text.casefold().split()
    if len(tokens) < size:
        return set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


@dataclass(frozen=True, slots=True)
class Scripture:
    """Verse fingerprints and n-grams, and the count they were built from."""

    exact: frozenset[bytes]
    grams: frozenset[tuple[str, ...]]
    verses: int

    def match(self, text: str) -> str | None:
        """Why `text` is biblical, or `None`."""
        if fingerprint(text) in self.exact:
            return EXACT
        if _grams(text) & self.grams:
            return NGRAM_HIT
        return None


def verses(path: Path = BIBLE) -> Iterator[str]:
    """The Kabyle side of the OPUS bundle, one verse per line."""
    if not path.is_file():
        message = f"biblical text not found: {path}"
        raise ScriptureError(message)
    try:
        for record in read_zip(path):
            text = record["kab"].strip()
            if text:
                yield text
    except (OSError, zipfile.BadZipFile) as error:
        message = f"{path} is not a readable OPUS bundle"
        raise ScriptureError(message) from error


def index(lines: Iterable[str], normaliser: Normaliser | None = None) -> Scripture:
    """Build the index over both the raw and the normalised form of each verse."""
    engine = normaliser if normaliser is not None else Normaliser()
    exact: set[bytes] = set()
    grams: set[tuple[str, ...]] = set()
    count = 0
    for line in lines:
        count += 1
        for variant in {line, engine.normalise(line)}:
            exact.add(fingerprint(variant))
            grams.update(_grams(variant))
    return Scripture(exact=frozenset(exact), grams=frozenset(grams), verses=count)


def control(lines: Iterable[str]) -> tuple[str, str]:
    """A verse the index must match, and an edit of it only the n-gram test can catch.

    Both branches are proved, not one: `match` returns on the fingerprint first, so a
    whole verse never reaches the n-gram test and an index whose n-grams are empty
    would pass a control built from one. Dropping the leading token changes the
    fingerprint and leaves the rest of the runs intact, which is also the shape of the
    hit this test exists for — a verse quoted with its opening clipped.
    """
    for line in lines:
        tokens = line.split()
        if len(tokens) > NGRAM:
            return line, " ".join(tokens[1:])
    message = f"no verse longer than {NGRAM} tokens, so the index cannot be controlled"
    raise ScriptureError(message)


def load(path: Path = BIBLE, normaliser: Normaliser | None = None) -> Scripture:
    """The index, proved against its own source before it is returned."""
    lines = list(verses(path))
    built = index(lines, normaliser)
    verse, clipped = control(lines)
    if built.match(verse) != EXACT:
        message = (
            f"the index does not match a verse of its own source ({verse[:60]!r}), "
            f"so a zero-hit result from it would mean nothing"
        )
        raise ScriptureError(message)
    if built.match(clipped) is None:
        message = f"the {NGRAM}-gram index misses a clipped verse ({clipped[:60]!r})"
        raise ScriptureError(message)
    return built
