"""Spotting a segment whose translation failed, so it can be decoded again.

A document is thousands of segments, and a rate low enough to be invisible in a benchmark
is still several ruined paragraphs. The response is not a global decoding change: penalties
and n-gram blocking are blunt, they alter the distribution the published numbers were
measured under, and the King James genealogies are the case that shows why — `X begat Y; Y
begat Z` repeats by design, and blocking repeated n-grams would damage the passages that are
correct. So the first pass decodes exactly as the benchmark does, this module names the
segments where that failed, and only those are decoded again with the penalties on.

Detect, then fall back, is what [Guerreiro et al. (2023)](https://aclanthology.org/2023.tacl-1.85)
recommend for hallucination in NLLB. Sequence log-probability is their most stable detector
and is not available here without changing what `generate` returns; these three surface
signals catch the failures that reached the output of real runs.
"""

from __future__ import annotations

import re
from typing import Final

SHORT_PHRASE_REPEATS: Final = 3
LONG_PHRASE_REPEATS: Final = 4
SHORT_PHRASE_WORDS: Final = 2
"""How many back-to-back repeats make a loop, and where the threshold changes.

Litany and genealogy repeat: *"Holy, holy, holy"* is three of a one-word phrase, so a
short phrase needs a fourth before it is a defect — except that `n tteɣt n tteɣt n tteɣt`,
the degeneration this catches, is three of a two-word phrase. Three above two words is a
figure no source document uses, so the threshold moves with the phrase length rather than
being one number that is wrong at one end."""

LOOP_WORDS: Final = 5
"""The longest phrase length checked for a loop. Beyond this a repeat is quotation."""

MIN_LENGTH_RATIO: Final = 0.25
MAX_LENGTH_RATIO: Final = 3.0
"""Output words over source words. Kabyle runs about 0.87 of English on this corpus —
23,070 words against 26,525 for *Alice* — so these bound gross failure, not register."""

RATIO_MIN_WORDS: Final = 4
"""Below this a ratio is noise: a three-word source can legitimately double or halve.
Also the floor for the copy check, where a short line may share every word with its
source by coincidence — `d ayen` is two words that occur in both languages' text."""

COPY_OVERLAP: Final = 0.75
"""Share of the hypothesis's words that also occur in its source before it reads as an
untranslated copy rather than a translation."""

LETTER: Final = re.compile(r"[^\W\d_]", re.UNICODE)
WORD: Final = re.compile(r"[^\W\d_]+", re.UNICODE)

SUBWORD_STUTTER: Final = re.compile(
    r"(\b[^\W\d_]{2,8}\b|\b[^\W\d_]{2,8}[\s·\-_]+)([\s·\-_]*\1){2,}", re.UNICODE
)
"""A fragment repeating with or without a separator — `·ak·ak·ak`, `nni-nni-nni`.

Not caught by the phrase check, which splits on whitespace: a stutter inside one
whitespace token is one word to `str.split` and a loop to a reader."""


def loops(text: str) -> bool:
    """Whether a phrase of up to `LOOP_WORDS` words repeats back to back."""
    words = text.split()
    for size in range(1, LOOP_WORDS + 1):
        repeats = SHORT_PHRASE_REPEATS if size <= SHORT_PHRASE_WORDS else LONG_PHRASE_REPEATS
        for start in range(len(words) - size * repeats + 1):
            phrase = words[start : start + size]
            if all(
                words[start + size * n : start + size * (n + 1)] == phrase
                for n in range(1, repeats)
            ):
                return True
    return bool(SUBWORD_STUTTER.search(text))


def empty(source: str, hypothesis: str) -> bool:
    """Whether a source with words in it came back with none."""
    return LETTER.search(source) is not None and LETTER.search(hypothesis) is None


def malformed_length(source: str, hypothesis: str) -> bool:
    """Whether the output is grossly shorter or longer than the source it came from."""
    source_words = len(source.split())
    if source_words < RATIO_MIN_WORDS:
        return False
    ratio = len(hypothesis.split()) / source_words
    return not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO


def copied(source: str, hypothesis: str) -> bool:
    """Whether the source came back untranslated rather than translated.

    Compared on word sets rather than characters, case-folded: a copy survives the typography
    fold, so `’E’s` and `'E's` must count as the same word.
    """
    words = [word.casefold() for word in WORD.findall(hypothesis)]
    if len(words) < RATIO_MIN_WORDS:
        return False
    present = {word.casefold() for word in WORD.findall(source)}
    return sum(1 for word in words if word in present) / len(words) >= COPY_OVERLAP


def failed(source: str, hypothesis: str) -> bool:
    """Whether this translation should be decoded again under the fallback settings."""
    return (
        empty(source, hypothesis)
        or loops(hypothesis)
        or malformed_length(source, hypothesis)
        or copied(source, hypothesis)
    )
