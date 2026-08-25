"""Kabyle grapheme-to-phoneme that keeps the punctuation, for Matoub-Pro.

`agbalu.tts.g2p` strips every mark before it phonemises and `agbalu.tts.corpus` hands it the
ASR target, which never carried one. Between them Matoub-82M was fitted to sequences in which
no utterance ever ends, and it produces none of the pauses `. , ? !` are the supervision for.

The marks are not new tokens. `resources/kokoro_vocabulary.json` already carries
`! " ( ) , . : ; ? — “ ” …` as rows Kokoro trained, so keeping them reinforces embeddings that
arrive knowing a comma from a full stop rather than asking 13 hours of Kabyle to rediscover it.

The letter rules are imported from `agbalu.tts.g2p`, never copied. Two tables that must agree
are one table.

Every non-letter character in `data/processed/speech/train.jsonl` is accounted for below —
measured, not assumed. `- '` are word-internal boundaries; the marks in `CLOSING`, `OPENING`
and `QUOTE` map onto the base's rows; `€ & _` and a stray combining dot below occur four times
in 152,478 transcripts between them and **raise**, so the clip is dropped by name rather than
silently mispronounced.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from agbalu.tts.g2p import BOUNDARY, PhonemeError, inventory, phonemize_word
from agbalu.tts.kokoro import FOLD, fold

if TYPE_CHECKING:
    from collections.abc import Mapping

CLOSING: Final[Mapping[str, str]] = {
    ".": ".",
    ",": ",",
    ";": ";",
    ":": ":",
    "!": "!",
    "?": "?",
    "…": "…",
    ")": ")",
    "»": "”",
}
"""Marks that bind to the word they follow. `»` folds onto the curly close quote, which is
the row the base has; there is no guillemet in Kokoro's table."""

OPENING: Final[Mapping[str, str]] = {"(": "(", "«": "“"}
"""Marks that bind to the word they precede."""

QUOTE: Final = '"'
QUOTE_OPEN: Final = "“"
QUOTE_CLOSE: Final = "”"
"""The straight double quote is 550 of the corpus's marks and carries no side. It alternates
within an utterance, which is what the two curly rows the base holds actually distinguish."""

DASH: Final = "—"
DASHES: Final = frozenset("-–—")
"""A dash alone between spaces opens a line of dialogue — 1,400 transcripts begin with one —
and Kokoro carries the em dash as a trained row. Inside a word it is the clitic boundary."""

WORD_SPLIT: Final = re.compile(r"[-']+")
"""Word-internal boundaries. The clitic hyphen is what takes sentence alignment from 46.4% to
99.0% (`docs/tts_design.md` §2), and the apostrophe — 32 occurrences — marks the same kind of
elision, so it is separated rather than dropped or pronounced."""


MARKS: Final[Mapping[str, str]] = {**CLOSING, **OPENING, QUOTE: QUOTE_OPEN}
"""Every character the scanner treats as a mark. The quote's value is a placeholder: which
curly row it becomes depends on how many preceded it, which only `_Line` knows."""


def punctuation() -> frozenset[str]:
    """Every symbol this front end can emit that is not a phoneme."""
    return frozenset({*CLOSING.values(), *OPENING.values(), QUOTE_OPEN, QUOTE_CLOSE, DASH})


TIE_BAR: Final = "\u0361"
"""The one character `agbalu.tts.kokoro.fold` removes. `t` and `ʃ` reach the inventory from
the spirant and plain tables independently, so the affricates cost nothing else."""


def emitted() -> frozenset[str]:
    """The full output alphabet, phonemes and marks, as the folded readings leave it."""
    letters = (inventory() - {TIE_BAR}) | set(FOLD.values())
    return frozenset(letters | punctuation() | {BOUNDARY})


def _reading(word: str, lexicon: Mapping[str, str] | None) -> str:
    """One orthographic word, attested reading first, rule table second."""
    attested = lexicon.get(word.casefold()) if lexicon is not None else None
    return attested if attested else phonemize_word(word)


class _Line:
    """One utterance under construction: rendered words, and the marks not yet bound to one.

    A class rather than a closure because three pieces of state move together — the words
    emitted so far, the opening marks waiting for their word, and whether a straight quote
    opens or closes — and a mark can only be placed once the word beside it is known.
    """

    __slots__ = ("_lexicon", "_pending", "_quoted", "_words")

    def __init__(self, lexicon: Mapping[str, str] | None) -> None:
        self._lexicon = lexicon
        self._words: list[str] = []
        self._pending = ""
        self._quoted = False

    def open(self, mark: str) -> None:
        """A mark that binds to the word after it."""
        self._pending += mark

    def close(self, mark: str) -> None:
        """A mark that binds to the word before it, or waits if there is none yet."""
        if self._words:
            self._words[-1] += mark
        else:
            self._pending += mark

    def quote(self) -> str:
        """The curly quote a straight one means here, alternating within the utterance."""
        mark = QUOTE_CLOSE if self._quoted else QUOTE_OPEN
        self._quoted = not self._quoted
        return mark

    def word(self, letters: str) -> None:
        """One orthographic word, with whatever opening marks were waiting for it."""
        parts = [part for part in WORD_SPLIT.split(letters) if part]
        if not parts:
            message = f"{letters!r} is word boundaries and nothing else"
            raise PhonemeError(message)
        reading = BOUNDARY.join(_reading(part, self._lexicon) for part in parts)
        self._words.append(self._pending + fold(reading))
        self._pending = ""

    def rendered(self, text: str) -> str:
        if not self._words:
            message = f"nothing to phonemise in {text!r}"
            raise PhonemeError(message)
        return BOUNDARY.join(self._words) + self._pending


def _token(line: _Line, token: str) -> None:
    """One whitespace-delimited token, scanned so an unspaced mark still separates words.

    `isental,asemres` is 70 of the corpus's 121 unphonemisable transcripts and is a missing
    space, not a defect: scanning rather than stripping the ends recovers every one of them.
    """
    if set(token) <= DASHES:
        line.open(DASH)
        return
    letters: list[str] = []
    for char in token:
        if char not in MARKS:
            letters.append(char)
            continue
        if letters:
            line.word("".join(letters))
            letters.clear()
        mark = line.quote() if char == QUOTE else MARKS[char]
        if char in OPENING or (char == QUOTE and mark == QUOTE_OPEN):
            line.open(mark)
        else:
            line.close(mark)
    if letters:
        line.word("".join(letters))


def phonemize(text: str, *, lexicon: Mapping[str, str] | None = None) -> str:
    """A sentence to the phoneme string the model trains on, punctuation included.

    Words are separated by a space and marks bind to their word with none, which is the shape
    the base's own front end produces. Raises on any character with no rule: silent deletion
    is the defect this front end exists to remove, and it has already shipped once.
    """
    line = _Line(lexicon)
    for token in text.split():
        _token(line, token)
    return line.rendered(text)
