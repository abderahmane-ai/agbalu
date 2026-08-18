"""Folding a document's typography onto what NLLB's vocabulary can represent.

NLLB's SentencePiece model has no entry for the curly quotes, angle quotes and long dashes
that typeset prose is written with, so each one reaches the encoder as `<unk>`. On the
Project Gutenberg text of *Alice's Adventures in Wonderland* that is 3,127 of 41,390 source
tokens — 7.55%, with 67.1% of segments carrying at least one — against 0 after this fold.
The characters are not rare marks in an unusual document; they are how quoted dialogue is
written, so a novel is mostly affected and a FLORES+ sentence is not.

Underscore emphasis is a Gutenberg convention rather than content. It is removed rather
than restored: putting the markers back would need an alignment between source and
translation that a sentence-level model does not produce.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

FOLD: Final[Mapping[str, str]] = {
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "«": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "»": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "❝": '"',  # HEAVY DOUBLE TURNED COMMA QUOTATION MARK ORNAMENT
    "❞": '"',  # HEAVY DOUBLE COMMA QUOTATION MARK ORNAMENT
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "—": "-",  # EM DASH
    "–": "-",  # EN DASH
    "⁃": "-",  # HYPHEN BULLET
}
"""Every key tokenizes to `<unk>` under `facebook/nllb-200-*`; every value does not.

Measured against the tokenizer rather than assumed, and re-measured by
`tests/integration/test_mt_typography_vocabulary.py`. Adding a key that the vocabulary can
already represent makes the fold a silent rewrite of the document, so the test asserts both
directions."""

_TABLE: Final = str.maketrans(dict(FOLD))

EMPHASIS: Final = re.compile(r"(?<![^\W_])_([^_\n]{1,200})_(?![^\W_])")
"""Project Gutenberg's `_emphasis_`, bounded to one line so an unpaired marker cannot
swallow a paragraph."""

LETTER: Final = re.compile(r"[^\W\d_]", re.UNICODE)


def fold(text: str) -> str:
    """`text` with every unrepresentable mark replaced by the ASCII it stands for."""
    return text.translate(_TABLE)


def strip_emphasis(text: str) -> str:
    """`text` with Gutenberg's underscore emphasis markers removed."""
    return EMPHASIS.sub(r"\1", text)


def prepare(text: str) -> str:
    """A document as it should reach the tokenizer: folded, and free of emphasis markers."""
    return fold(strip_emphasis(text))


def translatable(text: str) -> bool:
    """Whether a unit holds anything a translation model can act on.

    A scene break — `*      *      *` — and a rule of dashes carry no words, so sending them
    is a request for the model to invent one. Both are copied through instead.
    """
    return LETTER.search(text) is not None
