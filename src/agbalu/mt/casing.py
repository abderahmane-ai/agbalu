"""Sending a heading to the model in the case it was trained on.

Plain-text books set headings in capitals, and NLLB was not fitted on capitals. FLORES+ —
the sentences every published number in this project was measured on — is sentence case
with proper nouns capitalised, so `CHAPTER III. ATTACK BY STRATAGEM` is off the training
distribution in a way `Chapter III. Attack By Stratagem` is not. The damage shows up as
half-translated and repeating headings: that line came back `AƔAS wis III. ATTAY S
STRATAGEM`, and `CHAPTER VI` came back `TAGURT VI. TAMURT TAMERFUT D TAMARFUT`.

**985 lines of the staged corpus are shouted** — 643 in Aesop's fables, 125 in Grimm, 86 in
*Dracula* — against 4 genuine off-target decodes in 10,551 sentences, so this is where the
visible quality is lost, and it costs no decoding change to fix.

Title case rather than sentence case, deliberately. A shouted line carries no case
information to preserve, so something has to be assumed, and lowercasing `JONATHAN HARKER`
to `Jonathan harker` costs the model the signal that it is a name — where capitalising a
function word to `By` costs almost nothing. Roman numerals and short acronyms keep their
capitals, because `Iii` and `Nyc` are not what those tokens are.

The output is shouted again on the way back, so the document keeps its own typography: the
fold is for the encoder, not for the reader.
"""

from __future__ import annotations

import re
from typing import Final

WORD: Final = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
"""Letter runs of two or more. Single letters are excluded because an initial is upper case
in every case convention, so `J. Amrouche` would read as shouted."""

MIN_SHOUTED_WORDS: Final = 2
"""Below this there is nothing to distinguish a shouted line from a capitalised word.
A lone `CHAPTER` is left alone; so is a name at the head of a paragraph."""

ROMAN: Final = re.compile(r"^[IVXLCDM]+$")
DOTTED: Final = re.compile(r"^(?:[^\W\d_]\.){2,}$", re.UNICODE)
"""A dotted initialism — `N.Y.`, `M.D.`, `D.Ph.` — which is not a word whose case says
anything about the sentence. Length alone was tried first and is wrong: it kept `BY`, `CITY`
and `END` in capitals, because a four-letter word is a word far more often than an acronym."""


def shouted(text: str) -> bool:
    """Whether `text` is set in capitals rather than merely beginning with one."""
    words = WORD.findall(text)
    return len(words) >= MIN_SHOUTED_WORDS and all(word.isupper() for word in words)


def _fold_word(word: str) -> str:
    """One whitespace-separated token, title-cased unless it is a numeral or an initialism."""
    stripped = word.strip("\"'()[]“”‘’«»")
    letters = "".join(character for character in word if character.isalpha())
    if not letters or ROMAN.match(letters) or DOTTED.match(stripped):
        return word
    for position, character in enumerate(word):
        if character.isalpha():
            return word[:position] + character.upper() + word[position + 1 :].lower()
    return word


def soften(text: str) -> str:
    """`text` with its capitals folded to title case, numerals and initialisms untouched.

    A no-op on anything not shouted, which is what makes it safe to apply to every segment
    of a document rather than only the ones a caller believes are headings. Without that
    guard it title-cases ordinary prose — `Attack by stratagem` becomes `Attack By
    Stratagem` — and quietly rewrites the whole input instead of the 985 lines that need it.
    """
    if not shouted(text):
        return text
    return re.sub(r"\S+", lambda match: _fold_word(match.group()), text)


def restore(source: str, translation: str) -> str:
    """`translation` shouted again when `source` was, so the document's typography survives.

    Keyed on the source rather than on a flag carried alongside it: the two can then not
    disagree, and a caller that forgets to thread the flag through gets the source's own
    case rather than a silently wrong one.
    """
    return translation.upper() if shouted(source) else translation
