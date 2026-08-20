"""Canonical character vocabulary for Kabyle document OCR (Feraoun).

Recognition is per glyph, so the table is per glyph: character error rate is then a count of
this table's own symbols, with no subword fallback sequences in between.

**`ţ` U+0163 has no slot here and is written 21,058 times in AƔBALU-Text v1**
(`docs/orthography.md` §4). A line carrying it encodes as `<unk>` and cannot be transcribed
correctly. Adding it changes `VOCAB_SIZE` and with it the width of `lm_head`, so it costs a
retrain; the published card discloses the gap, and `tests/unit/test_ocr_vocabulary.py` pins
it so the disclosure fails the day the letter is added.
"""

from __future__ import annotations

from typing import Final

# Special tokens
PAD_TOKEN: Final = "<pad>"
BOS_TOKEN: Final = "<s>"
EOS_TOKEN: Final = "</s>"
UNK_TOKEN: Final = "<unk>"

PAD_ID: Final = 0
BOS_ID: Final = 1
EOS_ID: Final = 2
UNK_ID: Final = 3

# Base Latin letters (lower and upper)
_LOWER_LATIN: Final = "abcdefghijklmnopqrstuvwxyz"
_UPPER_LATIN: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Native Kabyle extended consonants
KABYLE_EXTENDED: Final = "ɣƔɛƐčČǧǦ"

# Sub-dot emphatic and pharyngeal consonants
KABYLE_SUBDOTS: Final = "ḍḌḥḤṛṚṣṢṭṬẓẒ"

# Digits
_DIGITS: Final = "0123456789"

# Common punctuation and typographical marks found in Kabyle literature
_PUNCTUATION: Final = " .,?!«»-—:;'\"/()[]{}%*+=_~–…\n\t"

# Tifinagh basic character inventory (for bilingual/transcribed manuscripts)
_TIFINAGH: Final = "ⴰⴱⴳⴷⴹⴻⴼⴽⵀⵃⵄⵅⵇⵉⵊⵍⵎⵏⵓⵔⵕⵙⵚⵛⵜⵟⵡⵢⵣⵥⵖ"

# Common accented Latin letters found in historical Kabyle publications
_ACCENTED_LATIN: Final = "éèêëàâîïôùûçÉÈÊËÀÂÎÏÔÙÛÇ"

# Construct the deduplicated, deterministic vocabulary
_INVENTORY: Final[str] = (
    _LOWER_LATIN
    + _UPPER_LATIN
    + KABYLE_EXTENDED
    + KABYLE_SUBDOTS
    + _ACCENTED_LATIN
    + _DIGITS
    + _PUNCTUATION
    + _TIFINAGH
)

# Deterministic list of all unique characters
_UNIQUE_CHARS: Final[tuple[str, ...]] = tuple(dict.fromkeys(_INVENTORY))

# Complete token list including special tokens
VOCABULARY: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    *_UNIQUE_CHARS,
)

VOCAB_SIZE: Final[int] = len(VOCABULARY)

# Fast bidirectional lookups
_CHAR_TO_ID: Final[dict[str, int]] = {char: idx for idx, char in enumerate(VOCABULARY)}
_ID_TO_CHAR: Final[dict[int, str]] = dict(enumerate(VOCABULARY))


def encode(text: str, add_special_tokens: bool = True) -> list[int]:
    """Encode a text string into a list of character token IDs."""
    ids: list[int] = []
    if add_special_tokens:
        ids.append(BOS_ID)
    ids.extend(_CHAR_TO_ID.get(char, UNK_ID) for char in text)
    if add_special_tokens:
        ids.append(EOS_ID)
    return ids


def decode(ids: list[int] | tuple[int, ...], skip_special_tokens: bool = True) -> str:
    """Decode a sequence of token IDs back into a Unicode text string."""
    special = {PAD_ID, BOS_ID, EOS_ID, UNK_ID} if skip_special_tokens else set()
    chars: list[str] = []
    for token_id in ids:
        if token_id in special:
            continue
        chars.append(_ID_TO_CHAR.get(token_id, ""))
    return "".join(chars)


def get_vocab_size() -> int:
    """Return the total vocabulary size."""
    return VOCAB_SIZE
