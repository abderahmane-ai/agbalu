"""Canonical character vocabulary for Feraoun-Pro document OCR.

173 classes: special tokens ([PAD]/blank=0, <s>=1, </s>=2, <unk>=3), standard Latin,
Kabyle extended letters including ţ/Ţ (U+0163/U+0162), Kabyle sub-dots, accented Latin,
digits, punctuation, and Neo-Tifinagh.
"""

from __future__ import annotations

from typing import Final

PAD_TOKEN: Final = "<pad>"
BOS_TOKEN: Final = "<s>"
EOS_TOKEN: Final = "</s>"
UNK_TOKEN: Final = "<unk>"

PAD_ID: Final = 0
BOS_ID: Final = 1
EOS_ID: Final = 2
UNK_ID: Final = 3

_LOWER_LATIN: Final = "abcdefghijklmnopqrstuvwxyz"
_UPPER_LATIN: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

KABYLE_EXTENDED: Final = "ɣƔɛƐčČǧǦţŢ"
KABYLE_SUBDOTS: Final = "ḍḌḥḤṛṚṣṢṭṬẓẒ"

_DIGITS: Final = "0123456789"
_PUNCTUATION: Final = " .,?!«»-—:;'\"/()[]{}%*+=_~–…\n\t"

_TIFINAGH: Final = "ⴰⴱⴳⴷⴹⴻⴼⴽⵀⵃⵄⵅⵇⵉⵊⵍⵎⵏⵓⵔⵕⵙⵚⵛⵜⵟⵡⵢⵣⵥⵖ"
_ACCENTED_LATIN: Final = "éèêëàâîïôùûçÉÈÊËÀÂÎÏÔÙÛÇ"

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

_UNIQUE_CHARS: Final[tuple[str, ...]] = tuple(dict.fromkeys(_INVENTORY))

VOCABULARY: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    *_UNIQUE_CHARS,
)

VOCAB_SIZE: Final[int] = len(VOCABULARY)

_CHAR_TO_ID: Final[dict[str, int]] = {char: idx for idx, char in enumerate(VOCABULARY)}
_ID_TO_CHAR: Final[dict[int, str]] = dict(enumerate(VOCABULARY))


def encode_ctc(text: str) -> list[int]:
    """Encode text for CTC loss: raw character token IDs without sequence delimiters."""
    return [_CHAR_TO_ID.get(char, UNK_ID) for char in text]


def encode_aux(text: str) -> list[int]:
    """Encode text for the auxiliary autoregressive decoder: <s> + characters + </s>."""
    ids = [BOS_ID]
    ids.extend(_CHAR_TO_ID.get(char, UNK_ID) for char in text)
    ids.append(EOS_ID)
    return ids


def decode_ctc(ids: list[int] | tuple[int, ...], blank_id: int = PAD_ID) -> str:
    """Collapse consecutive duplicate predictions and remove CTC blank tokens."""
    collapsed: list[int] = []
    prev = -1
    for token_id in ids:
        if token_id != prev:
            if token_id != blank_id and token_id not in (BOS_ID, EOS_ID):
                collapsed.append(token_id)
            prev = token_id
    return "".join(_ID_TO_CHAR.get(token_id, "") for token_id in collapsed)


def decode_text(ids: list[int] | tuple[int, ...], skip_special_tokens: bool = True) -> str:
    """Decode token IDs directly to string."""
    special = {PAD_ID, BOS_ID, EOS_ID, UNK_ID} if skip_special_tokens else set()
    return "".join(_ID_TO_CHAR.get(token_id, "") for token_id in ids if token_id not in special)


def get_vocab_size() -> int:
    """Return total number of classes in the vocabulary."""
    return VOCAB_SIZE
