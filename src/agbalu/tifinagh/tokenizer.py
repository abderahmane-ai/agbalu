"""The character inventory the script-conversion model is defined over.

One codepoint, one id. Nothing is merged and nothing is learned, which is the point:
Tifinagh omits the schwa, so restoring it means predicting a character that is not in
the input, and a subword vocabulary that memorises `ⵜⵛⴼⵉⴹ` whole has already thrown
away the boundary the prediction has to be made at.

Case is folded on the way in. The training pairs are lowercase, so an uppercase input
is spelled the same way rather than reaching `[UNK]`; capitalisation is not restored on
the way out and never was.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final


class TokenizerError(Exception):
    """A vocabulary file that cannot be used to decode a checkpoint's output."""


PAD_ID: Final = 0
UNK_ID: Final = 1
BOS_ID: Final = 2
EOS_ID: Final = 3

SPECIAL_IDS: Final = frozenset({PAD_ID, UNK_ID, BOS_ID, EOS_ID})
SPECIAL_TOKENS: Final[tuple[str, ...]] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")

LATIN: Final = " abcdefghijklmnopqrstuvwxyzɣɛḥḍṣṭẓṛčǧţ"
TIFINAGH: Final = "ⴰⴱⴳⴷⴹⴻⴼⴽⵀⵃⵄⵅⵇⵉⵊⵍⵎⵏⵓⵔⵕⵖⵙⵚⵛⵜⵟⵡⵢⵣⵥⵯ"
PUNCTUATION: Final = "0123456789-'.,!?:;\"()/"

VOCAB_CHARS: Final = LATIN + TIFINAGH + PUNCTUATION
"""Kabyle Latin per `docs/orthography.md`, the Neo-Tifinagh repertoire used to write it,
and the punctuation the corpus actually contains. `ModelConfig.vocab_size` is larger than
this on purpose — the embedding is padded, and the spare rows are never emitted."""


class CharTokenizer:
    """Codepoint-to-id mapping, with the four specials first."""

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        for char in VOCAB_CHARS:
            self.char_to_id.setdefault(char, len(self.char_to_id))
        self.id_to_char: dict[int, str] = {index: char for char, index in self.char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """`text` as ids, case-folded, with `[BOS]` and `[EOS]` unless suppressed."""
        ids = [BOS_ID] if add_special_tokens else []
        ids.extend(self.char_to_id.get(char.lower(), UNK_ID) for char in text)
        if add_special_tokens:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Ids back to text.

        With `skip_special_tokens` false the specials come back as their bracketed
        names, which is what makes a truncated or degenerate decode visible in a log.
        """
        pieces: list[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in SPECIAL_IDS:
                continue
            pieces.append(self.id_to_char.get(token_id, ""))
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"char_to_id": self.char_to_id, "vocab_size": self.vocab_size}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        """A saved mapping, rejecting one that is not a `{str: int}` table.

        Checked rather than trusted because the ids are positions in an embedding table:
        a mapping that disagrees with the checkpoint decodes to plausible-looking text in
        the wrong characters, and nothing raises.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = payload.get("char_to_id")
        if not isinstance(table, dict):
            message = f"{path} has no char_to_id table"
            raise TokenizerError(message)
        mapping: dict[str, int] = {}
        for char, index in table.items():
            if not isinstance(char, str) or not isinstance(index, int):
                message = f"{path} maps {char!r} to {index!r}, which is not str -> int"
                raise TokenizerError(message)
            mapping[char] = index
        tokenizer = cls()
        tokenizer.char_to_id = mapping
        tokenizer.id_to_char = {index: char for char, index in mapping.items()}
        return tokenizer
