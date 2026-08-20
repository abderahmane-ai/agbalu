"""Character-level tokenizer for orthography standardisation and diacritic recovery.

Operates at the character level with fixed token indices so the model vocabulary is
compact (128 slots) and free from subword out-of-vocabulary failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PAD_TOKEN: Final = "<pad>"  # noqa: S105
UNK_TOKEN: Final = "<unk>"  # noqa: S105
BOS_TOKEN: Final = "<s>"  # noqa: S105
EOS_TOKEN: Final = "</s>"  # noqa: S105

SPECIALS: Final = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)

# 128 character inventory
ALPHABET: Final = (
    # Latin letters lowercase
    "abcdefghijklmnopqrstuvwxyz"
    # Latin letters uppercase
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Digits & Arabizi
    "0123456789"
    # Berber canonical diacritics and IPA Latin extensions
    "čČğĞǦǧḍḌṭṬṣṢẓẒṛṚḥḤɛƐɣƔ"
    # Common French accent vowels in Kabyle loanwords/inputs
    "éèêàâîôûëï"
    # Punctuation and spacing
    " .,!?:;-'\"()[]/\\_+=%«»–—\n"
)


class TokenizerError(Exception):
    """A tokenization failure."""


@dataclass(frozen=True, slots=True)
class Tokenizer:
    """Character tokenizer mapping strings to/from token ids."""

    char_to_id: dict[str, int]
    id_to_char: dict[int, str]

    @classmethod
    def build(cls) -> Tokenizer:
        char_to_id: dict[str, int] = {}
        id_to_char: dict[int, str] = {}

        for idx, special in enumerate(SPECIALS):
            char_to_id[special] = idx
            id_to_char[idx] = special

        current_idx = len(SPECIALS)
        for char in ALPHABET:
            if char not in char_to_id:
                char_to_id[char] = current_idx
                id_to_char[current_idx] = char
                current_idx += 1

        return cls(char_to_id=char_to_id, id_to_char=id_to_char)

    @property
    def vocab_size(self) -> int:
        return 128

    @property
    def pad_id(self) -> int:
        return self.char_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.char_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[EOS_TOKEN]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        ids.extend(self.char_to_id.get(char, self.unk_id) for char in text)

        if add_eos:
            ids.append(self.eos_id)

        if max_length is not None:
            ids = ids[:max_length]
            if add_eos and (not ids or ids[-1] != self.eos_id):
                ids[-1] = self.eos_id

        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        for token_id in ids:
            char = self.id_to_char.get(token_id, "")
            if skip_special_tokens and char in SPECIALS:
                continue
            chars.append(char)
        return "".join(chars)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "char_to_id": self.char_to_id,
                    "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        char_to_id = data["char_to_id"]
        id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        return cls(char_to_id=char_to_id, id_to_char=id_to_char)
