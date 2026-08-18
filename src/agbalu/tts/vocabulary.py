"""The token table Matoub is trained against, and the refusal that keeps it honest.

Kokoro maps 114 symbols across 178 embedding rows: index 0 is pad and 63 rows are unused.
Kabyle's inventory is short exactly three of them — `ħ`, `ʕ` and `ˤ` — so each takes a free
row and the embedding matrix is never resized. `agbalu.tts.kokoro.fold` is what takes the
shortfall from four to three, by rewriting the tie-bar affricates onto symbols the table
already carries.

**The refusal is the point.** StyleTTS2's `TextCleaner` maps the symbols it recognises and
drops the rest without a word, and it does this in the *training* path — an unmapped symbol
does not fail, it leaves the target silently, and the model is fitted to a phoneme string
the corpus never held. `encode` raises instead, naming every symbol it could not place.
This project has been bitten by that defect twice already at other layers: the published
lexicon deleted `o` in 128 words and `ţ` in 73, and `mms-tts-kab`'s front end deletes `o`,
`p` and `v` in 21 of 1,000 prompts.

The table is vendored in `resources/kokoro_vocabulary.json` from Kokoro's own `config.json`,
pinned to the revision recorded there. It is checked against upstream where the two actually
meet rather than in a test that would need the network: `modal_app.matoub.matoub_prepare`
downloads that config anyway, and refuses to train if it no longer carries this mapping or if
any of the three assigned rows has since been filled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

TABLE: Final = Path("resources/kokoro_vocabulary.json")

PLACEHOLDER_BASE: Final = 0xE000
"""First Private Use Area codepoint. Rows no symbol claims are filled from here, so the
symbol list stays indexable without any filler colliding with a phoneme."""

PLBERT_LIMIT: Final = 510
"""Longest token sequence a sample may carry.

PL-BERT is a 512-position encoder and the recipe's own guidance is to filter above 510,
leaving the two special positions. A longer sample does not raise inside training; it
overflows, so the cap is applied when the lists are built.
"""


class VocabularyError(Exception):
    """A phoneme the base model has no row for, or a table that cannot be trusted."""


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Kokoro's token table, plus the rows this project assigned to Kabyle."""

    symbols: Mapping[str, int]
    assigned: Mapping[str, int]
    n_token: int
    pad_index: int

    def __post_init__(self) -> None:
        indices = sorted(self.symbols.values())
        if len(set(indices)) != len(indices):
            message = "two symbols share an index, so the table is not a mapping"
            raise VocabularyError(message)
        out_of_range = [i for i in indices if not 0 <= i < self.n_token]
        if out_of_range:
            message = f"indices outside 0..{self.n_token - 1}: {out_of_range}"
            raise VocabularyError(message)
        if self.pad_index in set(self.symbols.values()) - {self.pad_index}:
            message = f"pad index {self.pad_index} is also a symbol's index"
            raise VocabularyError(message)
        stolen = {
            symbol: index
            for symbol, index in self.assigned.items()
            if self.symbols.get(symbol) != index
        }
        if stolen:
            message = f"assigned rows absent from the merged table: {stolen}"
            raise VocabularyError(message)

    @classmethod
    def load(cls, path: Path = TABLE) -> Vocabulary:
        if not path.is_file():
            message = f"token table not found: {path}"
            raise VocabularyError(message)
        payload = json.loads(path.read_text(encoding="utf-8"))
        pad = payload["pad"]
        base: dict[str, int] = dict(payload["base"])
        assigned: dict[str, int] = dict(payload["assigned"])

        collisions = sorted(set(base.values()) & set(assigned.values()))
        if collisions:
            message = (
                f"assigned rows {collisions} are already Kokoro's, so the fine-tune would "
                f"overwrite a trained embedding"
            )
            raise VocabularyError(message)
        overwritten = sorted(set(base) & set(assigned))
        if overwritten:
            message = f"assigned symbols already in the base table: {overwritten}"
            raise VocabularyError(message)

        return cls(
            symbols={pad["symbol"]: pad["index"], **base, **assigned},
            assigned=assigned,
            n_token=int(payload["n_token"]),
            pad_index=int(pad["index"]),
        )

    def symbol_list(self) -> tuple[str, ...]:
        """The table as StyleTTS2 indexes it: one symbol per row, in index order.

        This is the surgery. The recipe ships a table whose rows 7, 8 and 26 hold Private
        Use Area placeholders and which has no entry for `ħ`, `ʕ` or `ˤ` at all — and its
        `TextCleaner` skips what it cannot find, so running it unmodified would delete
        three Kabyle consonants from every training target without a word. Rendering the
        list from this table instead is what puts them in the model.

        Unclaimed rows keep a PUA character, as the recipe's own generator does: a row has
        to hold something for the list to be indexable, and a PUA codepoint cannot collide
        with a phoneme.
        """
        by_index = {index: symbol for symbol, index in self.symbols.items()}
        rendered: list[str] = []
        filler = 0
        for index in range(self.n_token):
            symbol = by_index.get(index)
            if symbol is None:
                symbol = chr(PLACEHOLDER_BASE + filler)
                filler += 1
            rendered.append(symbol)
        return tuple(rendered)

    def free(self) -> tuple[int, ...]:
        """Embedding rows no symbol claims — what a further language would draw from."""
        taken = set(self.symbols.values())
        return tuple(index for index in range(self.n_token) if index not in taken)

    def unmapped(self, ipa: str) -> tuple[str, ...]:
        """Every distinct symbol in `ipa` the table cannot place, in first-seen order."""
        seen: dict[str, None] = {}
        for symbol in ipa:
            if symbol not in self.symbols:
                seen.setdefault(symbol, None)
        return tuple(seen)

    def encode(self, ipa: str) -> tuple[int, ...]:
        """Token ids for a phoneme string, raising on anything the table cannot place.

        Never a filter. The recipe's own cleaner skips what it does not recognise, which
        turns a front-end defect into a quietly wrong training target.
        """
        missing = self.unmapped(ipa)
        if missing:
            named = ", ".join(f"{symbol!r} U+{ord(symbol):04X}" for symbol in missing)
            message = f"no embedding row for {named} in {ipa!r}"
            raise VocabularyError(message)
        return tuple(self.symbols[symbol] for symbol in ipa)
