"""Kabyle text to the phoneme ids Matoub-82M was trained on.

The rules are a copy of `agbalu.tts.g2p` and `agbalu.tts.kokoro`, not an import: nothing
under `hub/` may import `agbalu`, because the published repository ships without it.
`tests/unit/test_hub_matoub.py` asserts the two agree over the corpus, and it is the only
thing keeping them from drifting.

A symbol with no rule raises. It is never dropped: silent deletion is the defect this
front end exists to remove, and it has already cost three Kabyle consonants once —
Kokoro's own G2P is built with `unk=''` and StyleTTS2's cleaner skips what it cannot find.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Final

from transformers import PreTrainedTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

VOCAB_FILE: Final = "vocab.json"

VOWELS: Final = frozenset("aeiou")

BACKING_TRIGGERS: Final = frozenset("ḍṣṭẓṛqɣx")

SPIRANTS: Final[dict[str, tuple[str, str]]] = {
    "b": ("β", "b"),
    "d": ("ð", "d"),
    "g": ("ʝ", "ɡ"),
    "k": ("ç", "k"),
    "t": ("θ", "t"),
    "ḍ": ("ðˤ", "dˤ"),
}

PLAIN: Final[dict[str, str]] = {
    "a": "æ",
    "c": "ʃ",
    "e": "ə",
    "f": "f",
    "h": "h",
    "i": "i",
    "j": "ʒ",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "u": "u",
    "v": "v",
    "w": "w",
    "x": "χ",
    "y": "j",
    "z": "z",
    "č": "t͡ʃ",
    "ǧ": "d͡ʒ",
    "ɛ": "ʕ",
    "ɣ": "ʁ",
    "ḥ": "ħ",
    "ṛ": "rˤ",
    "ṣ": "sˤ",
    "ṭ": "tˤ",
    "ẓ": "zˤ",
}

NASAL_ASSIMILATION: Final[dict[str, str]] = {"f": "m", "m": "m", "y": "ɲ", "q": "ŋ", "x": "ŋ"}

FOLD: Final[dict[str, str]] = {"t͡ʃ": "ʧ", "d͡ʒ": "ʤ"}
"""Tie-bar sequences the base model already carries as single symbols."""

LEGACY_TENSE_T: Final = "ţ"
LENGTH: Final = "ː"
BOUNDARY: Final = " "

_SPLIT: Final = re.compile(r"[\s\-]+")
_STRIP: Final = "«»\"'“”‘’.,;:!?()[]{}…"


class PhonemeError(ValueError):
    """A character with no rule."""


def _segment(word: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    index = 0
    while index < len(word):
        char = word[index]
        paired = index + 1 < len(word) and word[index + 1] == char and char not in VOWELS
        segments.append((char, paired))
        index += 2 if paired else 1
    return segments


def _backed(chars: list[str], position: int) -> bool:
    before = chars[position - 1] if position > 0 else ""
    after = chars[position + 1] if position + 1 < len(chars) else ""
    return before in BACKING_TRIGGERS or after in BACKING_TRIGGERS


def phonemize_word(word: str) -> str:
    """One orthographic word to IPA. Raises on any character without a rule."""
    segments = _segment(word.casefold())
    chars = [char for char, _ in segments]
    out: list[str] = []
    for position, (char, geminate) in enumerate(segments):
        if char == LEGACY_TENSE_T:
            out.append(SPIRANTS["t"][1] + LENGTH)
            continue
        if char in SPIRANTS:
            short, stop = SPIRANTS[char]
            out.append(stop + LENGTH if geminate else short)
            continue
        if char not in PLAIN:
            message = f"no rule for {char!r} (U+{ord(char):04X}) in {word!r}"
            raise PhonemeError(message)
        if char == "a" and _backed(chars, position):
            symbol = "ɑ"
        elif char == "n" and not geminate and position + 1 < len(chars):
            symbol = NASAL_ASSIMILATION.get(chars[position + 1], PLAIN["n"])
        else:
            symbol = PLAIN[char]
        out.append(symbol + LENGTH if geminate else symbol)
    return "".join(out)


def fold(ipa: str) -> str:
    """Rewrite tie-bar affricates onto the base model's own single symbols."""
    for sequence, symbol in FOLD.items():
        ipa = ipa.replace(sequence, symbol)
    return ipa


def phonemize(text: str) -> str:
    """A Kabyle sentence to the phoneme string the model was fitted on."""
    words = [token.strip(_STRIP) for token in _SPLIT.split(text) if token.strip(_STRIP)]
    return fold(BOUNDARY.join(phonemize_word(word) for word in words))


class MatoubTokenizer(PreTrainedTokenizer):
    """Kabyle Latin orthography in, phoneme ids out.

    Punctuation is dropped and the clitic hyphen is a word boundary, because that is what
    the training transcripts carried: the model has never been supervised on a comma.
    """

    vocab_files_names: dict[str, str] = {"vocab_file": VOCAB_FILE}
    model_input_names: list[str] = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "$",
        bos_token: str = "$",
        eos_token: str = "$",
        **kwargs: Any,
    ) -> None:
        with open(vocab_file, encoding="utf-8") as handle:  # noqa: PTH123
            self._vocab: dict[str, int] = json.load(handle)
        self._ids_to_symbols = {index: symbol for symbol, index in self._vocab.items()}
        super().__init__(pad_token=pad_token, bos_token=bos_token, eos_token=eos_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def phonemize(self, text: str) -> str:
        """The IPA string this tokenizer will encode, for inspection."""
        return phonemize(text)

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        return list(phonemize(text))

    def _convert_token_to_id(self, token: str) -> int:
        index = self._vocab.get(token)
        if index is None:
            message = (
                f"no embedding row for {token!r} (U+{ord(token):04X}); the model would be "
                f"given a phoneme it was never trained on"
            )
            raise KeyError(message)
        return index

    def _convert_id_to_token(self, index: int) -> str:
        symbol: str = self._ids_to_symbols.get(index, "")
        return symbol

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        # `meldataset` wraps every training target in the pad symbol on both sides, so a
        # sequence without them is off the distribution the durations were fitted on.
        boundary = [self._vocab["$"]]
        merged = boundary + token_ids_0 + boundary
        return merged if token_ids_1 is None else merged + token_ids_1 + boundary

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        mask = [1, *([0] * len(token_ids_0)), 1]
        return mask if token_ids_1 is None else mask + [0] * len(token_ids_1) + [1]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        from pathlib import Path

        name = f"{filename_prefix}-{VOCAB_FILE}" if filename_prefix else VOCAB_FILE
        path = Path(save_directory) / name
        path.write_text(
            json.dumps(self._vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return (str(path),)

    @staticmethod
    def write_vocabulary(symbols: Mapping[str, int], directory: Path) -> Path:
        """Write the `vocab.json` a staged release is constructed from."""
        path = directory / VOCAB_FILE
        path.write_text(
            json.dumps(dict(symbols), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return path


__all__ = ["MatoubTokenizer", "PhonemeError", "fold", "phonemize", "phonemize_word"]
