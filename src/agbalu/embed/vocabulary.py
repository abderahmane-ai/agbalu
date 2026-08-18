"""The characters no multilingual encoder holds a token for, and the repair.

Measured over `agbalu.tokenizer.spec.required_chars`, three candidate backbones each
map `ẓ` to `<unk>`; LaBSE also loses `ǧ`, and all three lose the capitals `Ɣ Ɛ Ẓ Ǧ`.
`ẓ` occurs 811 times in TaPaCo's 15,944 Kabyle sentences and stands at 3.94% of a
5,000-row AƔBALU-Text control, so this is a productive consonant, not a rarity.

Neither normalisation form rescues it. SentencePiece applies its own NFKC charmap
before lookup, so NFD `z` + U+0323 recomposes and misses identically — verified on
both forms. Lowercasing does rescue the capitals, which is why the donor for an
uppercase character is its lowercase form whenever that encodes.

`boffire/kabyle-sentence-transformer-mpnet` inherits the defect: it was fine-tuned on
~2.5M Kabyle pairs with `ẓ` as `<unk>` throughout, and its only published metric is
mean cosine similarity, which cannot see it.

`assert_covered` raises rather than warning. An encoder that silently drops a
consonant trains to a healthy loss and returns an embedding space in which `aẓar` and
`aar` are the same string.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from agbalu.tokenizer.spec import required_chars

Encode = Callable[[str], Sequence[int]]
"""Text to token ids, without special tokens. Special tokens are not text: added ones
carry no `<unk>` and would mask the measurement this module exists to make."""

DONORS: Final[Mapping[str, str]] = {
    "ẓ": "zṣ",
    "ǧ": "dj",
    "ḅ": "b",
}
"""Initialisation source for a lowercase character with no token of its own.

Each value is a string whose characters the tokenizer already carries; the new row is
the mean of their embeddings. `ẓ` takes voicing from `z` and emphasis from `ṣ` because
neither alone carries both; `ǧ` takes the affricate from `d` and `j`, which is the
decomposition `rules.ASCII_DIGRAPHS` already records for it.
"""


class VocabularyError(Exception):
    """A required character the tokenizer cannot encode and no donor can initialise."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """What a tokenizer does to Kabyle, measured on real sentences."""

    missing: tuple[str, ...]
    tokens_per_word: float
    unknown_rate: float
    sentences: int
    words: int

    @property
    def clean(self) -> bool:
        return not self.missing and self.unknown_rate == 0.0


def missing_characters(
    encode: Encode, unk_id: int, chars: Iterable[str] | None = None
) -> tuple[str, ...]:
    """Required characters that encode to `<unk>`, in the order `required_chars` gives."""
    inventory = required_chars() if chars is None else "".join(chars)
    return tuple(char for char in inventory if unk_id in encode(char))


def coverage(encode: Encode, unk_id: int, sentences: Sequence[str]) -> Coverage:
    """Fertility and unknown rate over `sentences`, beside the missing inventory.

    `tokens_per_word` is tokens divided by whitespace-delimited words, the same
    denominator `agbalu.llm.fertility` uses, so the numbers compare across phases.
    """
    words = sum(len(sentence.split()) for sentence in sentences)
    tokens = 0
    unknown = 0
    for sentence in sentences:
        ids = encode(sentence)
        tokens += len(ids)
        unknown += sum(1 for token in ids if token == unk_id)
    return Coverage(
        missing=missing_characters(encode, unk_id),
        tokens_per_word=tokens / words if words else 0.0,
        unknown_rate=unknown / tokens if tokens else 0.0,
        sentences=len(sentences),
        words=words,
    )


def donor_map(missing: Sequence[str], encodable: Callable[[str], bool]) -> dict[str, str]:
    """Where each missing character's embedding is initialised from.

    An uppercase character takes its lowercase form when that encodes, which covers
    `Ɣ Ɛ Ḍ Ṭ` without a table. When the lowercase is missing too — `Ẓ` over `ẓ`, `Ǧ`
    over `ǧ` — the pair shares the lowercase's declared donor rather than the capital
    failing on its own. A donor the tokenizer cannot encode either is a refusal, not a
    fallback to random initialisation.
    """
    chosen: dict[str, str] = {}
    for char in missing:
        lower = char.lower()
        if lower != char and encodable(lower):
            chosen[char] = lower
            continue
        donor = DONORS.get(char) or DONORS.get(lower)
        if donor is None:
            message = f"no donor declared for {char!r} (U+{ord(char):04X})"
            raise VocabularyError(message)
        unusable = [part for part in donor if not encodable(part)]
        if unusable:
            message = f"donor {donor!r} for {char!r} is itself unencodable: {unusable}"
            raise VocabularyError(message)
        chosen[char] = donor
    return chosen


def assert_covered(encode: Encode, unk_id: int) -> None:
    """Refuse a tokenizer that still drops a required character."""
    missing = missing_characters(encode, unk_id)
    if missing:
        listed = " ".join(f"{char} (U+{ord(char):04X})" for char in missing)
        message = f"tokenizer maps {len(missing)} required characters to <unk>: {listed}"
        raise VocabularyError(message)
