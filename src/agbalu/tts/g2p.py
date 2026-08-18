"""Kabyle grapheme-to-phoneme, as a total function of the spelling.

The table was recovered from `agbalu-pronunciations-v1` rather than declared. That
lexicon's IPA side is a deterministic per-word transliteration — no word takes two
readings across 292,921 tokens, `count(ə)` equals `count(<e>)` in 100.00% of 25,642
entries, and `ː` matches a doubled letter in 100.00% — so the mapping it encodes is
recoverable, and recovering it removes the dictionary's out-of-vocabulary hole.

**It is a transliteration, not a transcription.** It does not restore schwa: `ə` is
orthographic `e` relabelled. Schwa restoration is `agbalu.tifinagh`'s problem and it
is measured there at 93.51% positional F1.

Three conditioned rules, each kept only where it beats its own majority baseline:

- Gemination blocks spirantization. A singleton `b d g k t ḍ` is a fricative and a
  geminate is a stop. Exact count agreement on `t`, `b` and `k`; off by one on `g`.
- `a` backs to `ɑ` beside `ḍ ṣ ṭ ẓ ṛ q ɣ x`. Adjacency is necessary — recall is
  100.00% — and 92.44% accurate against an 87.19% always-`æ` baseline.
- `i` and `u` laxing is **not** applied. Its only recoverable condition, a closed
  syllable, scores 75.59% against a 74.99% baseline. Attested words get their real
  allophones from `Pronunciations`; unattested ones get the tense vowel.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Final

VOWELS: Final = frozenset("aeiou")
"""Gemination is a consonant property here: no vowel row in the recovered table
carries a length mark, and doubled vowels transliterate one symbol each."""

BACKING_TRIGGERS: Final = frozenset("ḍṣṭẓṛqɣx")
"""Adjacent to one of these, `a` is `ɑ`. Every `ɑ` in the lexicon has such a
neighbour; the converse does not hold, which is why this is an approximation."""

SPIRANTS: Final[Mapping[str, tuple[str, str]]] = {
    "b": ("β", "b"),
    "d": ("ð", "d"),
    "g": ("ʝ", "ɡ"),
    "k": ("ç", "k"),
    "t": ("θ", "t"),
    "ḍ": ("ðˤ", "dˤ"),
}
"""Singleton form, then the geminate's stop. `ɡ` is U+0261, not ASCII `g`."""

PLAIN: Final[Mapping[str, str]] = {
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

NASAL_ASSIMILATION: Final[Mapping[str, str]] = {
    "f": "m",
    "m": "m",
    "y": "ɲ",
    "q": "ŋ",
    "x": "ŋ",
}
"""`n` takes the place of what follows it. Deterministic in the lexicon: 115 of 115
before `f`, 58 of 58 before `y`, 19 of 19 before `x`, 47 of 51 before `q`, 23 of 24
before `m`. No other following letter moves it."""

LEGACY_TENSE_T: Final = "ţ"
"""U+0163, the Dallet-tradition tense t. The source generator emits nothing for it,
silently, in all 73 of its words. `docs/orthography.md` §4 settles it against
`kabyle-specs` on corpus evidence: of 1,407 word types, 43% have a `tt` counterpart
and 34% a `t` one, against 0.7% for `ṭ`. It is mapped to the geminate on that
plurality. It occurs 0 times in the speech corpus, so no audio can confirm this."""

LENGTH: Final = "ː"

BOUNDARY: Final = " "
"""The lexicon's IPA side splits hyphenated clitics — `deg-s` is `ðəʝ s` — which is
what takes sentence alignment from 46.4% to 99.0%."""

_SPLIT: Final = re.compile(r"[\s\-]+")
_STRIP: Final = "«»\"'“”‘’.,;:!?()[]{}…"


class PhonemeError(ValueError):
    """A character with no rule. Never silently dropped: dropping is the defect this
    table exists to remove, and it is how the source lost `o` in 128 words."""


def inventory() -> frozenset[str]:
    """Every IPA character the table can emit."""
    out = {LENGTH, "ɑ"}
    for symbol in PLAIN.values():
        out.update(symbol)
    for short, stop in SPIRANTS.values():
        out.update(short + stop)
    return frozenset(out)


def unsupported(vocabulary: Mapping[str, int]) -> frozenset[str]:
    """Symbols this table emits that `vocabulary` cannot represent.

    Kokoro's `config.json` is the argument. Passing it rather than restating it here
    keeps one copy of a fact that lives upstream: Kokoro declares `n_token` 178 with
    114 symbols mapped, so a shortfall is assignable to a free row rather than being
    a resize.
    """
    return frozenset(inventory() - set(vocabulary))


def _segment(word: str) -> Iterator[tuple[str, bool]]:
    """Characters, each flagged as geminate. Vowels never pair."""
    index = 0
    while index < len(word):
        char = word[index]
        paired = index + 1 < len(word) and word[index + 1] == char and char not in VOWELS
        yield char, paired
        index += 2 if paired else 1


def _backed(chars: list[str], position: int) -> bool:
    before = chars[position - 1] if position > 0 else ""
    after = chars[position + 1] if position + 1 < len(chars) else ""
    return before in BACKING_TRIGGERS or after in BACKING_TRIGGERS


def phonemize_word(word: str) -> str:
    """One orthographic word to IPA. Raises on any character without a rule."""
    segments = list(_segment(word.casefold()))
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


def tokenize(text: str) -> list[str]:
    """Orthographic words, split on whitespace and the clitic hyphen."""
    return [token.strip(_STRIP) for token in _SPLIT.split(text) if token.strip(_STRIP)]


def phonemize(text: str, *, lexicon: Mapping[str, str] | None = None) -> str:
    """A sentence to IPA, words separated by `BOUNDARY`.

    `lexicon` supplies attested readings — it carries the `i`/`u` allophony the table
    deliberately does not model — and the table covers everything it lacks. An entry
    whose reading is empty is treated as absent.
    """
    words = tokenize(text)
    readings = []
    for word in words:
        attested = lexicon.get(word.casefold()) if lexicon is not None else None
        readings.append(attested if attested else phonemize_word(word))
    return BOUNDARY.join(readings)
