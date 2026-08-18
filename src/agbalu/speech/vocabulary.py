"""The CTC output vocabulary, derived from normalised transcripts.

A CTC head emits one class per frame from a closed set, so the set *is* the
orthography the model can produce. It is built from the corpus rather than
declared, then audited against `agbalu.normalise.rules.ALPHABET`: a letter the
transcripts contain and the reference inventory does not is a defect in the data,
and it is reported rather than silently becoming a class.

The target text is casefolded and stripped of sentence punctuation, neither of
which is recoverable from audio. The hyphen is kept: `docs/orthography.md` §2
classes it as a letter of the writing system carrying clitics, coordination and
compounds, and it is 3.03% of transcript characters.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.normalise.rules import ALPHABET, HYPHEN

WORD_DELIMITER: Final = "|"
"""Wav2Vec2's convention: the space becomes a visible class so a blank-collapsed
frame sequence still carries word boundaries."""

PAD_TOKEN: Final = "[PAD]"
"""CTC's blank symbol. `Wav2Vec2CTCTokenizer` names it `[PAD]` and `Wav2Vec2ForCTC`
reads `pad_token_id` as the blank, so the two must be the same id."""

UNK_TOKEN: Final = "[UNK]"

SPEAKABLE: Final = frozenset({ch for ch in ALPHABET if ch.islower()} | {HYPHEN, " "})
"""What a CTC class may represent: a lowercase Kabyle letter, the hyphen, or a word
boundary. Case is not recoverable from audio and neither is sentence punctuation."""

_SILENT_CATEGORIES: Final = frozenset({"P", "Z", "C"})
"""Unicode general-category initials whose members are dropped without comment:
punctuation, separators, controls. A digit, a currency sign or a foreign letter is
*not* in this set, because each means a transcript the reader cannot speak — an
unexpanded numeral is a defect, not formatting."""


def ctc_target(text: str) -> str:
    """The transcript reduced to what the acoustic model is asked to emit.

    Case-folded, non-speakable characters dropped, whitespace runs collapsed. Input
    of only punctuation yields the empty string, which `corpus` rejects.
    """
    kept = [ch if ch in SPEAKABLE else " " for ch in text.casefold()]
    return " ".join("".join(kept).split())


def unspeakable(text: str) -> Iterable[str]:
    """Characters dropped by `ctc_target` that a reader could not have voiced."""
    return (
        ch
        for ch in text.casefold()
        if ch not in SPEAKABLE and unicodedata.category(ch)[0] not in _SILENT_CATEGORIES
    )


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The CTC classes, and what the transcripts held that could not become one."""

    counts: Mapping[str, int]
    unexpected: Mapping[str, int]

    @property
    def characters(self) -> tuple[str, ...]:
        return tuple(sorted(self.counts))

    def as_mapping(self) -> dict[str, int]:
        """`Wav2Vec2CTCTokenizer`'s `vocab.json`: every class to a contiguous id.

        `[PAD]` takes id 0 so the blank is the default-initialised class, and the
        space is exported as `WORD_DELIMITER` because a bare space cannot be read
        back out of JSON unambiguously.
        """
        tokens = [PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER]
        tokens += [ch for ch in self.characters if ch != " "]
        return {token: index for index, token in enumerate(tokens)}

    def as_dict(self) -> dict[str, object]:
        return {
            "classes": len(self.as_mapping()),
            "characters": "".join(self.characters),
            "counts": {
                ("<space>" if ch == " " else ch): n for ch, n in sorted(self.counts.items())
            },
            "unexpected": dict(sorted(self.unexpected.items())),
            "pad": PAD_TOKEN,
            "unk": UNK_TOKEN,
            "word_delimiter": WORD_DELIMITER,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_mapping(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def encoder(mapping: Mapping[str, int]) -> dict[str, int]:
    """Target characters to class ids.

    The space is mapped onto `WORD_DELIMITER`'s id, which is the whole reason this is a
    function: a target contains spaces and the vocabulary contains `|`, so encoding
    against the raw mapping drops every word boundary and the model never learns one.
    `[PAD]` and `[UNK]` are excluded by length — they are not characters.
    """
    if WORD_DELIMITER not in mapping:
        message = f"vocabulary has no {WORD_DELIMITER!r} class, so words cannot be delimited"
        raise KeyError(message)
    out = {token: index for token, index in mapping.items() if len(token) == 1}
    out[" "] = mapping[WORD_DELIMITER]
    return out


def decode(ids: Iterable[int], mapping: Mapping[str, int]) -> str:
    """Greedy CTC: collapse runs, drop the blank, put the delimiter back as a space.

    Collapsing before dropping the blank is the order that matters — a doubled letter
    is emitted as `l blank l`, so dropping blanks first would turn `ll` into `l`.
    """
    inverse = {index: token for token, index in mapping.items()}
    blank = mapping[PAD_TOKEN]
    out: list[str] = []
    previous: int | None = None
    for index in ids:
        if index not in (previous, blank):
            token = inverse.get(index, "")
            out.append(" " if token in (WORD_DELIMITER, UNK_TOKEN) else token)
        previous = index
    return " ".join("".join(out).split())


def build(targets: Iterable[str], *, sources: Iterable[str] = ()) -> Vocabulary:
    """Take the inventory of `targets`, and audit `sources` for what they lost.

    `targets` are already reduced by `ctc_target`; `sources` are the normalised
    transcripts they came from, so `unexpected` names what was in the data, is not
    a letter of Kabyle, and is not punctuation. Passing no sources skips the audit
    rather than reporting an empty one.
    """
    counts: Counter[str] = Counter()
    for target in targets:
        counts.update(target)

    strange: Counter[str] = Counter()
    for source in sources:
        strange.update(unspeakable(source))

    return Vocabulary(counts=dict(counts), unexpected=dict(strange))
