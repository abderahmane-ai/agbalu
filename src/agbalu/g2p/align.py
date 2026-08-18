"""Word-align the sentence-level Kabyle pronunciation data.

`kabyle-g2p-training-data` pairs a sentence with its IPA, which a TTS front-end cannot
use directly. The IPA side splits hyphenated clitics (`deg-s` is `ðəʝ s`) while the
orthography joins them, so splitting orthography on hyphens too takes the share of
equal-length pairs from 46.4% to 99.0%. Sentences that still disagree are dropped.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

DEFAULT_G2P: Final = Path("data/raw/hf.boffire.kabyle-g2p-training-data/kab_g2p_train.tsv")

_SPLIT: Final = re.compile(r"[\s\-]+")
_STRIP: Final = "«»\"'“”‘’.,;:!?()[]{}…"


class G2PError(Exception):
    """The pronunciation data could not be read or aligned."""


@dataclass(frozen=True, slots=True)
class Alignment:
    """One sentence whose sides carry the same number of tokens."""

    orthography: tuple[str, ...]
    phonemes: tuple[str, ...]


@dataclass
class AlignmentReport:
    sentences: int = 0
    aligned: int = 0
    words: int = 0
    skipped: Counter[int] = field(default_factory=Counter)

    @property
    def rate(self) -> float:
        return self.aligned / self.sentences if self.sentences else 0.0


def orthographic_tokens(text: str) -> list[str]:
    """Whitespace and hyphen split, matching how the IPA side is tokenised."""
    return [token.strip(_STRIP) for token in _SPLIT.split(text) if token.strip(_STRIP)]


def phoneme_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token]


def read_pairs(path: Path = DEFAULT_G2P) -> Iterator[tuple[str, str]]:
    if not path.is_file():
        msg = f"pronunciation data not found: {path}"
        raise G2PError(msg)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            orthography, tab, ipa = line.rstrip("\n").partition("\t")
            if not tab:
                continue
            if orthography.strip() and ipa.strip():
                yield orthography.strip(), ipa.strip()


def align(pairs: Iterator[tuple[str, str]]) -> tuple[list[Alignment], AlignmentReport]:
    report = AlignmentReport()
    aligned: list[Alignment] = []
    for orthography, ipa in pairs:
        report.sentences += 1
        left = orthographic_tokens(orthography)
        right = phoneme_tokens(ipa)
        if len(left) != len(right) or not left:
            report.skipped[len(left) - len(right)] += 1
            continue
        report.aligned += 1
        report.words += len(left)
        aligned.append(Alignment(orthography=tuple(left), phonemes=tuple(right)))
    return aligned, report


@dataclass
class Pronunciations:
    """Word to its attested pronunciations, by frequency."""

    entries: dict[str, Counter[str]]

    @classmethod
    def from_alignments(cls, alignments: list[Alignment]) -> Pronunciations:
        entries: dict[str, Counter[str]] = defaultdict(Counter)
        for alignment in alignments:
            for word, phones in zip(alignment.orthography, alignment.phonemes, strict=True):
                entries[word.casefold()][phones] += 1
        return cls(entries=dict(entries))

    def __len__(self) -> int:
        return len(self.entries)

    def best(self, word: str) -> str | None:
        """The most frequent pronunciation, or none when unattested."""
        readings = self.entries.get(word.casefold())
        if not readings:
            return None
        return readings.most_common(1)[0][0]

    def ambiguous(self) -> dict[str, Counter[str]]:
        """Words with more than one attested pronunciation.

        The rate bounds how far a context-free pronunciation dictionary can get.
        """
        return {word: readings for word, readings in self.entries.items() if len(readings) > 1}
