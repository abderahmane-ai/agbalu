"""The evaluation sets withheld from continued pretraining (task 11.2).

Three languages, because the exit criterion is two-sided: Jugurtha must be better at Kabyle
*and* no worse at English and French, and a retention claim needs a number measured on text
the added blocks never trained on. The English and French sides come from the aligned
sources, so each set is drawn from the same distribution the model will be trained on —
which is what makes a before-and-after perplexity comparable.

A document is selected when **its own text** is held out — the Kabyle sentence for a
monolingual row, the English or French side for an aligned one. `mixture` then drops any
record with a held-out side, which is the wider condition, so every sentence written here is
absent from the corpus however many partners it has there. Nothing here has to be built
before the corpus or kept in step with it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.llm.corpus import LANGUAGE_TAG, CorpusError, Source, is_held_out, records

MIN_CHARS: Final = 40
MAX_CHARS: Final = 1000
"""Titles and fragments distort a perplexity as much as they distort a fertility, and the
upper bound keeps every document inside the 512-token window it is scored in whole."""

CAP: Final = 2000
"""Documents per source per language. A cap rather than everything selected: the Kabyle
split alone is 30,468 records, far more than a perplexity needs, and the excess would cost
GPU minutes for no extra precision. Everything selected still leaves training, whether or
not it is evaluated — the cap bounds the measurement, never the exclusion."""


@dataclass(frozen=True, slots=True)
class Tally:
    """What one source contributed to one language."""

    source: str
    language: str
    documents: int
    characters: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "language": self.language,
            "documents": self.documents,
            "characters": self.characters,
        }


@dataclass(frozen=True, slots=True)
class Holdout:
    """Every evaluation set, and the split that produced them."""

    rate: int
    cap: int
    tallies: tuple[Tally, ...]
    paths: tuple[Path, ...]

    def documents(self, language: str) -> int:
        return sum(t.documents for t in self.tallies if t.language == language)

    def as_dict(self) -> dict[str, object]:
        return {
            "rate": self.rate,
            "cap": self.cap,
            "min_chars": MIN_CHARS,
            "max_chars": MAX_CHARS,
            "sets": [t.as_dict() for t in self.tallies],
            "files": [str(p) for p in self.paths],
        }


def eligible(text: str) -> bool:
    return MIN_CHARS <= len(text) <= MAX_CHARS


def short_code(language: str) -> str:
    """`kab_Latn` back to `kab`, so a filename is readable."""
    for short, code in LANGUAGE_TAG.items():
        if code == language:
            return short
    message = f"no short code for {language!r}"
    raise CorpusError(message)


def build(sources: Sequence[Source], out: Path, *, rate: int, cap: int = CAP) -> Holdout:
    """Write one file per language, and report what each source gave each of them."""
    if cap < 1:
        message = f"cap must be positive, got {cap}"
        raise CorpusError(message)

    out.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, set[str]] = {}
    tallies: list[Tally] = []

    for source in sources:
        taken: dict[str, int] = {}
        characters: dict[str, int] = {}
        for record in records(source):
            language = record.code if record.aligned else LANGUAGE_TAG["kab"]
            text = record.other if record.aligned else record.kabyle
            if not is_held_out(text, rate=rate):
                continue
            if not eligible(text) or taken.get(language, 0) >= cap:
                continue
            if text in seen.setdefault(language, set()):
                continue
            seen[language].add(text)
            selected.setdefault(language, []).append(
                {"text": text, "language": language, "source": source.name}
            )
            taken[language] = taken.get(language, 0) + 1
            characters[language] = characters.get(language, 0) + len(text)
        tallies.extend(
            Tally(source.name, language, count, characters[language])
            for language, count in sorted(taken.items())
        )

    paths: list[Path] = []
    for language, rows in sorted(selected.items()):
        path = out / f"heldout-{short_code(language)}.jsonl"
        with path.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths.append(path)

    return Holdout(rate=rate, cap=cap, tallies=tuple(tallies), paths=tuple(paths))


def read(path: Path) -> list[str]:
    """The documents of one evaluation set, in file order."""
    if not path.is_file():
        message = f"evaluation set not found: {path}"
        raise CorpusError(message)
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("text")
            if not isinstance(value, str) or not value.strip():
                message = f"{path}:{number} carries no text"
                raise CorpusError(message)
            texts.append(value)
    if not texts:
        message = f"evaluation set is empty: {path}"
        raise CorpusError(message)
    return texts
