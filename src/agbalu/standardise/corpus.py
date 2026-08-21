"""The corruption pass that builds the standardisation corpus.

Builds `agbalu/KabStandard` by corrupting canonical text rather than by collecting real
typing, so the pairs are a model of how Kabyle is typed and not a sample of it. What a
score over them measures is recovery from *this* distribution — the card says so, and it
is why the do-nothing baseline is printed beside every result.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

RANDOM_SEED: Final = 42
IDENTITY_RATE: Final = 0.15
"""15% of inputs are kept clean and canonical, teaching the model invariance."""

# Probabilistic replacement tables mapping canonical characters to real-world typing habits
SUBSTITUTIONS: Final[dict[str, list[tuple[str, float]]]] = {
    "ɣ": [("gh", 0.75), ("g", 0.10), ("3", 0.08), ("8", 0.07)],
    "Ɣ": [("Gh", 0.75), ("G", 0.10), ("3", 0.08), ("8", 0.07)],
    "x": [("kh", 0.85), ("k", 0.10), ("5", 0.05)],
    "X": [("Kh", 0.85), ("K", 0.10), ("5", 0.05)],
    "c": [("ch", 0.75), ("c", 0.20), ("sh", 0.05)],
    "C": [("Ch", 0.75), ("C", 0.20), ("Sh", 0.05)],
    "č": [("tch", 0.70), ("ch", 0.20), ("tc", 0.10)],
    "Č": [("Tch", 0.70), ("Ch", 0.20), ("Tc", 0.10)],
    "ğ": [("dj", 0.80), ("j", 0.15), ("g", 0.05)],
    "Ğ": [("Dj", 0.80), ("J", 0.15), ("G", 0.05)],
    "ḍ": [("dh", 0.75), ("d", 0.25)],
    "Ḍ": [("Dh", 0.75), ("D", 0.25)],
    "ṭ": [("th", 0.70), ("t", 0.30)],
    "Ṭ": [("Th", 0.70), ("T", 0.30)],
    "ṣ": [("s", 0.75), ("ss", 0.25)],
    "Ṣ": [("S", 0.75), ("Ss", 0.25)],
    "ẓ": [("z", 0.80), ("zz", 0.20)],
    "Ẓ": [("Z", 0.80), ("Zz", 0.20)],
    "ṛ": [("r", 0.90), ("rr", 0.10)],
    "Ṛ": [("R", 0.90), ("Rr", 0.10)],
    "ḥ": [("h", 0.70), ("7", 0.25), ("hh", 0.05)],
    "Ḥ": [("H", 0.70), ("7", 0.25), ("Hh", 0.05)],
    "ɛ": [("e", 0.35), ("a", 0.30), ("3", 0.25), ("'", 0.10)],
    "Ɛ": [("E", 0.35), ("A", 0.30), ("3", 0.25), ("'", 0.10)],
}


PROB_DIGRAPH_OU: Final = 0.45
PROB_SUBSTITUTION: Final = 0.90
PROB_CLITIC_DROP: Final = 0.50
PROB_PREP_SHORTEN: Final = 0.25


@dataclass(frozen=True, slots=True)
class Pair:
    """One parallel training example for orthography standardisation."""

    source: str
    target: str

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


def _choose_sub(char: str, rng: random.Random) -> str:
    options = SUBSTITUTIONS.get(char)
    if not options:
        return char
    r = rng.random()
    cumulative = 0.0
    for replacement, weight in options:
        cumulative += weight
        if r <= cumulative:
            return replacement
    return options[-1][0]


def corrupt_text(text: str, rng: random.Random) -> str:
    """Simulate real-world French keyboard, SMS, and Arabizi typing from canonical Kabyle."""
    if rng.random() < IDENTITY_RATE:
        return text

    chars: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]

        if char == "u" and rng.random() < PROB_DIGRAPH_OU:
            chars.append("ou")
            i += 1
            continue
        if char == "U" and rng.random() < PROB_DIGRAPH_OU:
            chars.append("Ou")
            i += 1
            continue

        if char in SUBSTITUTIONS and rng.random() < PROB_SUBSTITUTION:
            chars.append(_choose_sub(char, rng))
        else:
            chars.append(char)
        i += 1

    corrupted = "".join(chars)

    # `d-yeffeɣ` is typed `d yeffegh` or `dyeffegh`, so restoring the boundary is part of
    # the task and not only the diacritics.
    if "-" in corrupted and rng.random() < PROB_CLITIC_DROP:
        if rng.random() < PROB_CLITIC_DROP:
            corrupted = corrupted.replace("-", " ")
        else:
            corrupted = corrupted.replace("-", "")

    # `deg taddart` is typed `g taddart`.
    if rng.random() < PROB_PREP_SHORTEN:
        corrupted = re.sub(r"\bdeg\s+", "g ", corrupted)
        corrupted = re.sub(r"\bseg\s+", "s ", corrupted)

    return corrupted


def generate_pairs(
    canonical_sentences: Sequence[str],
    *,
    seed: int = RANDOM_SEED,
) -> Iterator[Pair]:
    """Generate (informal_source, canonical_target) pairs from canonical texts."""
    rng = random.Random(seed)  # noqa: S311
    for target in canonical_sentences:
        target_clean = target.strip()
        if not target_clean:
            continue
        source = corrupt_text(target_clean, rng)
        yield Pair(source=source, target=target_clean)


def save_jsonl(pairs: Sequence[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            pairs.append(Pair(source=data["source"], target=data["target"]))
    return pairs
