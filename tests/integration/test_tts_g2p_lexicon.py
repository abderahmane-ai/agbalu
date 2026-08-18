"""The G2P table against the published lexicon, not against a fixture.

A fixture satisfies the rule it was written for. These assertions run against
`data/processed/lexicon/agbalu-pronunciations-v1.jsonl`, which is where a rule can
actually fail, and they pin the rates `docs/tts_design.md` §2 and the KabLex card quote.

The table and the lexicon are two derivations of one mapping, so agreement between them
is exact once vowel allophony is set aside. Anything less means one of the two drifted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from agbalu.tts.g2p import PhonemeError, phonemize_word

pytestmark = pytest.mark.integration

LEXICON: Final = Path("data/processed/lexicon/agbalu-pronunciations-v1.jsonl")

OUTSIDE_THE_WRITING_SYSTEM: Final = frozenset(
    {"3d", "mp3", "androïd", "rosé", "supermarché", "muḥ€nd", "xelleṣ̣", "ṭeyyeb‟"}
)
"""Entries carrying a digit, a currency sign, a stray combining mark or a curly quote.
Named rather than counted: a length or inventory filter here would delete the most
frequent words in Kabyle (CLAUDE.md §6.1)."""

REPAIRED_ENTRIES: Final = 199
"""Characters the upstream generator dropped: `o` in 128 words, `ţ` in 73."""


def _entries() -> list[dict[str, object]]:
    if not LEXICON.is_file():
        pytest.skip(f"{LEXICON} not built")
    with LEXICON.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _fold(ipa: str, *, backing: bool) -> str:
    folded = ipa.replace("ɪ", "i").replace("ʊ", "u")
    return folded.replace("ɑ", "æ") if backing else folded


@pytest.fixture(scope="module")
def scored() -> tuple[int, int, int, int]:
    exact = laxing = both = total = 0
    for record in _entries():
        word, attested = str(record["word"]), str(record["ipa"])
        try:
            derived = phonemize_word(word)
        except PhonemeError:
            continue
        total += 1
        exact += derived == attested
        laxing += _fold(derived, backing=False) == _fold(attested, backing=False)
        both += _fold(derived, backing=True) == _fold(attested, backing=True)
    return exact, laxing, both, total


def test_the_table_reproduces_the_lexicon_exactly(scored: tuple[int, int, int, int]) -> None:
    """Set aside the two allophonies the table declines to model and agreement is total.

    This is the invariant, not a score: a single disagreement means the table and the
    published readings have diverged.
    """
    _, _, both, total = scored
    assert both == total


def test_exact_match_as_published(scored: tuple[int, int, int, int]) -> None:
    exact, _, _, total = scored
    assert exact / total == pytest.approx(0.7821, abs=0.005)


def test_ignoring_lax_allophony(scored: tuple[int, int, int, int]) -> None:
    """The gap to the line above is `i`/`u`, which scores 75.59% against a 74.99%
    baseline and is therefore left to the lexicon rather than modelled."""
    _, laxing, _, total = scored
    assert laxing / total == pytest.approx(0.9579, abs=0.005)


def test_no_entry_is_missing_a_symbol_the_table_emits() -> None:
    """The deletion defect, gone. Every reading is at least as long as the reference."""
    for record in _entries():
        word, attested = str(record["word"]), str(record["ipa"])
        if word in OUTSIDE_THE_WRITING_SYSTEM:
            continue
        assert len(attested) >= len(phonemize_word(word)), word


def test_repaired_entries_are_flagged_and_counted() -> None:
    flagged = [r for r in _entries() if r["repaired"]]
    assert len(flagged) == REPAIRED_ENTRIES
    assert all(set(str(r["word"])) & set("oţ") for r in flagged)


def test_t_cedilla_and_o_survive_into_the_readings() -> None:
    """`ţ` is real Kabyle — `docs/orthography.md` §4 — and was deleted in all 73 words."""
    readings = {str(r["word"]): str(r["ipa"]) for r in _entries()}
    assert readings["aţan"] == "ætːæn"
    assert readings["bob"] == "βoβ"
    assert readings["london"] == "lonðon"


def test_only_the_named_entries_have_no_rule() -> None:
    refused = {str(r["word"]) for r in _entries() if _refuses(str(r["word"]))}
    assert refused == OUTSIDE_THE_WRITING_SYSTEM


def _refuses(word: str) -> bool:
    try:
        phonemize_word(word)
    except PhonemeError:
        return True
    return False
