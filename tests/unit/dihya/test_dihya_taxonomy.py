"""Unit tests for Pan-Amazigh taxonomy and script detection heuristics."""

from __future__ import annotations

import pytest

from agbalu.dihya.taxonomy import (
    AMAZIGH_VARIETIES,
    COMPOSITE_CLASSES,
    SCRIPTS,
    detect_dominant_script,
    is_amazigh_label,
    script_error,
)


def test_taxonomy_members() -> None:
    """Verify core varieties and script mappings are non-empty and unique."""
    assert len(AMAZIGH_VARIETIES) == 15
    assert len(set(AMAZIGH_VARIETIES)) == 15
    assert len(SCRIPTS) == 5
    assert len(COMPOSITE_CLASSES) >= 20
    assert len(set(COMPOSITE_CLASSES)) == len(COMPOSITE_CLASSES)


@pytest.mark.parametrize(
    ("text", "expected_script"),
    [
        ("Azul fell-awen, amek tettilim ass-a?", "Latn"),
        ("Taqbaylit d tutlayt n tmaziɣt n Leqbayel.", "Latn"),
        ("ⴰⵣⵓⵍ ⴼⵍⵍⴰⵡⵏ ⵎⴰⵏⵉⴽ ⴰⵏⵜⴳⴰⵎ?", "Tfng"),
        ("ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⵚⵍⵉⵜ", "Tfng"),
        ("سلام خويا واش راك لاباس بخير؟", "Arab"),
        ("إنجيل س تـشـاويث س لوقا", "Arab"),
        ("wach 3lik 7bibi cava chwiya 9olili?", "Arabizi"),
        ("salam khoya 3afak 3tini 7aja", "Arabizi"),
        ("我們試試看！今天是６月１８号", "Hani"),
    ],
)
def test_detect_dominant_script(text: str, expected_script: str) -> None:
    """Verify script detection accuracy across distinct orthographies."""
    detected = detect_dominant_script(text)
    assert detected == expected_script


@pytest.mark.parametrize(
    ("label", "expected_is_amazigh"),
    [
        ("kab_Latn", True),
        ("shy_Latn", True),
        ("shy_Arab", True),
        ("mzb_Latn", True),
        ("shi_Tfng", True),
        ("zgh_Tfng", True),
        ("taq_Latn", True),
        ("arq_Arab", False),
        ("arq_Arabizi", False),
        ("ary_Arab", False),
        ("fra_Latn", False),
        ("NOT_AMAZIGH", False),
    ],
)
def test_is_amazigh_label(label: str, expected_is_amazigh: bool) -> None:
    """Verify Amazigh vs negative distractor classification."""
    assert is_amazigh_label(label) == expected_is_amazigh


@pytest.mark.parametrize(
    ("text", "label", "valid"),
    [
        ("Azul ʷ ɛ ɣ", "kab_Latn", True),
        ("azul ⴰ", "kab_Latn", False),
        ("azul سلام", "kab_Latn", False),
        ("ⴰⵣⵓⵍ azul", "kab_Tfng", False),
        ("bonjour 123 57", "fra_Latn", True),
        ("salam khoya labas", "arq_Arabizi", True),
        ("α β γ", "kab_Latn", False),
        ("Привет мир", "fra_Latn", False),
        ("سلام،؟", "arq_Arab", True),
        ("ﺳﻼﻡ", "arq_Arab", True),
        ("ⴰⵣⵓⵍ 2026!", "kab_Tfng", True),
        ("", "kab_Latn", False),
        ("123 ?", "kab_Latn", False),
    ],
)
def test_strict_script_validation(text: str, label: str, valid: bool) -> None:
    assert (script_error(text, label) is None) == valid


def test_script_metadata_must_match_label() -> None:
    assert (
        script_error("azul", "kab_Latn", "Tfng") == "declared script 'Tfng' != label script 'Latn'"
    )
