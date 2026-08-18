from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.normalise.rules import ALPHABET, DEFAULT_RULES, RulesError, load_rules

MINIMAL = """
version: "1.0.0"
primary:
  - from: "ε"
    to: "ɛ"
    confidence: certain
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "homoglyphs.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_the_shipped_table() -> None:
    rules = load_rules(DEFAULT_RULES)
    assert rules.version
    assert rules.homoglyphs["ε"] == "ɛ"


def test_shipped_table_preserves_t_cedilla() -> None:
    rules = load_rules(DEFAULT_RULES)
    assert "ţ" in rules.preserved_chars
    assert "ţ" not in rules.homoglyphs


def test_shipped_table_covers_all_six_primary_families() -> None:
    rules = load_rules(DEFAULT_RULES)
    for bad in "εΣγΓԐԑ":
        assert bad in rules.homoglyphs, f"missing primary homoglyph {bad!r}"


def test_multi_character_from_expands_to_each_character(tmp_path: Path) -> None:
    text = MINIMAL + '\ndiacritics:\n  - { from: "áàâ", to: "a" }\n'
    rules = load_rules(write(tmp_path, text))
    assert rules.diacritics == {"á": "a", "à": "a", "â": "a"}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="not found"):
        load_rules(tmp_path / "absent.yaml")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="not valid YAML"):
        load_rules(write(tmp_path, "version: '1'\n  bad: [unclosed\n"))


def test_non_mapping_raises(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="must be a YAML mapping"):
        load_rules(write(tmp_path, "- a\n- b\n"))


def test_a_chained_rule_is_rejected(tmp_path: Path) -> None:
    """`a -> b` plus `b -> c` would make one pass differ from two."""
    text = 'version: "1.0.0"\nprimary:\n  - { from: "a", to: "b" }\n  - { from: "b", to: "c" }\n'
    with pytest.raises(RulesError, match="not confluent"):
        load_rules(write(tmp_path, text))


def test_rewriting_a_preserved_character_is_rejected(tmp_path: Path) -> None:
    text = (
        'version: "1.0.0"\n'
        'primary:\n  - { from: "ţ", to: "ṭ" }\n'
        'preserve:\n  - { char: "ţ", reason: spirantised t }\n'
    )
    with pytest.raises(RulesError, match="rewrites a character marked preserve"):
        load_rules(write(tmp_path, text))


def test_alphabet_contains_every_kabyle_specific_letter() -> None:
    for char in "čḍɛǧɣḥṛṣṭẓ":
        assert char in ALPHABET
        assert char.upper() in ALPHABET


def test_alphabet_excludes_characters_the_spec_rejects() -> None:
    for char in "çñßøæœþ":
        assert char not in ALPHABET


def test_alphabet_keeps_loanword_letters() -> None:
    # p/o/v are not native but are attested in real words; stripping them is wrong.
    for char in "pov":
        assert char in ALPHABET
