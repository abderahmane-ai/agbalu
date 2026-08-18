"""Load and validate `resources/homoglyphs.yaml` into typed rule tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_RULES: Final = Path("resources/homoglyphs.yaml")

# The 33-letter INALCO-standardised Kabyle alphabet, per kabyle-orthography-specs
# v0.2. Every letter below is attested in a census of 288,881,863 characters of
# Kabyle running text. The converse does not hold: `ţ` U+0163 is attested 21,058
# times and is deliberately absent, because INALCO does not standardise it. Code
# that means "is this letter Kabyle?" must consult `preserved_chars` as well.
BASE_LETTERS: Final = "abcdefghijklmnqrstuwxyz"
SPECIFIC_LETTERS: Final = "čḍɛǧɣḥṛṣṭẓ"
LOANWORD_LETTERS: Final = "pov"
"""Not in the native inventory; attested only in borrowings and proper nouns.

Kept rather than stripped — 250,088 occurrences of `p` alone are real words.
"""

ALPHABET: Final = frozenset(
    BASE_LETTERS
    + BASE_LETTERS.upper()
    + SPECIFIC_LETTERS
    + SPECIFIC_LETTERS.upper()
    + LOANWORD_LETTERS
    + LOANWORD_LETTERS.upper()
)

HYPHEN: Final = "-"
"""Obligatory in Kabyle: clitics, coordination, compounds. Never stripped."""

ASCII_DIGRAPHS: Final = ("ch", "dj", "gh", "th", "dh", "sh", "zh", "rh")
"""Informal substitutes for č ǧ ɣ ṭ ḍ ṣ ẓ ṛ.

Never substituted automatically: `ch` collides with French *chose*, `gh` with
English *ghoul*, `dj` with Arabic *Djamel*.
"""


class RulesError(Exception):
    """The homoglyph table is missing, unparseable, or internally inconsistent."""


class Mapping(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    from_: str = Field(alias="from", min_length=1)
    to: str
    confidence: str = "certain"


class Preserved(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    char: str = Field(min_length=1, max_length=4)
    reason: str = ""
    action: str = ""


class Rejected(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    char: str = Field(min_length=1, max_length=4)
    name: str = ""


class RuleSet(BaseModel):
    """Every substitution the normaliser may perform, plus what it must not touch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    homoglyphs: dict[str, str]
    """Single-codepoint substitutions applied unconditionally."""

    diacritics: dict[str, str]
    """Foreign accents folded to their base letter. Opt-in; lossy for proper nouns."""

    whitespace: dict[str, str]
    punctuation: dict[str, str]
    preserved: tuple[Preserved, ...]
    rejected: tuple[Rejected, ...]

    @property
    def preserved_chars(self) -> frozenset[str]:
        return frozenset(p.char for p in self.preserved)

    @property
    def rejected_chars(self) -> frozenset[str]:
        return frozenset(r.char for r in self.rejected)


def _pairs(entries: list[dict[str, Any]] | None) -> dict[str, str]:
    table: dict[str, str] = {}
    for raw in entries or []:
        mapping = Mapping.model_validate(raw)
        # A `from` of several characters is shorthand for "each of these".
        for char in mapping.from_:
            table[char] = mapping.to
    return table


def load_rules(path: Path = DEFAULT_RULES) -> RuleSet:
    """Read the homoglyph table.

    Raises:
        RulesError: the file is missing, is not a YAML mapping, or defines a
            substitution whose target is itself a substitution source (which would
            make normalisation order-dependent and non-idempotent).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"homoglyph table not found: {path}"
        raise RulesError(msg) from exc

    try:
        document: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"homoglyph table is not valid YAML: {path}"
        raise RulesError(msg) from exc

    if not isinstance(document, dict):
        msg = f"homoglyph table must be a YAML mapping: {path}"
        raise RulesError(msg)

    homoglyphs = {**_pairs(document.get("primary")), **_pairs(document.get("secondary"))}
    diacritics = _pairs(document.get("diacritics"))
    whitespace = _pairs(document.get("whitespace"))
    punctuation = _pairs(document.get("punctuation"))

    rules = RuleSet(
        version=str(document.get("version", "0.0.0")),
        homoglyphs=homoglyphs,
        diacritics=diacritics,
        whitespace=whitespace,
        punctuation=punctuation,
        preserved=tuple(Preserved.model_validate(p) for p in document.get("preserve", [])),
        rejected=tuple(Rejected.model_validate(r) for r in document.get("reject", [])),
    )
    _check_confluent(rules)
    return rules


def _check_confluent(rules: RuleSet) -> None:
    """Reject a table whose output could itself be rewritten.

    If `a -> b` and `b -> c` both exist, one pass yields `b` and a second yields
    `c`; the normaliser would not be idempotent. Catching it here means the
    property test cannot be the first thing to notice.
    """
    for table_name, table in (
        ("homoglyphs", rules.homoglyphs),
        ("diacritics", rules.diacritics),
        ("whitespace", rules.whitespace),
        ("punctuation", rules.punctuation),
    ):
        for source, target in table.items():
            for char in target:
                if char in table:
                    msg = (
                        f"{table_name} is not confluent: {source!r} -> {target!r}, "
                        f"but {char!r} is itself rewritten to {table[char]!r}"
                    )
                    raise RulesError(msg)
        overlap = set(table) & rules.preserved_chars
        if overlap:
            msg = f"{table_name} rewrites a character marked preserve: {sorted(overlap)}"
            raise RulesError(msg)
