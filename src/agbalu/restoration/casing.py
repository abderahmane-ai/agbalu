"""Represent independent casing decisions within an unchanged lexical word."""

from dataclasses import dataclass

from agbalu.restoration.annotations import CASE_COMPONENT, case_label


@dataclass(frozen=True)
class CaseUnit:
    char_start: int
    char_end: int
    label: int


def case_units(word: str) -> tuple[CaseUnit, ...]:
    return tuple(
        CaseUnit(match.start(), match.end(), case_label(match.group()))
        for match in CASE_COMPONENT.finditer(word)
    )


def component_bytes(word: str, unit: CaseUnit) -> tuple[int, int]:
    return (
        len(word[: unit.char_start].lower().encode("utf-8")),
        len(word[: unit.char_end].lower().encode("utf-8")),
    )
