"""Represent lexical spans and ordered punctuation gaps without losing reference text."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

MARKS: Final = ("", ",", ".", "?", "!", ":", ";", "…", "?!")
CASES: Final = ("LOWER", "INITIAL_CAP", "ALL_CAP")
ALL_CAP: Final = CASES.index("ALL_CAP")
TERMINAL_LABELS: Final = frozenset(MARKS.index(mark) for mark in (".", "?", "!", "?!"))
IGNORE: Final = -100
PRIMARY: Final = "@"
CASE_COMPONENT: Final = re.compile(r"[^\-'’]+")
type CasePrediction = int | tuple[int, ...]
STRUCTURAL: Final = frozenset('«»“”„"()[]—–-')
OPENERS: Final = {"«": "»", "“": "”", "(": ")", "[": "]"}
CLOSERS: Final = frozenset(OPENERS.values())


@dataclass(frozen=True)
class Gap:
    mark: int
    template: str
    valid: bool
    raw: str


@dataclass(frozen=True)
class Document:
    reference: str
    words: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]
    casing: tuple[int, ...]
    gaps: tuple[Gap, ...]

    def __post_init__(self) -> None:
        count = len(self.words)
        if len(self.spans) != count or len(self.casing) != count or len(self.gaps) != count + 1:
            msg = "document words, spans, casing and gaps disagree"
            raise ValueError(msg)
        rebuilt = self.gaps[0].raw + "".join(
            self.reference[start:end] + self.gaps[index + 1].raw
            for index, (start, end) in enumerate(self.spans)
        )
        if rebuilt != self.reference:
            msg = "reference character conservation failed"
            raise ValueError(msg)


def is_letter_or_number(char: str) -> bool:
    return unicodedata.category(char)[0] in "LMN"


def lexical_spans(text: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        if not is_letter_or_number(text[cursor]):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text):
            char = text[cursor]
            connected = (
                char in "-'’" and cursor + 1 < len(text) and is_letter_or_number(text[cursor + 1])
            )
            numeric_separator = (
                char in ".,:/"
                and cursor + 1 < len(text)
                and text[cursor - 1].isdecimal()
                and text[cursor + 1].isdecimal()
            )
            if not is_letter_or_number(char) and not connected and not numeric_separator:
                break
            cursor += 1
        result.append((start, cursor))
    return tuple(result)


def case_label(word: str) -> int:
    letters = "".join(char for char in word if char.isalpha())
    if not letters:
        return IGNORE
    if letters.islower():
        return 0
    if letters[:1].isupper() and (len(letters) == 1 or letters[1:].islower()):
        return 1
    if letters.isupper():
        return 2
    return IGNORE


def parse_gap(raw: str) -> Gap:
    visible = "".join(char for char in raw if not char.isspace())
    visible = visible.replace("...", "…").replace("!?", "?!")
    visible = visible.replace("\u201e", "“")
    primary = "".join(char for char in visible if char in ",.?!:;…")
    valid = primary in MARKS and all(char in STRUCTURAL or char in ",.?!:;…" for char in visible)
    mark = MARKS.index(primary) if primary in MARKS else IGNORE
    template: list[str] = []
    inserted = False
    for char in visible:
        if char in ",.?!:;…":
            if not inserted:
                template.append(PRIMARY)
                inserted = True
        else:
            template.append(char)
    if not inserted:
        template.insert(0, PRIMARY)
    return Gap(mark, "".join(template), valid, raw)


def annotate(text: str) -> Document:
    reference = unicodedata.normalize("NFC", text)
    spans = lexical_spans(reference)
    surfaces = tuple(reference[start:end] for start, end in spans)
    gaps: list[Gap] = []
    cursor = 0
    for start, end in spans:
        gaps.append(parse_gap(reference[cursor:start]))
        cursor = end
    gaps.append(parse_gap(reference[cursor:]))
    return Document(
        reference,
        tuple(word.lower() for word in surfaces),
        spans,
        tuple(case_label(word) for word in surfaces),
        tuple(gaps),
    )


def apply_case(word: str, label: CasePrediction) -> str:
    if isinstance(label, tuple):
        components = list(CASE_COMPONENT.finditer(word))
        if len(components) != len(label):
            message = f"{word!r}: {len(components)} components disagree with case labels {label!r}"
            raise ValueError(message)
        result: list[str] = []
        cursor = 0
        for component, predicted in zip(components, label, strict=True):
            result.append(word[cursor : component.start()])
            result.append(apply_case(component.group(), predicted))
            cursor = component.end()
        result.append(word[cursor:])
        return unicodedata.normalize("NFC", "".join(result))
    if label == ALL_CAP:
        return word.upper()
    if label == 1:
        for index, char in enumerate(word):
            if char.isalpha():
                return word[:index] + char.upper() + word[index + 1 :]
    return word


def canonical_gap(gap: Gap, template: str | None = None) -> str:
    if not gap.valid:
        msg = f"unsupported gap {gap.raw!r}"
        raise ValueError(msg)
    return (template or gap.template).replace(PRIMARY, MARKS[gap.mark])
