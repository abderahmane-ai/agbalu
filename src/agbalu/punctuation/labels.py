"""The label scheme, and the two conversions that must be exact inverses.

`annotate` takes written Kabyle to the pair (what Fadhma would emit, what has to be put
back); `restore` takes that pair to written Kabyle. Everything downstream — the corpus
builder, the dataset, the metrics — is defined against these two, so a disagreement between
them would be invisible everywhere else and wrong everywhere.

Four marks survive. `!` and `;` fold into `PERIOD`: they are 0.406% of tokens together, and
the corpus carries `Medlet idlisen-nwen!` beside `Medlet idlisen-nwen.` — the same sentence
under both, so the distinction is not recoverable and a class for it only costs macro-F1.

Casing is two classes. `ALL_CAPS` was a third and is folded into `UPPER_INIT`: it occurs once
in the dev split and three times in test, which is too few to learn or to score, and the
model that had it emitted fourteen spurious all-caps words for the three real ones. Restoring
an acronym to its full capitals is therefore out of scope, deliberately.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final, Literal, get_args

PunctuationLabel = Literal["NONE", "COMMA", "PERIOD", "QUESTION", "COLON"]
CaseLabel = Literal["LOWER", "UPPER_INIT"]

PUNCTUATION: Final[tuple[PunctuationLabel, ...]] = get_args(PunctuationLabel)
CASE: Final[tuple[CaseLabel, ...]] = get_args(CaseLabel)

PUNCTUATION_INDEX: Final[dict[PunctuationLabel, int]] = {
    label: index for index, label in enumerate(PUNCTUATION)
}
CASE_INDEX: Final[dict[CaseLabel, int]] = {label: index for index, label in enumerate(CASE)}

NONE: Final = PUNCTUATION_INDEX["NONE"]
LOWER: Final = CASE_INDEX["LOWER"]

#: Ignored by the loss, and never predicted: a subword that does not begin a word.
IGNORE_INDEX: Final = -100

MARK_TO_LABEL: Final[dict[str, PunctuationLabel]] = {
    ",": "COMMA",
    ".": "PERIOD",
    "!": "PERIOD",
    ";": "PERIOD",
    "?": "QUESTION",
    ":": "COLON",
}

LABEL_TO_MARK: Final[dict[PunctuationLabel, str]] = {
    "NONE": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "COLON": ":",
}

#: Kept inside a word. Fadhma emits letters, `-` and a delimiter and nothing else, so any
#: other character in a training sentence is text the model can never see at inference.
WORD_CHARS: Final = "-"


def _keepable(char: str) -> bool:
    return char.isalnum() or char in WORD_CHARS


def split_words(text: str) -> list[str]:
    """The one definition of what a word is. Everything that splits text goes through here.

    Composed first, because a combining mark is not alphanumeric: decomposed `ḍ` would lose
    its dot and stop matching its composed twin, which is silent wherever it matters.

    Format characters are removed rather than split on. `Cf` covers the zero-width space, the
    BOM, the soft hyphen and the directional marks — all invisible, none a word boundary, and
    62 of them survive into the training text where splitting would cut a word in two.
    """
    composed = unicodedata.normalize("NFC", text)
    visible = "".join(char for char in composed if unicodedata.category(char) != "Cf")
    spaced = "".join(char if _keepable(char) else " " for char in visible)
    return [part for part in spaced.split() if any(char.isalnum() for char in part)]


def collation_key(text: str) -> str:
    """The form written Kabyle and an ASR transcript share, and the join key between them."""
    return " ".join(word.lower() for word in split_words(text))


@dataclass(frozen=True, slots=True)
class Annotation:
    """A sentence as the model sees it, beside what the model has to predict."""

    words: tuple[str, ...]
    punctuation: tuple[int, ...]
    case: tuple[int, ...]

    def __post_init__(self) -> None:
        if not len(self.words) == len(self.punctuation) == len(self.case):
            msg = (
                f"ragged annotation: {len(self.words)} words, "
                f"{len(self.punctuation)} punctuation, {len(self.case)} case"
            )
            raise ValueError(msg)

    @property
    def text(self) -> str:
        """What Fadhma would emit for this sentence. Equal to `collation_key` of the source."""
        return " ".join(self.words)


def _case_of(word: str) -> CaseLabel:
    return "UPPER_INIT" if word[:1].isupper() else "LOWER"


def _mark_of(token: str) -> PunctuationLabel:
    """The first mark in a token's trailing run, so `yenna."` and `tellid?»` still label."""
    trailing: list[str] = []
    for char in reversed(token):
        if _keepable(char):
            break
        trailing.append(char)
    for char in reversed(trailing):
        if char in MARK_TO_LABEL:
            return MARK_TO_LABEL[char]
    return "NONE"


def annotate(text: str) -> Annotation:
    """Split written Kabyle into ASR-shaped words and the labels that restore them.

    A token that is nothing but punctuation carries its mark back to the previous word, so a
    space-separated `azul , d acu` annotates the same as `azul, d acu`.
    """
    words: list[str] = []
    punctuation: list[int] = []
    case: list[int] = []

    for token in unicodedata.normalize("NFC", text).split():
        mark = _mark_of(token)
        parts = split_words(token)
        if not parts:
            if words and mark != "NONE" and punctuation[-1] == NONE:
                punctuation[-1] = PUNCTUATION_INDEX[mark]
            continue
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            words.append(part.lower())
            punctuation.append(PUNCTUATION_INDEX[mark] if last else NONE)
            case.append(CASE_INDEX[_case_of(part)])

    return Annotation(tuple(words), tuple(punctuation), tuple(case))


def _cased(word: str, label: CaseLabel) -> str:
    return word[:1].upper() + word[1:] if label == "UPPER_INIT" else word


def restore(words: tuple[str, ...], punctuation: tuple[int, ...], case: tuple[int, ...]) -> str:
    """Written Kabyle from the model's two predictions. The inverse of `annotate`."""
    annotation = Annotation(words, punctuation, case)
    return " ".join(
        _cased(word, CASE[casing]) + LABEL_TO_MARK[PUNCTUATION[mark]]
        for word, mark, casing in zip(
            annotation.words, annotation.punctuation, annotation.case, strict=True
        )
    )


def strip_text(text: str) -> str:
    """What an ASR model would emit for a written sentence."""
    return annotate(text).text
