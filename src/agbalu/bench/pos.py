"""POS scoring against `UD_Kabyle-ADPT`, the only gold Kabyle annotation there is.

Task 7.5. Two axes, both reported.

**Segmentation.** The treebank splits 29.6% of its syntactic words out of surface
tokens: `Di` is annotated as `ad` + `i`, `lbiru-ines` as `lbiru` + `in` + `as`. A
tagger emitting one label per whitespace token has no position for the second label.

- `surface` — one label per surface token, scored only where the treebank leaves the
  token whole. Drops the clitic-heavy tokens, so the number is optimistic.
- `gold-words` — one label per syntactic word, the standard UD setting, comparable to
  published UD numbers. The input is a segmentation no Kabyle tagger was trained on.

**Orthography.** The treebank is clean: 0 of 23,761 word forms change under normaliser
1.3.0. Tagging it once as published and once with every `ɛ`/`ɣ` replaced by the
homoglyph real Kabyle text carries gives the downstream cost of the corruption.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

from agbalu.normalise.rules import RuleSet
from agbalu.treebank import Sentence

Setting = Literal["surface", "gold-words"]
SETTINGS: Final[tuple[Setting, ...]] = ("surface", "gold-words")

Condition = Literal["canonical", "corrupted"]
CONDITIONS: Final[tuple[Condition, ...]] = ("canonical", "corrupted")

ABSTAIN: Final = "∅"
"""Stands where a tagger declined to answer. Not a label, and never predicted."""

UNANNOTATED: Final = "_"
"""A word the treebank itself left untagged; excluded from every denominator."""

CORRUPTIONS: Final[dict[str, str]] = {
    "ɛ": "ε",
    "Ɛ": "Σ",
    "ɣ": "γ",
    "Ɣ": "Γ",
}
"""Canonical Kabyle letter to the homoglyph that stands for it in the wild.

Counts in the seed monolingual corpus: `ε` 25,421, `Σ` 1,453, `γ` 222, `Γ` 129
(CLAUDE.md §2.1). Applied to every occurrence, so the corrupted condition is a
worst case, not a simulation of the corpus rate.
"""

_CORRUPTION_TABLE: Final[dict[int, str]] = {
    ord(canonical): homoglyph for canonical, homoglyph in CORRUPTIONS.items()
}


class PosScoringError(Exception):
    """A tagger's output could not be aligned with the treebank."""


class Tagger(Protocol):
    """One label per input token, or `None` where the system has no answer.

    Input is pre-tokenised because the treebank fixes the tokenisation under both
    settings; a tagger that re-splits its input cannot be aligned to gold.
    """

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]: ...


def corrupt(text: str, rules: RuleSet) -> str:
    """Rewrite canonical Kabyle letters as the homoglyphs found in real text.

    Raises:
        PosScoringError: a substitution the normaliser would not undo, which would
            make the corrupted condition measure something other than the defect.
    """
    for canonical, homoglyph in CORRUPTIONS.items():
        if rules.homoglyphs.get(homoglyph) != canonical:
            msg = f"the rules table does not fold {homoglyph!r} back to {canonical!r}"
            raise PosScoringError(msg)
    return text.translate(_CORRUPTION_TABLE)


@dataclass(frozen=True, slots=True)
class Unit:
    """One position a tagger must label, and the gold answer for it."""

    form: str
    gold: str | None
    """`None` where the treebank gives the position no single tag — a surface token
    it split into several words."""


@dataclass(frozen=True, slots=True)
class Item:
    """One sentence, as the positions a tagger will be asked about."""

    sent_id: str
    split: str
    units: tuple[Unit, ...]

    def forms(self) -> tuple[str, ...]:
        return tuple(unit.form for unit in self.units)


def items_for(
    sentences: Iterable[Sentence],
    setting: Setting,
    condition: Condition = "canonical",
    rules: RuleSet | None = None,
) -> list[Item]:
    """The scoring positions for one setting, with the input text under one condition.

    Gold labels never change between conditions; only what the tagger reads does.
    """
    if condition == "corrupted" and rules is None:
        msg = "the corrupted condition needs the homoglyph rules table"
        raise PosScoringError(msg)

    def surface_form(text: str) -> str:
        if condition == "canonical" or rules is None:
            return text
        return corrupt(text, rules)

    items: list[Item] = []
    for sentence in sentences:
        if setting == "gold-words":
            units = tuple(
                Unit(form=surface_form(word.form), gold=word.upos) for word in sentence.words
            )
        else:
            units = tuple(
                Unit(
                    form=surface_form(token.form),
                    gold=None if token.is_multiword else token.words[0].upos,
                )
                for token in sentence.tokens
            )
        items.append(Item(sent_id=sentence.sent_id, split=sentence.split, units=units))
    return items


@dataclass(frozen=True, slots=True)
class Prediction:
    """What one system answered at one scorable position."""

    sent_id: str
    split: str
    index: int
    form: str
    gold: str
    predicted: str | None


@dataclass(frozen=True, slots=True)
class Run:
    """One system's output over one setting and condition."""

    tagger: str
    revision: str
    setting: Setting
    condition: Condition
    sentences: int
    units: int
    unscorable: int
    predictions: tuple[Prediction, ...]

    @property
    def unscorable_rate(self) -> float:
        return self.unscorable / self.units if self.units else 0.0


def run(tagger: Tagger, items: Sequence[Item], setting: Setting, condition: Condition) -> Run:
    """Tag every item once. Scoring is a separate pass, so subsets cost no inference."""
    outputs = tagger.tag([item.forms() for item in items])
    if len(outputs) != len(items):
        msg = f"{tagger.name} returned {len(outputs)} sentences for {len(items)}"
        raise PosScoringError(msg)

    predictions: list[Prediction] = []
    units = 0
    unscorable = 0
    for item, labels in zip(items, outputs, strict=True):
        if len(labels) != len(item.units):
            msg = (
                f"{tagger.name} returned {len(labels)} labels for sentence "
                f"{item.sent_id!r}, which has {len(item.units)} positions"
            )
            raise PosScoringError(msg)
        for index, (unit, label) in enumerate(zip(item.units, labels, strict=True)):
            units += 1
            if unit.gold is None or unit.gold == UNANNOTATED:
                unscorable += 1
                continue
            predictions.append(
                Prediction(
                    sent_id=item.sent_id,
                    split=item.split,
                    index=index,
                    form=unit.form,
                    gold=unit.gold,
                    predicted=label,
                )
            )

    return Run(
        tagger=tagger.name,
        revision=tagger.revision,
        setting=setting,
        condition=condition,
        sentences=len(items),
        units=units,
        unscorable=unscorable,
        predictions=tuple(predictions),
    )


@dataclass(frozen=True, slots=True)
class LabelScore:
    """Precision, recall and F1 for one gold label."""

    label: str
    support: int
    predicted: int
    precision: float
    recall: float
    f1: float


@dataclass
class Score:
    """One system's result over one subset of the positions."""

    scored: int = 0
    answered: int = 0
    correct: int = 0
    ignored: frozenset[str] = frozenset()
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def accuracy(self) -> float:
        """Correct over every scored position. Abstention counts as wrong."""
        return self.correct / self.scored if self.scored else 0.0

    @property
    def coverage(self) -> float:
        """Positions the system was willing to answer at all."""
        return self.answered / self.scored if self.scored else 0.0

    @property
    def accuracy_when_answered(self) -> float:
        """Accuracy over answered positions only.

        The number a lookup-based system is usually quoted at, and not comparable
        to a neural tagger's, which answers everywhere.
        """
        return self.correct / self.answered if self.answered else 0.0

    @property
    def labels(self) -> tuple[str, ...]:
        """Gold labels with support, most frequent first.

        Macro-F1 averages over these. A label the treebank never uses has no recall
        to measure — predicting it is still punished, through the recall of whatever
        label was correct.
        """
        support: Counter[str] = Counter()
        for (gold, _), count in self.confusions.items():
            support[gold] += count
        return tuple(label for label, _ in support.most_common())

    def label_score(self, label: str) -> LabelScore:
        support = sum(n for (gold, _), n in self.confusions.items() if gold == label)
        predicted = sum(n for (_, hyp), n in self.confusions.items() if hyp == label)
        true_positive = self.confusions[(label, label)]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        denominator = precision + recall
        return LabelScore(
            label=label,
            support=support,
            predicted=predicted,
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / denominator if denominator else 0.0,
        )

    def label_scores(self) -> tuple[LabelScore, ...]:
        return tuple(self.label_score(label) for label in self.labels)

    @property
    def macro_f1(self) -> float:
        scores = self.label_scores()
        return sum(s.f1 for s in scores) / len(scores) if scores else 0.0

    def top_confusions(self, limit: int) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (gold, hypothesis, count)
            for (gold, hypothesis), count in self.confusions.most_common()
            if gold != hypothesis
        )[:limit]


def score(result: Run, ignore: frozenset[str] = frozenset()) -> Score:
    """Metrics over the positions whose gold label is not in `ignore`.

    `ignore={"PUNCT"}` is the comparison that matters against a lexicon: punctuation
    is 16.6% of this treebank and no lexicon holds it, so leaving it in scores the
    lexicon on tokens it cannot possibly know.
    """
    outcome = Score(ignored=ignore)
    for prediction in result.predictions:
        if prediction.gold in ignore:
            continue
        outcome.scored += 1
        hypothesis = prediction.predicted if prediction.predicted is not None else ABSTAIN
        if prediction.predicted is not None:
            outcome.answered += 1
            if prediction.predicted == prediction.gold:
                outcome.correct += 1
        outcome.confusions[(prediction.gold, hypothesis)] += 1
    return outcome
