"""Metrics for the language identifier, reported per class and per `(language, domain)`.

The headline is the worst group, not the mean. A mean over a corpus where one class is
five domains and another is two reports the biggest domain: the first classifier read
98.26% on a dev split drawn from the training sources and 76.27% on the held-out set,
and nothing in the per-class averages showed the difference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


def per_class(pairs: Sequence[tuple[int, int]], labels: Sequence[str]) -> list[ClassScore]:
    """Precision, recall and F1 for every label, with its support beside it.

    Support is returned rather than folded away because a macro average over a class
    with one held-out example is a lottery: read the support before averaging.
    """
    size = len(labels)
    true_positive = [0] * size
    false_positive = [0] * size
    false_negative = [0] * size
    for gold, predicted in pairs:
        if gold == predicted:
            true_positive[gold] += 1
        else:
            false_positive[predicted] += 1
            false_negative[gold] += 1

    scores: list[ClassScore] = []
    for index, label in enumerate(labels):
        predicted_total = true_positive[index] + false_positive[index]
        support = true_positive[index] + false_negative[index]
        precision = true_positive[index] / predicted_total if predicted_total else 0.0
        recall = true_positive[index] / support if support else 0.0
        denominator = precision + recall
        scores.append(
            ClassScore(
                label=label,
                support=support,
                precision=precision,
                recall=recall,
                f1=2 * precision * recall / denominator if denominator else 0.0,
            )
        )
    return scores


def macro_f1(scores: Sequence[ClassScore]) -> float:
    """Macro-F1 over the classes that have support. Absent classes are not averaged in."""
    present = [score.f1 for score in scores if score.support]
    return sum(present) / len(present) if present else 0.0


def accuracy(pairs: Sequence[tuple[int, int]]) -> float:
    return sum(1 for gold, predicted in pairs if gold == predicted) / len(pairs) if pairs else 0.0


@dataclass(frozen=True, slots=True)
class Report:
    """One evaluation pass, per class and per group."""

    accuracy: float
    macro_f1: float
    classes: tuple[ClassScore, ...]
    group_recall: Mapping[str, float] = field(default_factory=dict)
    group_support: Mapping[str, int] = field(default_factory=dict)

    @property
    def worst_group(self) -> tuple[str, float]:
        """The `(language, domain)` cell the model is worst on, and its recall."""
        if not self.group_recall:
            return "", 0.0
        return min(self.group_recall.items(), key=lambda item: (item[1], item[0]))

    @property
    def score(self) -> float:
        """What the checkpoint is selected on.

        Half macro-F1, half worst-group recall. Macro-F1 alone picks the checkpoint
        that wins the big domains; worst-group alone picks one that has learned a
        single hard cell and forgotten the rest.
        """
        return (self.macro_f1 + self.worst_group[1]) / 2

    def to_dict(self) -> dict[str, object]:
        worst, worst_recall = self.worst_group
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "worst_group": worst,
            "worst_group_recall": worst_recall,
            "score": self.score,
            "per_class": {
                score.label: {
                    "support": score.support,
                    "precision": score.precision,
                    "recall": score.recall,
                    "f1": score.f1,
                }
                for score in self.classes
            },
            "per_group": {
                group: {"recall": recall, "support": self.group_support.get(group, 0)}
                for group, recall in sorted(self.group_recall.items())
            },
        }


def build_report(
    pairs: Sequence[tuple[int, int]], labels: Sequence[str], groups: Sequence[str]
) -> Report:
    """Score one pass. `groups[i]` is the group name of `pairs[i]`."""
    if len(groups) != len(pairs):
        msg = f"{len(groups)} group names for {len(pairs)} predictions"
        raise ValueError(msg)
    correct: dict[str, int] = {}
    support: dict[str, int] = {}
    for (gold, predicted), group in zip(pairs, groups, strict=True):
        support[group] = support.get(group, 0) + 1
        correct[group] = correct.get(group, 0) + int(gold == predicted)
    scores = per_class(pairs, labels)
    return Report(
        accuracy=accuracy(pairs),
        macro_f1=macro_f1(scores),
        classes=tuple(scores),
        group_recall={group: correct[group] / support[group] for group in support},
        group_support=support,
    )
