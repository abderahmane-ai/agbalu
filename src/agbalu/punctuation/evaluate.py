"""Metrics, and the baselines that have to appear beside them.

Accuracy is never reported on its own: `NONE` is 82.9% of punctuation labels and `LOWER`
82.9% of casing labels, so a model that predicts nothing at all scores in the eighties on
both. Macro-F1 runs over the marks only, excluding `NONE`.

Casing is split at the sentence boundary because the first word is capitalised in 99.7% of
references. Folding the two together reports the trivial half: the number that carries
information is the one over non-initial words, whose base rate is 4.48%.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, TypedDict

from agbalu.punctuation.labels import (
    CASE,
    CASE_INDEX,
    LOWER,
    NONE,
    PUNCTUATION,
    PUNCTUATION_INDEX,
    Annotation,
)

PERIOD: Final = PUNCTUATION_INDEX["PERIOD"]
UPPER_INIT: Final = CASE_INDEX["UPPER_INIT"]


class ClassScore(TypedDict):
    label: str
    support: int
    predicted: int
    precision: float
    recall: float
    f1: float


class Report(TypedDict):
    sentences: int
    words: int
    punctuation: list[ClassScore]
    punctuation_macro_f1: float
    case: list[ClassScore]
    case_initial_accuracy: float
    case_noninitial_f1: float
    final_mark_accuracy: float
    exact_match: float


@dataclass(frozen=True, slots=True)
class Prediction:
    gold: Annotation
    punctuation: tuple[int, ...]
    case: tuple[int, ...]

    def __post_init__(self) -> None:
        if not len(self.gold.words) == len(self.punctuation) == len(self.case):
            msg = (
                f"prediction does not cover the sentence: {len(self.gold.words)} words, "
                f"{len(self.punctuation)} punctuation, {len(self.case)} case"
            )
            raise ValueError(msg)


def _f1(true_positive: int, predicted: int, support: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / support if support else 0.0
    harmonic = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, harmonic


def per_class(
    pairs: list[tuple[int, int]], names: tuple[str, ...], *, skip: int | None = None
) -> list[ClassScore]:
    support: Counter[int] = Counter(gold for gold, _ in pairs)
    predicted: Counter[int] = Counter(hypothesis for _, hypothesis in pairs)
    hits: Counter[int] = Counter(gold for gold, hypothesis in pairs if gold == hypothesis)

    scores: list[ClassScore] = []
    for index, name in enumerate(names):
        if index == skip:
            continue
        precision, recall, harmonic = _f1(hits[index], predicted[index], support[index])
        scores.append(
            {
                "label": name,
                "support": support[index],
                "predicted": predicted[index],
                "precision": precision,
                "recall": recall,
                "f1": harmonic,
            }
        )
    return scores


def macro_f1(scores: list[ClassScore]) -> float:
    present = [score["f1"] for score in scores if score["support"]]
    return sum(present) / len(present) if present else 0.0


def upper_init_f1(scores: list[ClassScore]) -> float:
    """The casing headline: whether a non-initial word is a proper noun.

    Named rather than a macro over whatever casing classes exist, so that adding a third one
    cannot quietly redefine the headline. That is not hypothetical: `ALL_CAPS` was a third
    class with one example in dev, a two-class macro made that single word worth a quarter of
    the number, and it swung 0.54 to 0.70 and back while the loss moved by 0.003.
    """
    for entry in scores:
        if entry["label"] == CASE[UPPER_INIT]:
            return entry["f1"]
    return 0.0


def score(predictions: list[Prediction]) -> Report:
    punctuation_pairs: list[tuple[int, int]] = []
    case_pairs: list[tuple[int, int]] = []
    noninitial_pairs: list[tuple[int, int]] = []
    initial_hits = 0
    initial_total = 0
    final_hits = 0
    final_total = 0
    exact = 0

    for prediction in predictions:
        gold = prediction.gold
        if not gold.words:
            continue
        punctuation_pairs.extend(zip(gold.punctuation, prediction.punctuation, strict=True))
        case_pairs.extend(zip(gold.case, prediction.case, strict=True))
        noninitial_pairs.extend(zip(gold.case[1:], prediction.case[1:], strict=True))

        initial_total += 1
        initial_hits += gold.case[0] == prediction.case[0]
        final_total += 1
        final_hits += gold.punctuation[-1] == prediction.punctuation[-1]
        exact += gold.punctuation == prediction.punctuation and gold.case == prediction.case

    punctuation_scores = per_class(punctuation_pairs, PUNCTUATION)
    marks = [entry for entry in punctuation_scores if entry["label"] != PUNCTUATION[NONE]]
    noninitial = per_class(noninitial_pairs, CASE, skip=LOWER)

    return {
        "sentences": final_total,
        "words": len(punctuation_pairs),
        "punctuation": punctuation_scores,
        "punctuation_macro_f1": macro_f1(marks),
        "case": per_class(case_pairs, CASE),
        "case_initial_accuracy": initial_hits / initial_total if initial_total else 0.0,
        "case_noninitial_f1": upper_init_f1(noninitial),
        "final_mark_accuracy": final_hits / final_total if final_total else 0.0,
        "exact_match": exact / final_total if final_total else 0.0,
    }


def trivial_baseline(gold: Annotation) -> Prediction:
    """Capitalise the first word, put a period at the end, predict nothing else.

    This is what a two-line rule already achieves, and it is the bar the model has to clear
    before any of its other numbers are interesting.
    """
    length = len(gold.words)
    punctuation = tuple(PERIOD if index == length - 1 else NONE for index in range(length))
    case = tuple(UPPER_INIT if index == 0 else LOWER for index in range(length))
    return Prediction(gold, punctuation, case)


def _row(entry: ClassScore) -> str:
    return (
        f"  {entry['label']:12} {entry['support']:8,} {entry['predicted']:8,}"
        f" {entry['precision']:7.3f} {entry['recall']:7.3f} {entry['f1']:7.3f}"
    )


def render(report: Report, title: str) -> str:
    lines = [
        f"{title}: {report['sentences']:,} sentences, {report['words']:,} words",
        f"  {'label':12} {'support':>8} {'pred':>8} {'P':>7} {'R':>7} {'F1':>7}",
        *(_row(entry) for entry in report["punctuation"]),
        f"  {'macro-F1 over marks':32} {report['punctuation_macro_f1']:7.3f}",
        "",
        *(_row(entry) for entry in report["case"]),
        f"  {'casing, first word':32} {report['case_initial_accuracy']:7.3f}",
        f"  {'casing, non-initial proper nouns':32} {report['case_noninitial_f1']:7.3f}",
        "",
        f"  {'sentence-final mark accuracy':32} {report['final_mark_accuracy']:7.3f}",
        f"  {'exact match (marks and case)':32} {report['exact_match']:7.3f}",
    ]
    return "\n".join(lines)
