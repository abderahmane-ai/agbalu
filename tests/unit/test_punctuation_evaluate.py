"""Metrics, checked against hand-counted cases.

The point of every assertion here is that a model which predicts nothing scores well on
accuracy and badly on macro-F1. If that inversion ever stops holding, the reported numbers
stop meaning what the card says they mean.
"""

from __future__ import annotations

import pytest

from agbalu.punctuation.evaluate import (
    Prediction,
    macro_f1,
    per_class,
    render,
    score,
    trivial_baseline,
)
from agbalu.punctuation.labels import CASE, CASE_INDEX, PUNCTUATION, PUNCTUATION_INDEX, annotate

NONE = PUNCTUATION_INDEX["NONE"]
PERIOD = PUNCTUATION_INDEX["PERIOD"]
QUESTION = PUNCTUATION_INDEX["QUESTION"]
LOWER = CASE_INDEX["LOWER"]
UPPER = CASE_INDEX["UPPER_INIT"]


def test_per_class_counts_are_hand_checkable() -> None:
    pairs = [(0, 0), (0, 1), (1, 1), (1, 1), (2, 0)]
    scores = {entry["label"]: entry for entry in per_class(pairs, ("a", "b", "c"))}

    assert scores["a"]["support"] == 2
    assert scores["a"]["predicted"] == 2
    assert scores["a"]["precision"] == pytest.approx(0.5)
    assert scores["a"]["recall"] == pytest.approx(0.5)
    assert scores["b"]["precision"] == pytest.approx(2 / 3)
    assert scores["b"]["recall"] == pytest.approx(1.0)
    assert scores["c"]["f1"] == 0.0


def test_a_class_never_predicted_scores_zero_rather_than_raising() -> None:
    scores = per_class([(0, 0), (1, 0)], ("a", "b"))
    absent = next(entry for entry in scores if entry["label"] == "b")
    assert absent["predicted"] == 0
    assert absent["precision"] == 0.0
    assert absent["f1"] == 0.0


def test_macro_ignores_classes_with_no_support() -> None:
    scores = per_class([(0, 0), (0, 0)], ("a", "b"))
    assert macro_f1(scores) == pytest.approx(1.0)


def test_per_class_can_skip_a_label() -> None:
    labels = [entry["label"] for entry in per_class([(0, 0)], ("a", "b"), skip=0)]
    assert labels == ["b"]


def test_predicting_nothing_scores_high_on_accuracy_and_zero_on_marks() -> None:
    gold = annotate("Tegzem yiwen n useklu deg tebḥirt-nneɣ.")
    silent = Prediction(gold, tuple(NONE for _ in gold.words), tuple(LOWER for _ in gold.words))
    report = score([silent])

    assert report["final_mark_accuracy"] == 0.0
    assert report["punctuation_macro_f1"] == 0.0
    accuracy = sum(1 for label in gold.punctuation if label == NONE) / len(gold.words)
    assert accuracy == pytest.approx(5 / 6)


def test_a_perfect_prediction_scores_one_everywhere() -> None:
    gold = annotate("Azul fell-awen, amek tellid?")
    report = score([Prediction(gold, gold.punctuation, gold.case)])

    assert report["punctuation_macro_f1"] == pytest.approx(1.0)
    assert report["final_mark_accuracy"] == pytest.approx(1.0)
    assert report["exact_match"] == pytest.approx(1.0)
    assert report["case_initial_accuracy"] == pytest.approx(1.0)


def test_casing_is_split_at_the_sentence_boundary() -> None:
    """The first word is capitalised almost always, so it must not carry the casing score."""
    gold = annotate("Azul Ḥasan d amdakel.")
    predicted_case = (UPPER, LOWER, LOWER, LOWER)
    report = score([Prediction(gold, gold.punctuation, predicted_case)])

    assert report["case_initial_accuracy"] == pytest.approx(1.0)
    assert report["case_noninitial_f1"] == 0.0


def test_casing_has_exactly_two_classes() -> None:
    """`ALL_CAPS` was a third and is folded into `UPPER_INIT`: one example in dev, three in
    test, and the model that carried it emitted fourteen spurious all-caps words for those
    three. A two-class macro over it made a single word worth a quarter of the headline."""
    assert set(CASE) == {"LOWER", "UPPER_INIT"}
    assert annotate("AƔBALU d tin").case == (UPPER, LOWER, LOWER)


def test_exact_match_needs_both_heads_right() -> None:
    gold = annotate("Azul Ḥasan.")
    wrong_case = Prediction(gold, gold.punctuation, (UPPER, LOWER))
    assert score([wrong_case])["exact_match"] == 0.0
    assert score([wrong_case])["final_mark_accuracy"] == pytest.approx(1.0)


def test_trivial_baseline_is_capitalise_first_and_a_final_period() -> None:
    gold = annotate("Azul fell-awen, amek tellid?")
    baseline = trivial_baseline(gold)

    assert baseline.punctuation == (NONE, NONE, NONE, PERIOD)
    assert baseline.case == (UPPER, LOWER, LOWER, LOWER)
    assert baseline.gold is gold


def test_trivial_baseline_gets_a_statement_right_and_a_question_wrong() -> None:
    statement = annotate("Tegzem yiwen n useklu.")
    question = annotate("D acu i txedmed assa?")
    report = score([trivial_baseline(statement), trivial_baseline(question)])

    assert report["final_mark_accuracy"] == pytest.approx(0.5)
    assert report["sentences"] == 2


def test_a_prediction_that_does_not_cover_the_sentence_is_rejected() -> None:
    gold = annotate("Azul fell-awen.")
    with pytest.raises(ValueError, match="does not cover"):
        Prediction(gold, (NONE,), (LOWER,))


def test_empty_sentences_do_not_divide_by_zero() -> None:
    report = score([Prediction(annotate(""), (), ())])
    assert report["sentences"] == 0
    assert report["final_mark_accuracy"] == 0.0
    assert report["exact_match"] == 0.0


def test_render_names_every_class() -> None:
    gold = annotate("Azul fell-awen, amek tellid?")
    text = render(score([trivial_baseline(gold)]), "BASELINE")

    assert "BASELINE" in text
    for label in (*PUNCTUATION, *CASE):
        assert label in text
    assert "macro-F1 over marks" in text
