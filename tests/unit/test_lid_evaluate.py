from __future__ import annotations

import pytest

from agbalu.lid.evaluate import accuracy, build_report, macro_f1, per_class

LABELS = ("kab_Latn", "rif_Latn", "tzm_Latn")


class TestPerClass:
    def test_a_perfect_pass_scores_one_everywhere(self) -> None:
        scores = per_class([(0, 0), (1, 1), (2, 2)], LABELS)
        assert [score.f1 for score in scores] == [1.0, 1.0, 1.0]

    def test_support_is_the_gold_count_not_the_predicted_count(self) -> None:
        """A macro over a class with one held-out example is a lottery; the support has to
        be readable beside the score, so it is returned rather than folded away."""
        scores = per_class([(0, 0), (0, 1), (0, 1)], LABELS)
        assert scores[0].support == 3
        assert scores[1].support == 0

    def test_precision_and_recall_come_apart_correctly(self) -> None:
        scores = per_class([(0, 0), (1, 0), (1, 1)], LABELS)
        assert scores[0].precision == pytest.approx(0.5)
        assert scores[0].recall == pytest.approx(1.0)
        assert scores[1].precision == pytest.approx(1.0)
        assert scores[1].recall == pytest.approx(0.5)

    def test_a_class_never_predicted_scores_zero_rather_than_raising(self) -> None:
        scores = per_class([(0, 0)], LABELS)
        assert scores[2].precision == 0.0
        assert scores[2].f1 == 0.0

    def test_no_predictions_at_all_is_all_zero(self) -> None:
        assert all(score.f1 == 0.0 for score in per_class([], LABELS))


class TestMacroF1:
    def test_a_class_with_no_support_is_not_averaged_in(self) -> None:
        """Otherwise the number reports the label inventory rather than the model."""
        scores = per_class([(0, 0), (1, 1)], LABELS)
        assert macro_f1(scores) == pytest.approx(1.0)

    def test_an_empty_pass_is_zero_not_an_error(self) -> None:
        assert macro_f1(per_class([], LABELS)) == 0.0


def test_accuracy_of_nothing_is_zero() -> None:
    assert accuracy([]) == 0.0


class TestReport:
    def test_the_worst_group_is_the_lowest_recall(self) -> None:
        pairs = [(0, 0), (0, 0), (1, 0), (1, 1)]
        groups = ["kab/web", "kab/web", "rif/scripture", "rif/web"]
        report = build_report(pairs, LABELS, groups)
        assert report.worst_group == ("rif/scripture", 0.0)

    def test_ties_break_on_the_group_name_so_the_report_is_stable(self) -> None:
        pairs = [(0, 1), (0, 1)]
        report = build_report(pairs, LABELS, ["b/web", "a/web"])
        assert report.worst_group[0] == "a/web"

    def test_the_score_is_half_macro_f1_and_half_worst_group(self) -> None:
        """Macro-F1 alone picks the checkpoint that wins the big domains; worst-group alone
        picks one that has learned a single hard cell and forgotten the rest."""
        pairs = [(0, 0), (1, 1), (1, 0)]
        report = build_report(pairs, LABELS, ["a/x", "b/y", "b/z"])
        assert report.score == pytest.approx((report.macro_f1 + report.worst_group[1]) / 2)

    def test_a_high_average_over_a_failing_cell_does_not_score_well(self) -> None:
        """The exact failure the previous set hid: 98% overall with one domain at zero."""
        pairs = [(0, 0)] * 99 + [(1, 0)]
        groups = ["kab/web"] * 99 + ["tzm/web"]
        report = build_report(pairs, LABELS, groups)
        assert report.accuracy == pytest.approx(0.99)
        assert report.score < 0.6

    def test_group_support_is_reported_beside_the_recall(self) -> None:
        report = build_report([(0, 0), (0, 1)], LABELS, ["a/x", "a/x"])
        assert report.group_support["a/x"] == 2
        assert report.group_recall["a/x"] == pytest.approx(0.5)

    def test_a_report_with_no_groups_still_has_a_score(self) -> None:
        report = build_report([], LABELS, [])
        assert report.worst_group == ("", 0.0)
        assert report.score == 0.0

    def test_the_dict_form_carries_everything_a_card_has_to_disclose(self) -> None:
        report = build_report([(0, 0), (1, 0)], LABELS, ["a/x", "b/y"])
        payload = report.to_dict()
        assert set(payload) >= {
            "accuracy",
            "macro_f1",
            "worst_group",
            "worst_group_recall",
            "score",
            "per_class",
            "per_group",
        }
        groups = payload["per_group"]
        assert isinstance(groups, dict)
        assert groups["b/y"] == {"recall": 0.0, "support": 1}

    def test_a_group_name_count_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="1 group names for 2 predictions"):
            build_report([(0, 0), (1, 1)], LABELS, ["a/x"])
