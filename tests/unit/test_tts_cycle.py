"""Task 12.5's harness: the conditions, the delta, and the control that licenses both."""

from __future__ import annotations

import pytest

from agbalu.speech.metrics import MetricError
from agbalu.tts.cycle import (
    BASELINE,
    CYCLE,
    FLOOR,
    PUBLISHED_CER,
    TOLERANCE,
    Condition,
    Control,
    CycleError,
    Report,
    read_result,
    restricted,
)


def condition(name: str, cer: float, *, utterances: int = 10, **extra: object) -> Condition:
    """A scored condition stated directly, so a test fixes the rate it is about."""
    return Condition(
        name=name,
        utterances=utterances,
        cer_percent=cer,
        wer_percent=cer * 3,
        **extra,  # type: ignore[arg-type]
    )


class TestConditionScoring:
    def test_scores_through_the_project_metric(self) -> None:
        scored = Condition.score(FLOOR, [("azul fell-awen", "azul fell-awen")])
        assert scored.cer_percent == 0.0
        assert scored.wer_percent == 0.0
        assert scored.utterances == 1

    def test_one_substitution_in_fourteen_characters(self) -> None:
        scored = Condition.score(CYCLE, [("azul fell-awen", "azul fell-awem")])
        assert scored.utterances == 1
        assert scored.cer_percent == pytest.approx(100 / 14, abs=1e-3)

    def test_empty_input_is_unmeasured_not_perfect(self) -> None:
        scored = Condition.score(BASELINE, [])
        assert scored.utterances == 0
        assert scored.cer_percent is None
        assert scored.wer_percent is None

    def test_rate_is_pooled_over_the_set_not_averaged_per_utterance(self) -> None:
        """One long correct line beside one short wrong one: a per-utterance mean would
        read 50%, and the corpus rate is what the field reports."""
        pairs = [("a" * 99, "a" * 99), ("b", "c")]
        scored = Condition.score(CYCLE, pairs)
        assert scored.cer_percent == pytest.approx(1.0, abs=1e-6)

    def test_reference_that_reduces_to_nothing_raises_from_the_metric(self) -> None:
        with pytest.raises(MetricError, match="no reference units"):
            Condition.score(CYCLE, [("...", "azul")])

    def test_kabyle_characters_survive_scoring(self) -> None:
        scored = Condition.score(CYCLE, [("aɣbalu ḥemmleɣ tiẓgi", "aɣbalu ḥemmleɣ tiẓgi")])
        assert scored.cer_percent == 0.0

    def test_greek_homoglyph_is_an_error_not_a_match(self) -> None:
        """Latin `ɛ` U+025B against Greek `ε` U+03B5 — the corpus's own 2.6% defect."""
        scored = Condition.score(CYCLE, [("aɛumu", "aεumu")])
        assert scored.cer_percent is not None
        assert scored.cer_percent > 0

    def test_decomposed_and_composed_forms_differ(self) -> None:
        """NFC `ḥ` against NFD `h` plus a combining dot: the metric compares what it is
        given, and the normaliser is what makes the two one."""
        scored = Condition.score(CYCLE, [("ḥed", "ḥed")])
        assert scored.cer_percent is not None
        assert scored.cer_percent > 0


class TestConditionInvariants:
    def test_utterances_without_a_rate_is_refused(self) -> None:
        with pytest.raises(CycleError, match="if and only if"):
            Condition(name=FLOOR, utterances=5, cer_percent=None, wer_percent=None)

    def test_rate_without_utterances_is_refused(self) -> None:
        with pytest.raises(CycleError, match="if and only if"):
            Condition(name=FLOOR, utterances=0, cer_percent=8.0, wer_percent=24.0)

    def test_half_measured_is_refused(self) -> None:
        with pytest.raises(CycleError, match="if and only if"):
            Condition(name=FLOOR, utterances=3, cer_percent=8.0, wer_percent=None)

    def test_negative_utterances_is_refused(self) -> None:
        with pytest.raises(CycleError, match="negative utterance count"):
            Condition(name=FLOOR, utterances=-1, cer_percent=None, wer_percent=None)

    def test_negative_rate_is_refused(self) -> None:
        with pytest.raises(CycleError, match="negative error rate"):
            Condition(name=FLOOR, utterances=1, cer_percent=-0.5, wer_percent=1.0)

    def test_error_rate_above_one_hundred_is_allowed(self) -> None:
        """Insertions are unbounded, so a hypothesis longer than its reference exceeds
        100% and clamping it would hide a degenerate decode."""
        assert condition(CYCLE, 250.0).cer_percent == 250.0


class TestConditionPayload:
    def test_omits_what_was_not_measured(self) -> None:
        assert Condition.score(BASELINE, []).as_dict() == {"utterances": 0}

    def test_carries_no_name_because_the_payload_keys_by_it(self) -> None:
        assert "name" not in condition(FLOOR, 8.0).as_dict()

    def test_matches_the_shape_task_12_1_wrote(self) -> None:
        payload = condition(
            FLOOR,
            8.3331,
            utterances=1000,
            loss=0.3814,
            audio_seconds=4251.2,
            previews=(("azul", "azul"),),
        ).as_dict()
        assert payload == {
            "loss": 0.3814,
            "wer_percent": 8.3331 * 3,
            "cer_percent": 8.3331,
            "utterances": 1000,
            "previews": [{"reference": "azul", "hypothesis": "azul"}],
            "audio_seconds": 4251.2,
        }


class TestRestricted:
    def test_scores_only_the_kept_references(self) -> None:
        pairs = (("azul fell", "azul fell"), ("ṛuḥ ass-a", "ṛuḥ ass-b"))

        kept = restricted(BASELINE, pairs, {"ṛuḥ ass-a"})

        assert kept.utterances == 1
        assert kept.cer_percent == round(100 / len("ṛuḥ ass-a"), 4)

    def test_an_empty_restriction_is_unmeasured_rather_than_a_rate(self) -> None:
        """A rate with no denominator is not a number, and `Score.rate` says so; reporting
        zero here would read as a perfect score."""
        kept = restricted(BASELINE, (("azul fell", "azul fell"),), set())

        assert kept.cer_percent is None
        assert kept.as_dict() == {"utterances": 0}

    def test_targets_may_name_a_reference_that_is_absent(self) -> None:
        kept = restricted(BASELINE, [("azul", "azul")], {"azul", "never-decoded"})
        assert kept.utterances == 1


class TestControl:
    def test_holds_at_task_12_1_s_measured_floor(self) -> None:
        control = Control(measured=8.3331)
        assert control.gap == 0.3195
        assert control.holds

    def test_gap_is_signed(self) -> None:
        assert Control(measured=PUBLISHED_CER - 0.5).gap == -0.5

    def test_exactly_at_the_tolerance_holds(self) -> None:
        assert Control(measured=PUBLISHED_CER + TOLERANCE).holds

    def test_just_past_the_tolerance_fails(self) -> None:
        assert not Control(measured=PUBLISHED_CER + TOLERANCE + 0.01).holds

    def test_a_broken_decoder_fails_the_control(self) -> None:
        """No n-gram loaded reads as tens of points, which is what the tolerance is for."""
        assert not Control(measured=41.0).holds

    def test_a_floor_far_below_the_published_rate_also_fails(self) -> None:
        """Too good is a defect too: it means the floor is not scoring what it claims."""
        assert not Control(measured=0.0).holds


class TestReport:
    def test_delta_is_the_condition_minus_the_floor(self) -> None:
        report = Report((condition(FLOOR, 8.3331), condition(BASELINE, 11.8902)))
        assert report.delta(BASELINE) == 3.5571

    def test_reproduces_task_12_1_s_published_delta(self) -> None:
        report = Report(
            (
                condition(FLOOR, 8.3331, utterances=1000),
                condition(BASELINE, 11.8902, utterances=1000),
            )
        )
        assert report.deltas == {BASELINE: 3.5571}
        assert report.control.holds

    def test_a_report_without_a_floor_is_refused(self) -> None:
        with pytest.raises(CycleError, match="no 'floor_real_audio' condition"):
            Report((condition(BASELINE, 11.0),))

    def test_duplicate_condition_names_are_refused(self) -> None:
        with pytest.raises(CycleError, match="share a name"):
            Report((condition(FLOOR, 8.0), condition(FLOOR, 9.0)))

    def test_the_floor_has_no_delta_against_itself(self) -> None:
        report = Report((condition(FLOOR, 8.0), condition(CYCLE, 9.0)))
        with pytest.raises(CycleError, match="not a result"):
            report.delta(FLOOR)

    def test_unknown_condition_names_what_it_has(self) -> None:
        report = Report((condition(FLOOR, 8.0),))
        with pytest.raises(CycleError, match="no condition named"):
            report.delta("matoub-v9")

    def test_an_unmeasured_condition_has_no_delta(self) -> None:
        report = Report((condition(FLOOR, 8.0), Condition(CYCLE, 0, None, None)))
        with pytest.raises(CycleError, match="no delta exists"):
            report.delta(CYCLE)

    def test_the_smaller_delta_wins(self) -> None:
        report = Report((condition(FLOOR, 8.0), condition(CYCLE, 10.0), condition(BASELINE, 11.5)))
        assert report.beats(CYCLE, BASELINE)
        assert not report.beats(BASELINE, CYCLE)

    def test_a_system_can_win_on_delta_while_both_sit_above_the_floor(self) -> None:
        report = Report((condition(FLOOR, 8.0), condition(CYCLE, 8.9), condition(BASELINE, 11.5)))
        assert report.delta(CYCLE) == pytest.approx(0.9)
        assert report.beats(CYCLE, BASELINE)

    def test_deltas_exclude_the_floor(self) -> None:
        report = Report((condition(FLOOR, 8.0), condition(CYCLE, 9.0), condition(BASELINE, 11.0)))
        assert set(report.deltas) == {CYCLE, BASELINE}

    def test_a_floor_only_report_is_valid_and_has_no_deltas(self) -> None:
        """12.1's smoke shape: the floor alone still evaluates the control."""
        report = Report((condition(FLOOR, 8.2),))
        assert report.deltas == {}
        assert report.control.holds

    def test_an_unmeasured_floor_has_no_control(self) -> None:
        report = Report((Condition(FLOOR, 0, None, None),))
        with pytest.raises(CycleError, match="control cannot be evaluated"):
            _ = report.control


class TestReadResult:
    def test_round_trips_a_report_it_wrote(self) -> None:
        report = Report(
            (
                condition(FLOOR, 8.3331, utterances=1000, loss=0.38, audio_seconds=4251.2),
                condition(BASELINE, 11.8902, utterances=1000),
            )
        )
        recovered = read_result(report.as_dict())
        assert recovered.deltas == report.deltas
        assert recovered.floor.audio_seconds == 4251.2
        assert recovered.control.as_dict() == report.control.as_dict()

    def test_reads_a_payload_that_predates_the_deltas_block(self) -> None:
        """Task 12.1's result on disk carries `cycle_cer_delta`, not `deltas`, and the
        delta is derived from the conditions rather than read out of the file."""
        legacy = {
            "conditions": {
                FLOOR: {"cer_percent": 8.3331, "wer_percent": 24.8149, "utterances": 1000},
                BASELINE: {"cer_percent": 11.8902, "wer_percent": 34.3551, "utterances": 1000},
            },
            "cycle_cer_delta": 3.5571,
        }
        assert read_result(legacy).delta(BASELINE) == 3.5571

    def test_a_stored_delta_that_disagrees_with_its_conditions_raises(self) -> None:
        payload = {
            "conditions": {
                FLOOR: {"cer_percent": 8.0, "wer_percent": 24.0, "utterances": 10},
                BASELINE: {"cer_percent": 11.0, "wer_percent": 34.0, "utterances": 10},
            },
            "deltas": {BASELINE: 1.0},
        }
        with pytest.raises(CycleError, match="records delta"):
            read_result(payload)

    def test_previews_survive_the_round_trip(self) -> None:
        report = Report((condition(FLOOR, 8.0, previews=(("azul", "azuk"),)),))
        assert read_result(report.as_dict()).floor.previews == (("azul", "azuk"),)

    def test_a_payload_without_conditions_is_refused(self) -> None:
        with pytest.raises(CycleError, match="not a Cycle-CER result"):
            read_result({"task": "12.1"})

    def test_an_empty_conditions_block_is_refused(self) -> None:
        with pytest.raises(CycleError, match="not a Cycle-CER result"):
            read_result({"conditions": {}})

    def test_a_condition_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(CycleError, match="not an object"):
            read_result({"conditions": {FLOOR: 8.0}})

    def test_a_condition_without_utterances_is_refused(self) -> None:
        with pytest.raises(CycleError, match="no integer `utterances`"):
            read_result({"conditions": {FLOOR: {"cer_percent": 8.0}}})

    def test_a_boolean_utterance_count_is_refused(self) -> None:
        """`True` is an `int` in Python, and a payload saying `utterances: true` is
        malformed rather than a set of one."""
        with pytest.raises(CycleError, match="no integer `utterances`"):
            read_result({"conditions": {FLOOR: {"cer_percent": 8.0, "utterances": True}}})

    def test_an_unmeasured_condition_reads_back_unmeasured(self) -> None:
        payload = {
            "conditions": {
                FLOOR: {"cer_percent": 8.0, "wer_percent": 24.0, "utterances": 10},
                BASELINE: {"utterances": 0},
            }
        }
        assert read_result(payload).condition(BASELINE).cer_percent is None
