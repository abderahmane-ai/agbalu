"""Keep speech selection distinct from teacher-forced loss."""

from agbalu.codec_tts.evaluation import (
    ListenerReview,
    SpeechObservation,
    edit_count,
    selection_report,
)


def test_exact_edit_count_not_character_histogram() -> None:
    assert edit_count("abc", "acb") == 2
    assert edit_count("", "abc") == 3
    assert edit_count("ţɛɣ", "ţeɣ") == 1


def test_missing_review_and_metrics_never_become_zero_error() -> None:
    observation = SpeechObservation(
        record_id="a", speaker="s", reference_characters=100, terminated=True
    )
    result = selection_report([observation])
    assert result["eligible"] is False
    assert result["cer_excess"] is None
    assert result["mean_voice_cosine"] is None
    assert result["measured_characters"] == 0


def test_human_expression_failure_blocks_other_good_scores() -> None:
    observations = [
        SpeechObservation(
            record_id="a",
            speaker="s",
            reference_characters=100,
            character_errors=1,
            real_character_errors=1,
            voice_cosine=0.99,
            terminated=True,
        )
    ]
    review = ListenerReview(
        evaluation_sha256="x",
        naturalness_mean=5,
        pronunciation_mean=5,
        voice_retention_fraction=1,
        expression_retained=False,
        language_retained=True,
        ratings=10,
        listeners=2,
    )
    assert selection_report(observations, review)["eligible"] is False
