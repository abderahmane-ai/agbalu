from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.bench.cli import _read_hypotheses, _result_payload
from agbalu.bench.flores import Sentence
from agbalu.bench.mt import (
    DIRECTIONS,
    LANGUAGE_CODE,
    Direction,
    ScoringError,
    as_direction,
    references_for,
    score,
    source_language,
    target_language,
)
from agbalu.normalise import Normaliser

# FLORES+ dev id 5, in its corrupted and repaired forms. Long enough to carry the
# 4-grams BLEU needs — a two-token sentence scores 0.0 BLEU however perfect it is.
CORRECT = (
    "Mi yesɛa Fidal 28 n yiseggasen di leɛmeṛ-is yekcem-d ɣer Lbarṣa "
    "kraḍ n tsemhuyin ɣer deffir, si Sefiya."
)
CORRUPT = (
    "Mi yesεa Fidal 28 n yiseggasen di leεmeṛ-is yekcem-d ɣer Lbarṣa "
    "kraḍ n tsemhuyin ɣer deffir, si Sefiya."
)
OTHER = "Azul fell-awen ay imdanen n tmurt, ansuf yes-wen ɣer wexxam-nneɣ ass-a."


def sentence(text: str, ident: int, split: str) -> Sentence:
    return Sentence(
        id=ident,
        text=text,
        split=split,  # type: ignore[arg-type]
        domain="wikinews",
        topic="News",
        url="",
        last_updated="1.0",
        iso_639_3="kab",
        iso_15924="Latn",
    )


def test_a_correctly_spelled_perfect_translation_is_penalised_by_the_raw_condition() -> None:
    """A flawless translation, losing points only because the reference spells
    `ɛ` U+025B as Greek `ε` U+03B5."""
    result = score([CORRECT], [CORRUPT], "eng-kab", "devtest")
    assert result.raw.get("bleu").score < 100.0
    assert result.normalised.get("bleu").score == 100.0
    assert result.gap("bleu") > 0


def test_an_identical_pair_has_no_gap() -> None:
    result = score([CORRECT], [CORRECT], "eng-kab", "devtest")
    assert result.gap("chrf++") == 0.0
    assert result.raw.get("chrf++").score == 100.0


def test_into_english_is_never_run_through_the_kabyle_normaliser() -> None:
    """Kabyle rules over English is the `Chişinău` defect, so the gap is empty by
    construction for into-foreign directions."""
    hyp = ["The town of Chişinău is the capital."]
    ref = ["The city of Chişinău is the capital."]
    result = score(hyp, ref, "kab-eng", "devtest")
    assert result.gap("bleu") == 0.0
    assert result.raw.get("bleu").score == result.normalised.get("bleu").score


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_every_direction_scores(direction: Direction) -> None:
    result = score([OTHER], [OTHER], direction, "dev")
    assert result.direction == direction
    assert result.sentences == 1


def test_every_language_a_direction_names_has_a_flores_code() -> None:
    """A direction whose language is absent from `LANGUAGE_CODE` raises a `KeyError` on the
    GPU, inside the reference reader, after the model has already been downloaded."""
    named = {source_language(d) for d in DIRECTIONS} | {target_language(d) for d in DIRECTIONS}
    assert named <= set(LANGUAGE_CODE)


def test_the_synthesis_targets_are_into_kabyle_only() -> None:
    """`kab-arb` and its siblings would have to be trained on NLLB's own Kabyle output,
    which caps them at the teacher; they are served by pivoting through `kab-eng`."""
    assert {d for d in DIRECTIONS if source_language(d) in {"arb", "spa", "deu"}} == {
        "arb-kab",
        "spa-kab",
        "deu-kab",
    }
    assert not [d for d in DIRECTIONS if target_language(d) in {"arb", "spa", "deu"}]


@pytest.mark.parametrize(
    ("direction", "source", "target"),
    [
        ("kab-eng", "kab", "eng"),
        ("eng-kab", "eng", "kab"),
        ("kab-fra", "kab", "fra"),
        ("fra-kab", "fra", "kab"),
        ("arb-kab", "arb", "kab"),
        ("spa-kab", "spa", "kab"),
        ("deu-kab", "deu", "kab"),
    ],
)
def test_direction_is_parsed_both_ways(direction: Direction, source: str, target: str) -> None:
    assert source_language(direction) == source
    assert target_language(direction) == target


def test_a_length_mismatch_is_an_error_not_a_truncation() -> None:
    """Silently scoring the overlap produces a plausible number from misaligned data."""
    with pytest.raises(ScoringError, match="2 hypotheses against 1 references"):
        score([CORRECT, OTHER], [CORRECT], "eng-kab", "devtest")


def test_scoring_nothing_is_an_error() -> None:
    with pytest.raises(ScoringError, match="nothing to score"):
        score([], [], "eng-kab", "devtest")


def test_every_metric_carries_its_signature() -> None:
    result = score([CORRECT], [CORRECT], "eng-kab", "devtest")
    for condition in (result.raw, result.normalised):
        names = {m.name for m in condition.metrics}
        assert names == {"chrf++", "bleu", "spbleu"}
        for metric in condition.metrics:
            assert "version:" in metric.signature

    # spBLEU and BLEU are different metrics, not the same number twice.
    assert "flores200" in result.raw.get("spbleu").signature
    assert "13a" in result.raw.get("bleu").signature


def test_chrf_plus_plus_is_not_plain_chrf() -> None:
    """`nw:2` makes it chrF++; `nw:0` would be chrF under the wrong name."""
    result = score([CORRECT], [CORRECT], "eng-kab", "devtest")
    assert "nw:2" in result.raw.get("chrf++").signature


def test_an_unknown_metric_is_an_error() -> None:
    result = score([CORRECT], [CORRECT], "eng-kab", "devtest")
    with pytest.raises(ScoringError, match="no metric named"):
        result.raw.get("comet")


def test_the_normaliser_version_is_carried_into_the_result() -> None:
    engine = Normaliser()
    result = score([CORRECT], [CORRUPT], "eng-kab", "devtest", engine)
    assert result.normaliser_version == engine.version


def test_references_are_ordered_by_split_then_id() -> None:
    """FLORES+ ids restart at 0 per split; ordering on id alone interleaves them."""
    sentences = [
        sentence("devtest-zero", 0, "devtest"),
        sentence("dev-zero", 0, "dev"),
        sentence("devtest-one", 1, "devtest"),
        sentence("dev-one", 1, "dev"),
    ]
    assert references_for(sentences) == ["dev-zero", "dev-one", "devtest-zero", "devtest-one"]


def test_hypotheses_keep_an_interior_blank_line(tmp_path: Path) -> None:
    """An empty translation must stay in place, or every later line is misaligned."""
    path = tmp_path / "hyp.txt"
    path.write_text("first\n\nthird\n", encoding="utf-8")
    assert _read_hypotheses(path) == ["first", "", "third"]


def test_hypotheses_without_a_trailing_newline_keep_the_last_line(tmp_path: Path) -> None:
    path = tmp_path / "hyp.txt"
    path.write_text("first\nsecond", encoding="utf-8")
    assert _read_hypotheses(path) == ["first", "second"]


def test_an_empty_hypotheses_file_reads_as_no_hypotheses(tmp_path: Path) -> None:
    path = tmp_path / "hyp.txt"
    path.write_text("", encoding="utf-8")
    assert _read_hypotheses(path) == []


def test_a_missing_hypotheses_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ScoringError, match="hypotheses not found"):
        _read_hypotheses(tmp_path / "absent.txt")


def test_the_written_result_carries_every_mechanically_checkable_field(tmp_path: Path) -> None:
    """`docs/benchmark.md` §4 items 1-5."""
    result = score([CORRECT], [CORRUPT], "eng-kab", "devtest")
    payload = _result_payload(result, "boffire/marianmt-en-kab@a1b2c3d", tmp_path / "hyp.txt")

    assert payload["model"] == "boffire/marianmt-en-kab@a1b2c3d"
    assert payload["direction"] == "eng-kab"
    assert payload["split"] == "devtest"
    assert str(payload["normaliser_version"]).startswith("1.")

    conditions = payload["conditions"]
    assert isinstance(conditions, dict)
    assert set(conditions) == {"raw", "normalised"}
    for condition in conditions.values():
        for metric in condition.values():
            assert "signature" in metric
            assert "score" in metric

    gaps = payload["orthography_gap"]
    assert isinstance(gaps, dict)
    assert set(gaps) == {"chrf++", "bleu", "spbleu"}
    assert json.dumps(payload)


class TestAsDirection:
    """Narrowing a name off a command line. Membership and the type are one check, so a
    direction the harness cannot score is rejected before a model is downloaded."""

    def test_every_known_direction_narrows_to_itself(self) -> None:
        assert [as_direction(d) for d in DIRECTIONS] == list(DIRECTIONS)

    def test_an_unknown_direction_names_what_is_available(self) -> None:
        with pytest.raises(ScoringError, match="unknown direction 'kab-deu'"):
            as_direction("kab-deu")

    @pytest.mark.parametrize("name", ["", "kab", "kab-eng-fra", "KAB-ENG", " kab-eng"])
    def test_a_malformed_name_is_refused(self, name: str) -> None:
        with pytest.raises(ScoringError, match="unknown direction"):
            as_direction(name)
