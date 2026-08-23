from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agbalu.lid.calibrate import (
    Calibration,
    CalibrationError,
    adjusted_probabilities,
    calibrate,
    conformal_quantile,
    load,
    save,
)
from agbalu.lid.config import ModelConfig, TrainConfig
from agbalu.lid.dataset import Corpus, Record
from agbalu.lid.model import CharClassifier

LANGUAGES = ("kab_Latn", "rif_Latn", "tzm_Latn")


def corpus(count: int = 120) -> Corpus:
    config = ModelConfig(
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        intermediate_dim=64,
        conv_dim=8,
        projection_dim=16,
        max_length=64,
        classes=LANGUAGES,
    )
    records = [
        Record(
            text=f"{language} line number {index} of this small corpus",
            language=language,
            domain="web",
            group=f"{language}/web",
            source="s",
        )
        for index in range(count)
        for language in LANGUAGES
    ]
    return Corpus(records, config)


def calibration() -> Calibration:
    return Calibration(
        target_coverage=0.99,
        q_hat=0.87,
        fitted_on=60,
        measured_on=60,
        empirical_coverage=0.983,
        mean_set_size=1.4,
        classes=LANGUAGES,
        prior_shift=(-0.1, -0.2, -0.9),
    )


class TestConformalQuantile:
    def test_the_quantile_is_the_finite_sample_rank(self) -> None:
        scores = torch.arange(100, dtype=torch.float) / 100
        assert conformal_quantile(scores, 0.9) == pytest.approx(0.90)

    def test_full_coverage_takes_the_largest_score(self) -> None:
        scores = torch.tensor([0.1, 0.5, 0.9])
        assert conformal_quantile(scores, 1.0) == pytest.approx(0.9)

    def test_no_scores_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="no calibration scores"):
            conformal_quantile(torch.zeros(0), 0.9)


class TestPriorAdjustment:
    def test_a_zero_shift_is_a_plain_softmax(self) -> None:
        logits = torch.randn(4, 3)
        assert torch.allclose(
            adjusted_probabilities(logits, torch.zeros(3)), torch.softmax(logits, dim=-1)
        )

    def test_the_shift_moves_probability_towards_the_rare_class(self) -> None:
        """`logit_k - tau * log(prior_k)` with a negative shift on the rare class raises it."""
        logits = torch.zeros(1, 3)
        shift = torch.tensor([0.0, 0.0, -2.0])
        adjusted = adjusted_probabilities(logits, shift)
        assert float(adjusted[0, 2]) > float(adjusted[0, 0])

    def test_the_rows_still_sum_to_one(self) -> None:
        adjusted = adjusted_probabilities(torch.randn(5, 3), torch.tensor([-0.1, -1.0, -3.0]))
        assert torch.allclose(adjusted.sum(dim=1), torch.ones(5), atol=1e-6)


class TestCalibrate:
    def test_the_quantile_is_fitted_and_measured_on_disjoint_halves(self) -> None:
        """Measuring coverage on the rows that set the quantile reports the fit, not the
        guarantee."""
        data = corpus()
        model = CharClassifier(data.config)
        result = calibrate(model, data, settings=TrainConfig(), target_coverage=0.9)
        assert result.fitted_on + result.measured_on == len(data)
        assert abs(result.fitted_on - result.measured_on) <= 1

    def test_the_realised_coverage_is_reported_rather_than_assumed(self) -> None:
        data = corpus()
        model = CharClassifier(data.config)
        result = calibrate(model, data, settings=TrainConfig(), target_coverage=0.95)
        assert 0.0 <= result.empirical_coverage <= 1.0
        assert 0.0 < result.mean_set_size <= len(LANGUAGES)

    def test_a_higher_target_never_shrinks_the_quantile(self) -> None:
        data = corpus()
        model = CharClassifier(data.config)
        settings = TrainConfig()
        loose = calibrate(model, data, settings=settings, target_coverage=0.8)
        tight = calibrate(model, data, settings=settings, target_coverage=0.99)
        assert tight.q_hat >= loose.q_hat

    def test_the_prior_shift_is_recorded_so_inference_reproduces_it(self) -> None:
        data = corpus()
        result = calibrate(CharClassifier(data.config), data, settings=TrainConfig())
        assert len(result.prior_shift) == len(LANGUAGES)
        assert all(value <= 0.0 for value in result.prior_shift)

    def test_too_little_data_is_refused_rather_than_quantified(self) -> None:
        data = corpus(count=2)
        with pytest.raises(CalibrationError, match="at least 40 rows"):
            calibrate(CharClassifier(data.config), data, settings=TrainConfig())

    @pytest.mark.parametrize("coverage", [0.0, 1.0, -0.5, 2.0])
    def test_an_impossible_coverage_target_is_refused(self, coverage: float) -> None:
        data = corpus()
        with pytest.raises(CalibrationError, match="target_coverage must be"):
            calibrate(
                CharClassifier(data.config), data, settings=TrainConfig(), target_coverage=coverage
            )


class TestPersistence:
    def test_a_calibration_round_trips(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "calibration.json"
        save(calibration(), target)
        assert load(target) == calibration()

    def test_the_classes_and_shift_come_back_as_tuples(self, tmp_path: Path) -> None:
        target = tmp_path / "calibration.json"
        save(calibration(), target)
        restored = load(target)
        assert isinstance(restored.classes, tuple)
        assert isinstance(restored.prior_shift, tuple)

    def test_a_missing_calibration_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="calibration not found"):
            load(tmp_path / "absent.json")
