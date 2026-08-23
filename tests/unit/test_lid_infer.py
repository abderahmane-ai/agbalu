from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from agbalu.bench.lid import Identifier as IdentifierProtocol
from agbalu.lid.calibrate import Calibration
from agbalu.lid.config import ModelConfig, TrainConfig
from agbalu.lid.infer import Identifier, InferenceError
from agbalu.lid.model import CharClassifier

LANGUAGES = ("kab_Latn", "rif_Latn", "NOT_AMAZIGH")


def config() -> ModelConfig:
    return ModelConfig(
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        intermediate_dim=64,
        conv_dim=8,
        projection_dim=16,
        max_length=64,
        classes=LANGUAGES,
    )


def calibration(q_hat: float = 0.9) -> Calibration:
    return Calibration(
        target_coverage=0.99,
        q_hat=q_hat,
        fitted_on=50,
        measured_on=50,
        empirical_coverage=0.99,
        mean_set_size=1.2,
        classes=LANGUAGES,
        prior_shift=(0.0, 0.0, 0.0),
    )


def staged(tmp_path: Path) -> Path:
    """A checkpoint directory shaped exactly as the trainer leaves one."""
    model = CharClassifier(config())
    bundle = torch.nn.Module()
    bundle.add_module("model", model)
    torch.save(
        {
            "model": bundle.state_dict(),
            "optimizer": {},
            "state": {"step": 10},
            "rng": {},
        },
        tmp_path / "best.pt",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model": asdict(config()), "train": asdict(TrainConfig())}), encoding="utf-8"
    )
    return tmp_path


class TestLoading:
    def test_a_staged_checkpoint_loads(self, tmp_path: Path) -> None:
        identifier = Identifier.load(staged(tmp_path))
        assert identifier.classes == LANGUAGES

    def test_the_training_only_submodules_are_not_required(self, tmp_path: Path) -> None:
        """`best.pt` bundles the group weights and the memory bank; an exported classifier
        must load without them."""
        directory = staged(tmp_path)
        payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        payload["model"]["group_weights.weights"] = torch.ones(4)
        payload["model"]["contrastive.bank_fill"] = torch.zeros((), dtype=torch.long)
        torch.save(payload, directory / "best.pt")
        assert Identifier.load(directory).classes == LANGUAGES

    def test_a_missing_checkpoint_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(InferenceError, match="checkpoint not found"):
            Identifier.load(tmp_path)

    def test_a_missing_config_is_named_rather_than_guessed(self, tmp_path: Path) -> None:
        directory = staged(tmp_path)
        (directory / "config.json").unlink()
        with pytest.raises(InferenceError, match="config not found"):
            Identifier.load(directory)

    def test_a_calibration_covering_other_classes_is_refused(self) -> None:
        model = CharClassifier(config())
        wrong = Calibration(
            target_coverage=0.99,
            q_hat=0.5,
            fitted_on=10,
            measured_on=10,
            empirical_coverage=0.99,
            mean_set_size=1.0,
            classes=("a", "b"),
            prior_shift=(0.0, 0.0),
        )
        with pytest.raises(InferenceError, match="calibration covers 2 classes"):
            Identifier(model, calibration=wrong)


class TestPrediction:
    def test_the_protocol_the_benchmark_scores_is_satisfied(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        assert isinstance(identifier, IdentifierProtocol)
        assert identifier.labels == frozenset(LANGUAGES)

    def test_every_input_gets_exactly_one_label_in_order(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        texts = ["awal ameqqran", "hello there", "ur ten-teg ca"]
        assert len(identifier.identify(texts)) == 3

    def test_probabilities_sum_to_one_per_row(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        rows = identifier.probabilities(["one line", "another line"])
        assert torch.allclose(rows.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_no_input_returns_nothing_rather_than_raising(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        assert identifier.predict([]) == []
        assert identifier.probabilities([]).shape == (0, len(LANGUAGES))

    def test_an_empty_string_is_classified_rather_than_crashing(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        assert identifier.predict([""])[0].language in LANGUAGES

    def test_batching_does_not_change_a_row_s_answer(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        texts = [f"line number {index} of the batch" for index in range(9)]
        assert identifier.identify(texts) == [identifier.identify([t])[0] for t in texts]

    def test_without_a_calibration_the_set_is_the_single_top_label(self) -> None:
        prediction = Identifier(CharClassifier(config())).predict(["awal"])[0]
        assert prediction.prediction_set == (prediction.language,)

    def test_a_wide_quantile_admits_every_class(self) -> None:
        identifier = Identifier(CharClassifier(config()), calibration=calibration(q_hat=1.0))
        assert len(identifier.predict(["awal ameqqran"])[0].prediction_set) == len(LANGUAGES)

    def test_a_narrow_quantile_never_yields_an_empty_set(self) -> None:
        """An empty set reads as `no answer` at every call site; the top label stands."""
        identifier = Identifier(CharClassifier(config()), calibration=calibration(q_hat=0.0))
        prediction = identifier.predict(["awal ameqqran"])[0]
        assert prediction.prediction_set == (prediction.language,)

    def test_the_negative_class_is_reported_as_not_amazigh(self) -> None:
        identifier = Identifier(CharClassifier(config()))
        prediction = identifier.predict(["some text"])[0]
        assert prediction.is_amazigh == (prediction.language != "NOT_AMAZIGH")

    @pytest.mark.parametrize(
        "text", ["﻿bom", "zero\u200bwidth", "null\x00byte", "🙂🙂🙂", "ⵜⴰⵎⴰⵣⵉⵖⵜ"]
    )
    def test_awkward_unicode_is_classified_without_raising(self, text: str) -> None:
        identifier = Identifier(CharClassifier(config()))
        assert identifier.predict([text])[0].language in LANGUAGES
