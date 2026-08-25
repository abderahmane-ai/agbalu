"""The head, the two calibrations, and the selection that decides what gets published."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from agbalu.bench.sentiment import LABEL_NAMES, NUM_CLASSES, Item
from agbalu.model.config import ModelConfig
from agbalu.model.modeling import Encoder
from agbalu.sentiment.calibrate import (
    CalibrationError,
    ConformalPredictor,
    TemperatureScaler,
    evaluate_calibration,
    expected_calibration_error,
)
from agbalu.sentiment.config import SentimentConfig, SentimentConfigError
from agbalu.sentiment.model import MultiSampleHead, SequenceClassifier, masked_mean
from agbalu.sentiment.release import ReleaseError, calibration_of, config, write_config
from agbalu.sentiment.train import (
    Fitting,
    Split,
    TrainConfig,
    TrainingError,
    average_states,
    parameter_groups,
    save_checkpoint,
    select,
    train_classifier,
)

SMALL = ModelConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_hidden_layers=2,
    max_position_embeddings=32,
    position_bucket_size=8,
)

HEAD = SentimentConfig(hidden_size=16, bottleneck_size=8)


class PieceTokenizer:
    """One id per character, offset past the reserved ids, as SentencePiece would hand back."""

    def encode(self, text: str, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [4 + (ord(character) % 40) for character in text] or [4]


def _model() -> SequenceClassifier:
    torch.manual_seed(0)
    return SequenceClassifier(Encoder(SMALL), HEAD)


def _items(count: int, prefix: str) -> list[Item]:
    return [
        Item(id=f"{prefix}_{index:03d}", text=f"azul {prefix} {index}", label=index % NUM_CLASSES)
        for index in range(count)
    ]


def test_the_head_refuses_a_configuration_it_cannot_build() -> None:
    with pytest.raises(SentimentConfigError, match="hidden_size must be positive"):
        SentimentConfig(hidden_size=0)
    with pytest.raises(SentimentConfigError, match="num_classes must be >= 2"):
        SentimentConfig(num_classes=1)
    with pytest.raises(SentimentConfigError, match="dropouts cannot be empty"):
        SentimentConfig(dropouts=())
    with pytest.raises(SentimentConfigError, match=r"dropout rate must be in \[0, 1\)"):
        SentimentConfig(dropouts=(0.1, 1.0))
    with pytest.raises(SentimentConfigError, match="label_smoothing must be in"):
        SentimentConfig(label_smoothing=1.0)


def test_a_head_sized_for_a_different_encoder_is_refused_at_construction() -> None:
    """Otherwise the mismatch surfaces inside a Linear, naming a tensor and not the pair."""
    with pytest.raises(SentimentConfigError, match="16-dimensional"):
        SequenceClassifier(Encoder(SMALL), SentimentConfig(hidden_size=384))


def test_pooling_ignores_padding_and_a_padded_row_matches_its_unpadded_self() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    padded = masked_mean(hidden, mask)
    unpadded = masked_mean(hidden[:, :2], torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(padded, unpadded)


def test_pooling_an_all_padding_row_is_finite_rather_than_a_division_by_zero() -> None:
    """A row of pure padding cannot arrive from `encode`, which always brackets with
    `[CLS]` and `[SEP]` — but a caller batching by hand can produce one, and a NaN there
    propagates into every logit in the batch through the LayerNorm."""
    hidden = torch.randn(1, 3, 8)
    pooled = masked_mean(hidden, torch.zeros(1, 3, dtype=torch.bool))
    assert bool(torch.isfinite(pooled).all())
    assert torch.equal(pooled, torch.zeros_like(pooled))


def test_the_head_averages_its_dropout_samples_in_training_and_takes_one_path_in_eval() -> None:
    torch.manual_seed(2)
    head = MultiSampleHead(HEAD)
    hidden = torch.randn(4, 6, 16, requires_grad=True)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[0, 4:] = False
    labels = torch.tensor([0, 1, 2, 1])

    head.train()
    logits, loss = head(hidden, mask, labels=labels)
    assert logits.shape == (4, NUM_CLASSES)
    assert loss is not None
    loss.backward()
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all())

    head.eval()
    with torch.no_grad():
        first, _ = head(hidden, mask)
        second, _ = head(hidden, mask)
    assert torch.equal(first, second)


def test_the_head_returns_no_loss_when_it_is_given_no_labels() -> None:
    head = MultiSampleHead(HEAD).eval()
    logits, loss = head(torch.randn(2, 3, 16), torch.ones(2, 3, dtype=torch.bool))
    assert logits.shape == (2, NUM_CLASSES)
    assert loss is None


def test_the_encoder_receives_gradient_through_the_head() -> None:
    model = _model()
    model.train()
    ids = torch.randint(5, SMALL.vocab_size, (2, 7))
    _, loss = model(ids, torch.ones_like(ids, dtype=torch.bool), labels=torch.tensor([0, 2]))
    assert loss is not None
    loss.backward()
    embedding = model.encoder.embedding.word_embedding.weight
    assert embedding.grad is not None
    assert float(embedding.grad.abs().sum()) > 0.0


def test_the_two_learning_rates_reach_the_parts_they_name() -> None:
    model = _model()
    recipe = TrainConfig()
    groups = parameter_groups(model, recipe)
    assert {group["lr"] for group in groups} == {recipe.encoder_lr, recipe.head_lr}
    assert {group["weight_decay"] for group in groups} == {0.0, recipe.weight_decay}
    counted = sum(len(group["params"]) for group in groups)
    assert counted == sum(1 for _ in model.parameters())


def test_norm_gains_and_biases_are_never_weight_decayed() -> None:
    """Decaying a LayerNorm gain pulls it towards zero, and with it the activation it
    normalises. Nothing raises; the fine-tune is just worse."""
    model = _model()
    decayed = {
        id(parameter)
        for group in parameter_groups(model, TrainConfig())
        if group["weight_decay"] > 0.0
        for parameter in group["params"]
    }
    for name, parameter in model.named_parameters():
        if "norm" in name or name.endswith("bias") or "relative_embedding" in name:
            assert id(parameter) not in decayed, name


def test_the_soup_is_the_uniform_average_and_leaves_integer_buffers_alone() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor([7])},
        {"weight": torch.tensor([3.0, 5.0]), "count": torch.tensor([7])},
    ]
    soup = average_states(states)
    assert torch.equal(soup["weight"], torch.tensor([2.0, 4.0]))
    assert soup["count"].dtype == torch.int64
    assert torch.equal(soup["count"], torch.tensor([7]))


def test_averaging_nothing_is_refused_rather_than_returning_an_empty_model() -> None:
    with pytest.raises(TrainingError, match="nothing to average"):
        average_states([])


def test_a_single_seed_is_published_without_a_sweep_summary() -> None:
    model = _model()
    split = Split([[2, 5, 3]], [0])
    fitting = Fitting(model, split, split, TrainConfig(), torch.device("cpu"))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    selected, score, summary = select(fitting, {42: 0.5}, {42: state})
    assert summary is None
    assert score == pytest.approx(0.5)
    assert set(selected) == set(state)


def test_the_soup_is_taken_only_when_it_beats_every_seed_on_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The average is a candidate, not an assumption: on a corpus where the runs land in
    different basins it is worse than all of them, and publishing it anyway is a silent
    loss no reported number would show."""
    model = _model()
    split = Split([[2, 5, 3]], [0])
    fitting = Fitting(model, split, split, TrainConfig(), torch.device("cpu"))
    states = {
        seed: {key: value.clone() for key, value in model.state_dict().items()} for seed in (1, 2)
    }

    monkeypatch.setattr(Fitting, "dev_macro_f1", lambda _self: 0.10)
    _, score, summary = select(fitting, {1: 0.80, 2: 0.60}, states)
    assert summary is not None
    assert summary.selected == "seed_1"
    assert score == pytest.approx(0.80)

    monkeypatch.setattr(Fitting, "dev_macro_f1", lambda _self: 0.90)
    _, better, won = select(fitting, {1: 0.80, 2: 0.60}, states)
    assert won is not None
    assert won.selected == "model_soup"
    assert better == pytest.approx(0.90)
    assert won.mean_dev_macro_f1 == pytest.approx(0.70)


def test_temperature_reduces_calibration_error_and_moves_no_prediction() -> None:
    torch.manual_seed(3)
    labels = torch.randint(0, NUM_CLASSES, (400,))
    logits = torch.randn(400, NUM_CLASSES) * 4.0
    logits[torch.arange(400), labels] += 2.0

    scaler = TemperatureScaler()
    fitted = scaler.fit(logits, labels)
    assert fitted == pytest.approx(scaler.value)
    calibrated = scaler.probabilities(logits)
    assert (
        expected_calibration_error(calibrated, labels)
        <= expected_calibration_error(logits.softmax(dim=-1), labels) + 1e-4
    )
    assert torch.equal(calibrated.argmax(dim=-1), logits.argmax(dim=-1))


def test_the_reported_temperature_is_the_one_that_is_applied() -> None:
    """The clamp is in `scale`, so an unclamped parameter written into `config.json` is a
    number the published model does not use."""
    scaler = TemperatureScaler(temperature=1e-6)
    assert scaler.value > 1e-6
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    assert torch.allclose(scaler.scale(logits), logits / scaler.value)


def test_calibration_refuses_an_empty_split_rather_than_inventing_a_constant() -> None:
    empty = torch.zeros((0, NUM_CLASSES))
    with pytest.raises(CalibrationError, match="empty split"):
        TemperatureScaler().fit(empty, torch.zeros(0, dtype=torch.long))
    with pytest.raises(CalibrationError, match="empty split"):
        ConformalPredictor().fit(empty, torch.zeros(0, dtype=torch.long))
    with pytest.raises(CalibrationError, match="empty split"):
        expected_calibration_error(empty, torch.zeros(0, dtype=torch.long))
    with pytest.raises(CalibrationError, match=r"alpha must be in \(0, 1\)"):
        ConformalPredictor().fit(torch.tensor([[0.5, 0.3, 0.2]]), torch.tensor([0]), alpha=0.0)


def test_the_conformal_set_covers_the_target_rate_on_data_it_did_not_see() -> None:
    torch.manual_seed(4)
    labels = torch.randint(0, NUM_CLASSES, (600,))
    probabilities = torch.softmax(
        torch.randn(600, NUM_CLASSES) + torch.eye(NUM_CLASSES)[labels] * 2.0, dim=-1
    )
    conformal = ConformalPredictor()
    conformal.fit(probabilities[:300], labels[:300], alpha=0.10)

    sets = conformal.predict_set(probabilities[300:])
    covered = sum(
        1 for gold, members in zip(labels[300:].tolist(), sets, strict=True) if gold in members
    )
    assert 0.85 <= covered / len(sets) <= 1.0
    assert all(members for members in sets)


def test_an_unfitted_conformal_predictor_returns_the_argmax_and_says_so() -> None:
    conformal = ConformalPredictor()
    assert conformal.threshold is None
    sets = conformal.predict_set(torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.2, 0.1]]))
    assert sets == [(1,), (0,)]


def test_the_conformal_quantile_uses_the_finite_sample_correction() -> None:
    """`ceil((n + 1)(1 - alpha))` and not `n(1 - alpha)`. At n = 19 and alpha = 0.05 the
    corrected rank is 19 and the naive one is 18, which is the whole guarantee."""
    probabilities = torch.full((19, NUM_CLASSES), 1 / 3)
    probabilities[:, 0] = torch.linspace(0.10, 0.90, 19)
    labels = torch.zeros(19, dtype=torch.long)
    conformal = ConformalPredictor()
    q_hat = conformal.fit(probabilities, labels, alpha=0.05)
    scores = sorted((1.0 - probabilities[:, 0]).tolist())
    assert q_hat == pytest.approx(scores[math.ceil(20 * 0.95) - 1])


def test_the_quantile_saturates_rather_than_indexing_past_the_split() -> None:
    """At n = 5 and alpha = 0.05 the corrected rank is 6, which does not exist. The honest
    answer is a set containing everything, not the largest score seen."""
    probabilities = torch.full((5, NUM_CLASSES), 1 / 3)
    labels = torch.zeros(5, dtype=torch.long)
    assert ConformalPredictor().fit(probabilities, labels, alpha=0.05) == pytest.approx(1.0)


def test_the_calibration_report_is_measured_on_the_split_it_is_given() -> None:
    scaler = TemperatureScaler(temperature=1.2)
    conformal = ConformalPredictor(q_hat=0.9)
    logits = torch.tensor([[3.0, 0.5, -1.0], [-0.5, 3.0, 0.1]])
    metrics = evaluate_calibration(logits, torch.tensor([0, 1]), scaler, conformal)
    assert metrics.temperature == pytest.approx(1.2)
    assert metrics.coverage == pytest.approx(1.0)
    assert metrics.mean_set_size >= 1.0
    assert 0.0 <= metrics.ambiguity_rate <= 1.0
    assert set(metrics.as_dict()) == {
        "temperature",
        "q_hat",
        "raw_ece",
        "calibrated_ece",
        "coverage",
        "mean_set_size",
        "ambiguity_rate",
    }


def test_a_fine_tune_over_one_seed_produces_a_report_and_a_checkpoint(tmp_path: Path) -> None:
    splits = {"train": _items(24, "train"), "dev": _items(9, "dev"), "test": _items(9, "test")}
    model = _model()
    report, scaler, conformal = train_classifier(
        model,
        PieceTokenizer(),
        splits,
        device=torch.device("cpu"),
        config=TrainConfig(epochs=1, batch_size=8, seeds=(11,)),
    )
    assert report.sweep is None
    assert 0.0 <= report.accuracy <= 1.0
    assert sum(sum(row) for row in report.confusion) == len(splits["test"])
    assert set(report.per_class) == set(LABEL_NAMES)
    assert scaler.value > 0.0
    assert conformal.q_hat is not None

    save_checkpoint(tmp_path, model, report)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evaluation.json", "model.pt"]
    written = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert written["calibration"]["temperature"] == pytest.approx(scaler.value, rel=1e-4)
    assert set(torch.load(tmp_path / "model.pt", weights_only=True)) == {"model_state_dict"}


def test_training_refuses_a_split_that_was_never_built() -> None:
    with pytest.raises(TrainingError, match=r"no rows for \['test'\]"):
        train_classifier(
            _model(),
            PieceTokenizer(),
            {"train": _items(4, "train"), "dev": _items(2, "dev"), "test": []},
            device=torch.device("cpu"),
        )


def test_training_refuses_a_recipe_with_no_seed() -> None:
    with pytest.raises(TrainingError, match="at least one seed"):
        train_classifier(
            _model(),
            PieceTokenizer(),
            {"train": _items(4, "t"), "dev": _items(2, "d"), "test": _items(2, "e")},
            device=torch.device("cpu"),
            config=TrainConfig(seeds=()),
        )


def test_the_published_config_carries_the_calibration_the_run_measured() -> None:
    written = config(temperature=0.94, q_hat=0.27)
    assert written["temperature"] == pytest.approx(0.94)
    assert written["q_hat"] == pytest.approx(0.27)
    assert written["labels"] == list(LABEL_NAMES)
    assert written["hidden_size"] == 384


def test_a_report_with_no_calibration_block_cannot_be_published() -> None:
    """A config defaulted to temperature 1.0 and no quantile is a model that silently does
    neither correction, and from the outside it looks exactly like a calibrated one."""
    with pytest.raises(ReleaseError, match="no calibration block"):
        calibration_of({"accuracy": 0.9})
    with pytest.raises(ReleaseError, match="no quantile"):
        calibration_of({"calibration": {"temperature": 1.0}})
    with pytest.raises(ReleaseError, match="no temperature"):
        calibration_of({"calibration": {"q_hat": 0.8}})


def test_the_quantile_is_read_under_the_name_the_run_that_trained_it_used() -> None:
    """The published checkpoint's report calls it `conformal_quantile`. Renaming a JSON key
    is not worth an A10, so both spellings resolve to the same number."""
    assert calibration_of({"calibration": {"temperature": 0.98, "q_hat": 0.80}}) == (0.98, 0.80)
    assert calibration_of({"calibration": {"temperature": 0.98, "conformal_quantile": 0.80}}) == (
        0.98,
        0.80,
    )


def test_the_release_config_is_refused_when_the_report_is_not_beside_the_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReleaseError, match="no evaluation report"):
        write_config(tmp_path)
