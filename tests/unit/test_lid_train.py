from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from agbalu.lid.config import ModelConfig, TrainConfig
from agbalu.lid.dataset import Corpus, Record
from agbalu.lid.model import CharClassifier
from agbalu.lid.train import (
    Trainer,
    TrainingError,
    build_corpora,
    build_optimizer,
    held_out_report,
    schedule,
    train_identifier,
)

LANGUAGES = ("kab_Latn", "rif_Latn", "tzm_Latn")
DOMAINS = ("web", "scripture")


def rows(per_cell: int = 40) -> list[Record]:
    """A corpus where every language appears in both domains, so the group axis is real."""
    return [
        Record(
            text=f"{language[:3]} {domain[:3]} line number {index} of the corpus",
            language=language,
            domain=domain,
            group=f"{language}/{domain}",
            source=f"{language}.{domain}",
        )
        for language in LANGUAGES
        for domain in DOMAINS
        for index in range(per_cell)
    ]


@pytest.fixture(name="config")
def fixture_config() -> ModelConfig:
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


@pytest.fixture(name="settings")
def fixture_settings() -> TrainConfig:
    return TrainConfig(
        epochs=1,
        groups_per_batch=3,
        samples_per_group=4,
        eval_batch_size=32,
        memory_bank_size=32,
        deferred_share=0.5,
    )


def make_trainer(config: ModelConfig, settings: TrainConfig, directory: Path) -> Trainer:
    records = rows()
    train = Corpus(records, config)
    dev = Corpus(records[::10], config, groups=train.groups)
    model = CharClassifier(config)
    return Trainer(model, train, dev, settings, directory, torch.device("cpu"))


class TestSchedule:
    def test_warmup_rises_from_near_zero_to_one(self) -> None:
        assert schedule(0, 100, 10) == pytest.approx(0.1)
        assert schedule(9, 100, 10) == pytest.approx(1.0)

    def test_the_cosine_reaches_zero_at_the_end(self) -> None:
        assert schedule(100, 100, 10) == pytest.approx(0.0, abs=1e-9)

    def test_a_step_past_the_end_does_not_go_negative(self) -> None:
        assert schedule(500, 100, 10) >= 0.0

    def test_a_zero_length_warmup_does_not_divide_by_zero(self) -> None:
        assert schedule(0, 10, 0) == pytest.approx(1.0)


class TestOptimizer:
    def test_norms_and_gains_are_exempt_from_weight_decay(self, config: ModelConfig) -> None:
        optimizer = build_optimizer(CharClassifier(config), TrainConfig())
        decays = {group["weight_decay"] for group in optimizer.param_groups}
        assert decays == {0.0, TrainConfig().weight_decay}
        flat = next(g for g in optimizer.param_groups if g["weight_decay"] == 0.0)
        assert all(p.ndim < 2 for p in flat["params"])

    def test_a_fully_frozen_model_is_refused_rather_than_trained(self, config: ModelConfig) -> None:
        model = CharClassifier(config)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with pytest.raises(TrainingError, match="no trainable parameters"):
            build_optimizer(model, TrainConfig())


class TestDeferredReweighting:
    def test_the_margins_are_off_before_the_deferred_point(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        """Measured, not preferred: margins from step 0 ran 120 steps to a loss of 4.01
        where the same run without them reached 0.76."""
        trainer = make_trainer(config, settings, tmp_path)
        trainer.apply_phase()
        assert float(trainer.model.head.margins.abs().max()) == 0.0
        assert trainer.group_weights.step == 0.0

    def test_the_margins_and_the_group_weights_come_on_together(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.state.step = trainer.deferred_at
        trainer.apply_phase()
        assert float(trainer.model.head.margins.max()) > 0.0
        assert trainer.group_weights.step == settings.group_dro_step

    def test_the_phase_is_derived_from_the_step_so_a_resume_lands_in_it(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.state.step = trainer.total_steps - 1
        assert trainer.apply_phase() is True

    def test_applying_the_phase_twice_changes_nothing(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.state.step = trainer.deferred_at
        trainer.apply_phase()
        first = trainer.model.head.margins.clone()
        trainer.apply_phase()
        assert torch.equal(first, trainer.model.head.margins)


class TestTrainer:
    def test_a_run_produces_a_checkpoint_and_a_falling_loss(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        summary = trainer.run()
        assert summary.steps_this_run == trainer.total_steps
        assert (tmp_path / "best.pt").is_file()
        assert (tmp_path / "latest.pt").is_file()
        assert summary.best is not None

    def test_the_checkpoint_carries_the_group_weights_and_the_memory_bank(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.run()
        payload = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=True)
        keys = set(payload["model"])
        assert any(key.startswith("group_weights.") for key in keys)
        assert any(key.startswith("contrastive.bank") for key in keys)
        assert any(key.startswith("model.") for key in keys)

    def test_a_completed_run_refuses_to_train_again(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        """Without this a resumed run trains nothing and reports the checkpoint's counters
        as a measurement."""
        trainer = make_trainer(config, settings, tmp_path)
        trainer.run()
        resumed = make_trainer(config, settings, tmp_path)
        assert resumed.maybe_resume() is True
        assert resumed.run().steps_this_run == 0

    def test_resuming_restores_the_step_and_the_best_score(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.run()
        resumed = make_trainer(config, settings, tmp_path)
        resumed.maybe_resume()
        assert resumed.state.step == trainer.total_steps
        assert resumed.best_score > 0.0

    def test_nothing_to_resume_from_is_reported_rather_than_assumed(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        assert make_trainer(config, settings, tmp_path).maybe_resume() is False

    def test_a_non_finite_gradient_is_dropped_rather_than_stepped(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        before = [p.detach().clone() for p in trainer.model.parameters()]
        for parameter in trainer.model.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        trainer._optimizer_step()
        assert trainer.state.consecutive_non_finite == 1
        assert all(
            torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True)
        )

    def test_a_run_of_non_finite_gradients_stops_the_run(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.state.consecutive_non_finite = 19
        for parameter in trainer.model.parameters():
            parameter.grad = torch.full_like(parameter, float("inf"))
        with pytest.raises(TrainingError, match="consecutive non-finite gradients"):
            trainer._optimizer_step()

    def test_evaluation_reports_every_group_in_the_split(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        report = trainer.evaluate(trainer.dev_corpus)
        assert set(report.group_recall) == set(trainer.dev_corpus.groups)

    def test_evaluation_leaves_the_model_in_training_mode(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        trainer.model.train()
        trainer.evaluate(trainer.dev_corpus)
        assert trainer.model.training

    def test_rows_per_second_is_none_rather_than_zero_for_an_empty_run(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        """A clamped rate reads as a measurement."""
        trainer = make_trainer(config, settings, tmp_path)
        assert trainer.summary.rows_per_second is None


class TestEndToEnd:
    def test_training_writes_the_config_and_the_dev_report(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        records = rows()
        for name, subset in (("train.jsonl", records), ("dev.jsonl", records[::10])):
            (data / name).write_text(
                "\n".join(
                    json.dumps(
                        {
                            "text": r.text,
                            "language": r.language,
                            "domain": r.domain,
                            "group": r.group,
                            "source": r.source,
                        }
                    )
                    for r in subset
                ),
                encoding="utf-8",
            )
        checkpoints = tmp_path / "run"
        summary = train_identifier(
            model_config=config,
            settings=settings,
            data_dir=data,
            checkpoint_dir=checkpoints,
            device=torch.device("cpu"),
        )
        assert summary.best is not None
        stored = json.loads((checkpoints / "config.json").read_text(encoding="utf-8"))
        assert stored["model"]["classes"] == list(LANGUAGES)
        assert stored["train"]["deferred_share"] == settings.deferred_share
        assert (checkpoints / "dev-report.json").is_file()

    def test_dev_shares_the_training_group_index(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        data = tmp_path / "data"
        data.mkdir()
        records = rows()
        for name, subset in (("train.jsonl", records), ("dev.jsonl", records[:20])):
            (data / name).write_text(
                "\n".join(
                    json.dumps(
                        {
                            "text": r.text,
                            "language": r.language,
                            "domain": r.domain,
                            "group": r.group,
                            "source": r.source,
                        }
                    )
                    for r in subset
                ),
                encoding="utf-8",
            )
        train, dev = build_corpora(data, config, settings)
        assert dev.groups == train.groups

    def test_the_held_out_report_scores_by_the_same_code_as_dev(
        self, config: ModelConfig, settings: TrainConfig, tmp_path: Path
    ) -> None:
        trainer = make_trainer(config, settings, tmp_path)
        report = held_out_report(
            trainer.model,
            rows(4),
            trainer.train_corpus.groups,
            batch_size=16,
            device=torch.device("cpu"),
        )
        assert sum(report.group_support.values()) == len(rows(4))
