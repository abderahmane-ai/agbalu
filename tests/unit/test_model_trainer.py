"""Trainer-level failure drills: preemption, divergence and rollback.

`test_model_checkpoint.py` proves a checkpoint survives; this proves the trainer *uses*
it — that a killed run resumes where it stopped, and that a run whose loss goes non-finite
recovers instead of writing garbage weights over a good checkpoint.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from agbalu.model.checkpoint import BEST, LATEST, ResumeFrom, TrainingState, load, save
from agbalu.model.config import PRESETS, RunConfig, TrainConfig
from agbalu.model.data import PackedDataset
from agbalu.model.lock import RunLockedError, acquire
from agbalu.model.lock import read as read_lock
from agbalu.model.trainer import (
    MAX_NON_FINITE,
    RunSummary,
    Trainer,
    build_optimizer,
    quieten_dynamo,
    select_device,
)

TINY_MODEL = replace(
    PRESETS["kab"],
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_hidden_layers=2,
)


def tiny_config(max_steps: int = 4, *, compile_model: bool = False) -> RunConfig:
    train = TrainConfig(
        seq_length=16,
        local_batch_size=4,
        global_batch_size=8,
        max_steps=max_steps,
        validate_every=1_000_000,
        log_every=1_000_000,
        save_every=2,
        mixed_precision=False,
        compile=compile_model,
    )
    return RunConfig(model=TINY_MODEL, train=train, name="drill")


def tiny_dataset(config: RunConfig, tokens: int = 4_000) -> PackedDataset:
    rng = np.random.default_rng(0)
    stream = rng.integers(5, TINY_MODEL.vocab_size, size=tokens, dtype=np.uint16)
    return PackedDataset(
        stream, config.train, TINY_MODEL.vocab_size, mask_token_id=4, n_special_tokens=5
    )


def _read_events(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def make_trainer(
    out: Path,
    config: RunConfig | None = None,
    *,
    compile_model: bool = False,
    resume_from: ResumeFrom = "latest",
) -> Trainer:
    config = config or tiny_config(compile_model=compile_model)
    return Trainer(
        config,
        tiny_dataset(config),
        tiny_dataset(config, 1_000),
        out,
        device=torch.device("cpu"),
        resume_from=resume_from,
    )


class TestDeviceSelection:
    def test_an_explicit_request_is_honoured(self) -> None:
        assert select_device("cpu") == torch.device("cpu")

    def test_the_default_is_a_real_device(self) -> None:
        assert select_device().type in {"cuda", "mps", "cpu"}


class TestOptimizerGroups:
    def test_norms_and_biases_are_excluded_from_decay(self, tmp_path: Path) -> None:
        config = tiny_config()
        trainer = make_trainer(tmp_path, config)
        optimizer = build_optimizer(trainer.model, config)
        decay, no_decay = optimizer.param_groups
        assert decay["weight_decay"] == config.train.weight_decay
        assert no_decay["weight_decay"] == 0.0
        assert len(decay["params"]) > 0
        assert len(no_decay["params"]) > 0

    def test_every_trainable_parameter_lands_in_exactly_one_group(self, tmp_path: Path) -> None:
        config = tiny_config()
        trainer = make_trainer(tmp_path, config)
        optimizer = build_optimizer(trainer.model, config)
        grouped = sum(len(group["params"]) for group in optimizer.param_groups)
        trainable = sum(1 for p in trainer.model.parameters() if p.requires_grad)
        assert grouped == trainable


class TestRunSummary:
    def test_a_measured_rate_is_tokens_over_training_seconds(self) -> None:
        summary = RunSummary(
            steps=4, tokens=800, masked=200, training_seconds=2.0, wall_seconds=4.0
        )
        assert summary.training_tokens_per_second == 400.0
        assert summary.wall_clock_tokens_per_second == 200.0

    @pytest.mark.parametrize(
        ("steps", "seconds"),
        [(0, 2.0), (4, 0.0), (0, 0.0)],
        ids=["no steps", "no elapsed time", "neither"],
    )
    def test_an_unmeasurable_rate_is_none_not_zero(self, steps: int, seconds: float) -> None:
        summary = RunSummary(
            steps=steps, tokens=0, masked=0, training_seconds=seconds, wall_seconds=seconds
        )
        assert summary.training_tokens_per_second is None
        assert summary.wall_clock_tokens_per_second is None


class TestTraining:
    def test_a_short_run_reaches_the_step_count(self, tmp_path: Path) -> None:
        state = make_trainer(tmp_path).train()
        assert state.step == 4
        assert state.tokens_seen > 0

    def test_a_fresh_run_measures_its_own_throughput(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        state = trainer.train()
        summary = trainer.summary
        assert summary.steps == 4
        assert summary.tokens == state.tokens_seen
        rate = summary.training_tokens_per_second
        assert rate is not None
        assert rate > 0.0

    def test_masked_positions_are_counted_alongside_tokens(self, tmp_path: Path) -> None:
        """Scored positions and tokens read are different counts, differing by the mask
        rate. Throughput is per token read; reporting it per scored position overstates
        it by roughly 1/mask_p."""
        trainer = make_trainer(tmp_path)
        trainer.train()
        summary = trainer.summary
        assert 0 < summary.masked < summary.tokens

    def test_validation_time_is_excluded_from_the_training_rate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the rate describes the run's wall clock rather than the training loop,
        and the two differ several-fold on a short run that validates often."""
        base = tiny_config()
        config = replace(base, train=replace(base.train, validate_every=1))
        trainer = Trainer(
            config,
            tiny_dataset(config),
            tiny_dataset(config, 1_000),
            tmp_path,
            device=torch.device("cpu"),
        )
        delay = 0.05

        def slow_validate(_batches: int | None = None) -> float:
            time.sleep(delay)
            return 1.0

        monkeypatch.setattr(trainer, "validate", slow_validate)
        trainer.train()
        summary = trainer.summary
        assert summary.wall_seconds - summary.training_seconds >= config.train.max_steps * delay

    def test_the_run_writes_a_resumable_checkpoint(self, tmp_path: Path) -> None:
        make_trainer(tmp_path).train()
        assert (tmp_path / LATEST).is_file()
        assert (tmp_path / f"{LATEST}.sha256").is_file()

    def test_metrics_are_one_json_event_per_line(self, tmp_path: Path) -> None:
        make_trainer(tmp_path).train()
        events = [
            json.loads(line)
            for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert events[0]["event"] == "start"
        assert events[-1]["event"] == "finish"
        assert sum(1 for e in events if e["event"] == "step") == 4


class TestPreemption:
    def test_a_resumed_run_continues_rather_than_restarting(self, tmp_path: Path) -> None:
        """The preemption drill: the container dies, Modal retries, and the second
        attempt must not redo the first attempt's work."""
        first = make_trainer(tmp_path, tiny_config(max_steps=4))
        first.train()

        second = make_trainer(tmp_path, tiny_config(max_steps=6))
        assert second.maybe_resume()
        assert second.state.step == 4
        final = second.train()
        assert final.step == 6

    def test_a_continuation_can_start_from_best_rather_than_latest(self, tmp_path: Path) -> None:
        first = make_trainer(tmp_path, tiny_config(max_steps=4))
        first.train()
        save(tmp_path, first.model, first.optimizer, TrainingState(step=2), name=BEST)

        second = make_trainer(tmp_path, tiny_config(max_steps=6), resume_from="best")
        assert second.maybe_resume() == BEST
        assert second.state.step == 2

    def test_a_corrupt_latest_resumes_from_best_instead_of_raising(self, tmp_path: Path) -> None:
        """The trainer must load the checkpoint `resume_point` chose. Loading a fixed
        `latest` raises on the very checksum the fallback exists to survive."""
        first = make_trainer(tmp_path, tiny_config(max_steps=4))
        first.train()
        save(tmp_path, first.model, first.optimizer, TrainingState(step=2), name=BEST)
        (tmp_path / LATEST).write_bytes(b"shredded")

        second = make_trainer(tmp_path, tiny_config(max_steps=6))
        assert second.maybe_resume() == BEST
        assert second.state.step == 2

    def test_an_already_finished_run_is_a_no_op_on_retry(self, tmp_path: Path) -> None:
        config = tiny_config(max_steps=4)
        make_trainer(tmp_path, config).train()
        again = make_trainer(tmp_path, config)
        assert again.train().step == 4

    def test_a_no_op_retry_reports_no_throughput_rather_than_zero(self, tmp_path: Path) -> None:
        """The smoke that trained nothing: it resumed at step 20 of 20, so `last window`
        throughput was 0.0, and a `max(rate, 1.0)` clamp downstream turned that into a
        projected 11.6 million hours. A rate that was never measured must read as absent."""
        config = tiny_config(max_steps=4)
        make_trainer(tmp_path, config).train()

        again = make_trainer(tmp_path, config)
        again.train()
        assert again.summary.steps == 0
        assert again.summary.tokens == 0
        assert again.summary.training_tokens_per_second is None
        assert again.summary.wall_clock_tokens_per_second is None

    def test_a_no_op_retry_records_the_reason_in_the_metrics(self, tmp_path: Path) -> None:
        config = tiny_config(max_steps=4)
        make_trainer(tmp_path, config).train()
        make_trainer(tmp_path, config).train()
        events = [
            json.loads(line)
            for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert events[-1] == {"event": "noop", "step": 4, "max_steps": 4}

    def test_a_resumed_run_counts_only_its_own_tokens(self, tmp_path: Path) -> None:
        """`tokens_seen` is cumulative and `summary.tokens` is not; dividing the former by
        this process's elapsed time credits the hardware with the previous run's work."""
        first = make_trainer(tmp_path, tiny_config(max_steps=2))
        first.train()

        second = make_trainer(tmp_path, tiny_config(max_steps=4))
        state = second.train()
        assert second.summary.steps == 2
        assert 0 < second.summary.tokens < state.tokens_seen
        assert state.tokens_seen == first.summary.tokens + second.summary.tokens

    def test_resume_restores_the_weights_not_just_the_counter(self, tmp_path: Path) -> None:
        first = make_trainer(tmp_path)
        first.train()
        second = make_trainer(tmp_path)
        second.maybe_resume()
        for before, after in zip(first.model.parameters(), second.model.parameters(), strict=True):
            assert torch.allclose(before, after)


class TestRunLock:
    def test_the_lock_is_released_when_the_run_finishes(self, tmp_path: Path) -> None:
        make_trainer(tmp_path).train()
        assert read_lock(tmp_path) is None

    def test_a_second_trainer_cannot_start_on_a_locked_directory(self, tmp_path: Path) -> None:
        """Two containers on one run directory resume from the same step and overwrite each
        other's work, which costs GPU hours and interleaves `metrics.jsonl`."""
        acquire(tmp_path, "someone-else")
        with pytest.raises(RunLockedError, match="someone-else"):
            make_trainer(tmp_path).train()

    def test_force_takes_over_a_held_directory(self, tmp_path: Path) -> None:
        acquire(tmp_path, "a-dead-container")
        config = tiny_config()
        trainer = Trainer(
            config,
            tiny_dataset(config),
            tiny_dataset(config, 1_000),
            tmp_path,
            device=torch.device("cpu"),
            force=True,
        )
        assert trainer.train().step == 4

    def test_the_lock_is_released_even_when_the_run_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise one crash locks the run out until the heartbeat expires."""
        trainer = make_trainer(tmp_path)

        def explode() -> int:
            message = "dataloader died"
            raise RuntimeError(message)

        monkeypatch.setattr(trainer, "_announce", explode)
        with pytest.raises(RuntimeError, match="dataloader died"):
            trainer.train()
        assert read_lock(tmp_path) is None

    def test_the_heartbeat_is_refreshed_at_every_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run longer than `STALE_AFTER` must keep its own lock alive, or a second launch
        decides it is dead and takes the directory over mid-run."""
        refreshed: list[str] = []

        def record(_directory: Path, owner: str) -> None:
            refreshed.append(owner)

        monkeypatch.setattr("agbalu.model.trainer.refresh", record)
        base = tiny_config(max_steps=4)
        config = replace(base, train=replace(base.train, save_every=1))
        trainer = Trainer(
            config,
            tiny_dataset(config),
            tiny_dataset(config, 1_000),
            tmp_path,
            device=torch.device("cpu"),
        )
        trainer.train()
        assert refreshed == [trainer.owner] * 4

    def test_a_resumed_run_reacquires_the_lock_it_released(self, tmp_path: Path) -> None:
        make_trainer(tmp_path, tiny_config(max_steps=2)).train()
        assert read_lock(tmp_path) is None
        assert make_trainer(tmp_path, tiny_config(max_steps=4)).train().step == 4


class TestStepMetrics:
    """A step's reported loss must be the token-weighted mean over its micro-batches, the
    same statistic `validate` returns. The train/eval gap is read as an overfitting
    signal, which requires both sides to be the same measurement."""

    @staticmethod
    def _spy(trainer: Trainer, monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float, int]]:
        seen: list[tuple[float, float, int]] = []
        original = trainer._forward

        def record(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
            loss, accuracy, masked, tokens = original(batch)
            seen.append((float(loss.detach()), float(accuracy), masked))
            return loss, accuracy, masked, tokens

        monkeypatch.setattr(trainer, "_forward", record)
        return seen

    def test_the_reported_loss_is_the_token_weighted_mean_not_the_last_micro_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config(max_steps=3)
        accumulation = config.train.accumulation_steps
        assert accumulation > 1, "a single micro-batch cannot tell the two apart"

        trainer = make_trainer(tmp_path, config)
        seen = self._spy(trainer, monkeypatch)
        trainer.train()

        steps = [e for e in _read_events(tmp_path) if e["event"] == "step"]
        assert len(steps) == 3
        for index, event in enumerate(steps):
            chunk = seen[index * accumulation : (index + 1) * accumulation]
            weighted = sum(loss * m for loss, _, m in chunk) / sum(m for _, _, m in chunk)
            assert event["loss"] == pytest.approx(weighted, abs=1e-5)

    def test_a_step_whose_micro_batches_differ_does_not_report_one_of_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config(max_steps=1)
        trainer = make_trainer(tmp_path, config)
        seen = self._spy(trainer, monkeypatch)
        trainer.train()

        losses = [loss for loss, _, _ in seen]
        reported = next(e for e in _read_events(tmp_path) if e["event"] == "step")["loss"]
        if len(set(losses)) > 1:
            assert min(losses) <= reported <= max(losses)
            assert reported != pytest.approx(losses[-1], abs=1e-9)


class TestNonFiniteRecovery:
    def test_an_isolated_non_finite_batch_does_not_stop_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainer = make_trainer(tmp_path)
        original = trainer._forward
        calls = {"n": 0}

        def sometimes_nan(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
            calls["n"] += 1
            if calls["n"] == 1:
                return torch.tensor(float("nan"), requires_grad=True), torch.zeros(()), 0, 0
            return original(batch)

        monkeypatch.setattr(trainer, "_forward", sometimes_nan)
        assert trainer.train().step == 4

    def test_a_non_finite_micro_batch_resets_the_step_accumulator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`micro` restarts at 0 after a NaN, so whatever was already accumulated belongs
        to a step that will never be reported. Leaving it in inflates the next one."""
        config = tiny_config(max_steps=1)
        assert config.train.accumulation_steps == 2
        trainer = make_trainer(tmp_path, config)
        original = trainer._forward
        calls = {"n": 0}
        finite: list[float] = []

        def one_nan(
            batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
            calls["n"] += 1
            if calls["n"] == 2:
                return torch.tensor(float("nan"), requires_grad=True), torch.zeros(()), 0, 0
            loss, accuracy, masked, tokens = original(batch)
            finite.append(float(accuracy))
            return loss, accuracy, masked, tokens

        monkeypatch.setattr(trainer, "_forward", one_nan)
        trainer.train()

        contributing = finite[1:]
        assert len(contributing) == 2
        step = next(e for e in _read_events(tmp_path) if e["event"] == "step")
        assert step["accuracy"] == pytest.approx(sum(contributing) / 2, abs=1e-6)

    def test_persistent_non_finite_loss_rolls_back_to_the_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The divergence drill. A run of NaNs must restore known-good weights rather
        than continue multiplying them."""
        trainer = make_trainer(tmp_path)
        save(tmp_path, trainer.model, trainer.optimizer, TrainingState(step=1))
        good = [p.detach().clone() for p in trainer.model.parameters()]
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.add_(torch.full_like(parameter, 5.0))

        monkeypatch.setattr(
            trainer,
            "_forward",
            lambda _: (torch.tensor(float("nan"), requires_grad=True), 0.0, 0),
        )
        trainer.state.consecutive_non_finite = MAX_NON_FINITE - 1
        trainer._rollback()

        for restored, expected in zip(trainer.model.parameters(), good, strict=True):
            assert torch.allclose(restored, expected)
        assert trainer.state.consecutive_non_finite == 0

    def test_divergence_before_any_checkpoint_is_a_hard_failure(self, tmp_path: Path) -> None:
        """Nothing to roll back to is worth crashing over — silently continuing from
        destroyed weights would burn the whole budget producing nothing."""
        trainer = make_trainer(tmp_path)
        with pytest.raises(RuntimeError, match="nothing to roll back to"):
            trainer._rollback()


class TestCompilation:
    def test_it_is_off_by_default_so_the_forward_pass_is_the_model_itself(
        self, tmp_path: Path
    ) -> None:
        trainer = make_trainer(tmp_path)
        assert trainer.compiled is trainer.model

    def test_it_is_skipped_off_cuda_however_it_is_configured(self, tmp_path: Path) -> None:
        """The CPU failure drills would otherwise be testing the compiler, and inductor is
        slower than eager for a model this small on CPU."""
        trainer = make_trainer(tmp_path, compile_model=True)
        assert trainer.device.type != "cuda"
        assert trainer.compiled is trainer.model

    def test_checkpoints_are_written_from_the_eager_module(self, tmp_path: Path) -> None:
        """`torch.compile` prefixes every state-dict key with `_orig_mod.`; a checkpoint
        saved through the wrapper cannot be resumed by a run with compilation off."""
        trainer = make_trainer(tmp_path, compile_model=True)
        assert not any(k.startswith("_orig_mod.") for k in trainer.model.state_dict())


class TestDynamoNoise:
    GRAPH_BREAK = "torch._dynamo.variables.tensor"
    RECOMPILES = "torch._dynamo.convert_frame"

    @pytest.fixture(autouse=True)
    def _restore_levels(self) -> Iterator[None]:
        names = (self.GRAPH_BREAK, self.RECOMPILES)
        levels = {name: logging.getLogger(name).level for name in names}
        yield
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)

    def test_the_graph_break_explainer_is_silenced(self) -> None:
        quieten_dynamo()
        assert not logging.getLogger(self.GRAPH_BREAK).isEnabledFor(logging.WARNING)

    def test_a_recompilation_storm_still_reports_itself(self) -> None:
        """The failure this silence must not hide: shape churn that recompiles every step
        would otherwise cost the whole speedup, silently."""
        quieten_dynamo()
        assert logging.getLogger(self.RECOMPILES).level != logging.ERROR
        assert logging.getLogger("torch._dynamo").level != logging.ERROR

    def test_nothing_is_silenced_when_the_model_is_not_compiled(self, tmp_path: Path) -> None:
        logging.getLogger(self.GRAPH_BREAK).setLevel(logging.NOTSET)
        make_trainer(tmp_path, compile_model=True)
        assert logging.getLogger(self.GRAPH_BREAK).level == logging.NOTSET


class TestValidation:
    def test_validation_returns_a_finite_loss(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        assert np.isfinite(trainer.validate(batches=2))

    def test_the_held_out_split_is_masked_at_a_fixed_rate_however_far_the_run_has_gone(
        self, tmp_path: Path
    ) -> None:
        """The eval loss is only comparable across steps because the validation split
        stays at step 0, and so at `mask_p_start`, while training anneals 0.30 -> 0.15.
        Advancing it would make every eval easier than the last and the curve would
        improve on its own."""
        trainer = make_trainer(tmp_path)
        config = trainer.config.train
        for step in (1, 500, config.max_steps):
            trainer.train_data.set_step(step)
            trainer.state.step = step
            trainer.validate(batches=1)
            assert trainer.validation_data.step == 0
        assert config.mask_probability(trainer.validation_data.step) == config.mask_p_start

    def test_an_unscoreable_validation_set_raises_rather_than_returning_infinity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A validation pass that scores no tokens has no loss. Returning infinity instead
        would be accepted as a measurement and poison `best_validation_loss` permanently,
        since nothing can ever beat it."""
        trainer = make_trainer(tmp_path)
        monkeypatch.setattr(
            trainer,
            "_forward",
            lambda _: (torch.tensor(1.0, requires_grad=True), torch.zeros(()), 0, 0),
        )
        with pytest.raises(RuntimeError, match="no scored tokens"):
            trainer.validate(batches=1)

    def test_the_model_is_returned_to_training_mode(self, tmp_path: Path) -> None:
        trainer = make_trainer(tmp_path)
        trainer.model.train()
        trainer.validate(batches=1)
        assert trainer.model.training


class TestCheckpointInteroperability:
    def test_the_trainer_checkpoint_loads_into_a_fresh_model(self, tmp_path: Path) -> None:
        make_trainer(tmp_path).train()
        fresh = make_trainer(tmp_path / "other")
        state = load(tmp_path, fresh.model, fresh.optimizer)
        assert state.step == 4
