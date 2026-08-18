"""Failure drills for checkpointing (CLAUDE.md §3.7).

Each test here corresponds to one row of the table in `agbalu.model.checkpoint`'s module
docstring. The point is not that the happy path works — it is that every named way a run
dies leaves something resumable behind, and that a checkpoint which cannot be trusted is
refused rather than loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from agbalu.model.checkpoint import (
    BEST,
    LATEST,
    MANIFEST,
    TrainingState,
    load,
    resume_point,
    save,
    sha256,
)
from agbalu.model.config import ModelError
from agbalu.model.optim import Lamb


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))


def make_optimizer(model: nn.Module) -> Lamb:
    return Lamb(model.parameters(), lr=1e-3, weight_decay=0.1)


def take_a_step(model: nn.Module, optimizer: Lamb) -> None:
    loss = model(torch.randn(4, 8)).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def written(directory: Path, step: int = 7) -> TrainingState:
    model, state = make_model(), TrainingState(step=step, epoch=1, batch_in_epoch=3)
    optimizer = make_optimizer(model)
    take_a_step(model, optimizer)
    save(directory, model, optimizer, state)
    return state


class TestAtomicity:
    def test_no_staging_file_survives_a_successful_write(self, tmp_path: Path) -> None:
        written(tmp_path)
        assert list(tmp_path.glob("*.partial")) == []

    def test_a_failed_write_leaves_the_previous_checkpoint_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mid-write kill. `torch.save` dying must not consume the resumable file."""
        written(tmp_path, step=5)
        good = (tmp_path / LATEST).read_bytes()

        def explode(*_: object, **__: object) -> None:
            message = "simulated kill during write"
            raise OSError(message)

        monkeypatch.setattr(torch, "save", explode)
        model = make_model()
        with pytest.raises(OSError, match="simulated kill"):
            save(tmp_path, model, make_optimizer(model), TrainingState(step=6))

        assert (tmp_path / LATEST).read_bytes() == good
        assert load(tmp_path, make_model(), name=LATEST).step == 5

    def test_the_recorded_checksum_matches_the_file(self, tmp_path: Path) -> None:
        written(tmp_path)
        recorded = (tmp_path / f"{LATEST}.sha256").read_text(encoding="utf-8").strip()
        assert recorded == sha256(tmp_path / LATEST)


class TestCorruption:
    def test_a_flipped_byte_is_refused(self, tmp_path: Path) -> None:
        written(tmp_path)
        target = tmp_path / LATEST
        payload = bytearray(target.read_bytes())
        payload[len(payload) // 2] ^= 0xFF
        target.write_bytes(payload)

        with pytest.raises(ModelError, match="is corrupt"):
            load(tmp_path, make_model())

    def test_a_truncated_checkpoint_is_refused(self, tmp_path: Path) -> None:
        written(tmp_path)
        target = tmp_path / LATEST
        target.write_bytes(target.read_bytes()[: -1 << 10])
        with pytest.raises(ModelError, match="is corrupt"):
            load(tmp_path, make_model())

    def test_resume_falls_back_to_best_when_latest_is_corrupt(self, tmp_path: Path) -> None:
        """A corrupt rolling checkpoint costs progress, not the job."""
        model = make_model()
        optimizer = make_optimizer(model)
        take_a_step(model, optimizer)
        save(tmp_path, model, optimizer, TrainingState(step=10), name=BEST)
        save(tmp_path, model, optimizer, TrainingState(step=20), name=LATEST)

        (tmp_path / LATEST).write_bytes(b"shredded")
        chosen = resume_point(tmp_path)
        assert chosen is not None
        assert chosen.name == BEST
        assert load(tmp_path, make_model(), name=BEST).step == 10

    def test_no_checkpoint_at_all_means_start_fresh(self, tmp_path: Path) -> None:
        assert resume_point(tmp_path) is None


class TestPreference:
    """Which checkpoint a continuation starts from is the operator's choice, so the
    default must not quietly win."""

    def _both(self, directory: Path) -> None:
        model = make_model()
        optimizer = make_optimizer(model)
        take_a_step(model, optimizer)
        save(directory, model, optimizer, TrainingState(step=10), name=BEST)
        save(directory, model, optimizer, TrainingState(step=20), name=LATEST)

    def test_best_is_chosen_when_asked_for_though_latest_is_newer(self, tmp_path: Path) -> None:
        self._both(tmp_path)
        chosen = resume_point(tmp_path, prefer="best")
        assert chosen is not None
        assert chosen.name == BEST

    def test_latest_remains_the_default(self, tmp_path: Path) -> None:
        self._both(tmp_path)
        chosen = resume_point(tmp_path)
        assert chosen is not None
        assert chosen.name == LATEST

    def test_asking_for_best_before_one_exists_falls_back_to_latest(self, tmp_path: Path) -> None:
        written(tmp_path, step=3)
        chosen = resume_point(tmp_path, prefer="best")
        assert chosen is not None
        assert chosen.name == LATEST

    def test_a_corrupt_best_falls_back_rather_than_failing_the_run(self, tmp_path: Path) -> None:
        self._both(tmp_path)
        (tmp_path / BEST).write_bytes(b"shredded")
        chosen = resume_point(tmp_path, prefer="best")
        assert chosen is not None
        assert chosen.name == LATEST

    def test_neither_being_usable_still_starts_fresh(self, tmp_path: Path) -> None:
        self._both(tmp_path)
        (tmp_path / BEST).write_bytes(b"shredded")
        (tmp_path / LATEST).write_bytes(b"shredded")
        assert resume_point(tmp_path, prefer="best") is None

    def test_a_payload_that_is_not_a_checkpoint_is_named_as_such(self, tmp_path: Path) -> None:
        torch.save({"model": {}}, tmp_path / LATEST)
        with pytest.raises(ModelError, match="missing `optimizer`"):
            load(tmp_path, make_model())

    def test_a_missing_checkpoint_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError, match="no checkpoint"):
            load(tmp_path, make_model())


class TestDiskFull:
    def test_writing_is_refused_when_the_disk_cannot_hold_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written(tmp_path)
        monkeypatch.setattr("agbalu.model.checkpoint._free_bytes", lambda _: 1)
        model = make_model()
        with pytest.raises(ModelError, match="refusing to checkpoint"):
            save(tmp_path, model, make_optimizer(model), TrainingState(step=8))

    def test_the_refusal_does_not_damage_the_existing_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written(tmp_path, step=4)
        monkeypatch.setattr("agbalu.model.checkpoint._free_bytes", lambda _: 1)
        model = make_model()
        with pytest.raises(ModelError):
            save(tmp_path, model, make_optimizer(model), TrainingState(step=9))
        assert load(tmp_path, make_model()).step == 4


class TestResume:
    def test_weights_survive_the_round_trip(self, tmp_path: Path) -> None:
        model = make_model(seed=1)
        optimizer = make_optimizer(model)
        take_a_step(model, optimizer)
        save(tmp_path, model, optimizer, TrainingState(step=3))

        restored = make_model(seed=99)
        load(tmp_path, restored)
        for before, after in zip(model.parameters(), restored.parameters(), strict=True):
            assert torch.equal(before, after)

    def test_optimizer_moments_survive_the_round_trip(self, tmp_path: Path) -> None:
        """Resuming without the moments silently restarts LAMB's adaptation."""
        model = make_model()
        optimizer = make_optimizer(model)
        take_a_step(model, optimizer)
        take_a_step(model, optimizer)
        save(tmp_path, model, optimizer, TrainingState(step=2))

        restored = make_model()
        restored_optimizer = make_optimizer(restored)
        load(tmp_path, restored, restored_optimizer)
        original = next(iter(optimizer.state.values()))
        recovered = next(iter(restored_optimizer.state.values()))
        assert original["step"] == recovered["step"] == 2
        assert torch.equal(original["exp_avg"], recovered["exp_avg"])
        assert torch.equal(original["exp_avg_sq"], recovered["exp_avg_sq"])

    def test_the_dataloader_position_survives(self, tmp_path: Path) -> None:
        model = make_model()
        optimizer = make_optimizer(model)
        save(tmp_path, model, optimizer, TrainingState(step=11, epoch=4, batch_in_epoch=137))
        state = load(tmp_path, make_model())
        assert (state.step, state.epoch, state.batch_in_epoch) == (11, 4, 137)

    def test_rng_restoration_makes_the_next_draw_identical(self, tmp_path: Path) -> None:
        model = make_model()
        optimizer = make_optimizer(model)
        save(tmp_path, model, optimizer, TrainingState(step=1))
        expected = torch.randn(4)

        load(tmp_path, make_model(), restore_rng=True)
        assert torch.equal(torch.randn(4), expected)

    def test_rng_restoration_can_be_declined(self, tmp_path: Path) -> None:
        model = make_model()
        save(tmp_path, model, make_optimizer(model), TrainingState(step=1))
        # Built before the seed is set: `make_model` seeds internally, so constructing it
        # after would itself move the state this test is guarding.
        target = make_model()
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        load(tmp_path, target, restore_rng=False)
        assert torch.equal(torch.get_rng_state(), before)

    def test_history_and_best_loss_survive(self, tmp_path: Path) -> None:
        model = make_model()
        state = TrainingState(step=6, best_validation_loss=2.5)
        state.history.append({"step": 5.0, "validation_loss": 2.5})
        save(tmp_path, model, make_optimizer(model), state)
        restored = load(tmp_path, make_model())
        assert restored.best_validation_loss == 2.5
        assert restored.history == [{"step": 5.0, "validation_loss": 2.5}]

    def test_unknown_fields_in_an_older_state_are_ignored(self) -> None:
        payload: dict[str, Any] = {"step": 3, "retired_field": "gone"}
        assert TrainingState.from_dict(payload).step == 3


class TestManifest:
    def test_every_write_appends_one_row(self, tmp_path: Path) -> None:
        model = make_model()
        optimizer = make_optimizer(model)
        for step in (1, 2, 3):
            save(tmp_path, model, optimizer, TrainingState(step=step))
        rows = [
            json.loads(line)
            for line in (tmp_path / MANIFEST).read_text(encoding="utf-8").splitlines()
        ]
        assert [row["step"] for row in rows] == [1, 2, 3]
        assert all(row["sha256"] and row["bytes"] > 0 for row in rows)

    def test_best_and_latest_are_recorded_separately(self, tmp_path: Path) -> None:
        model = make_model()
        optimizer = make_optimizer(model)
        save(tmp_path, model, optimizer, TrainingState(step=1), name=LATEST)
        save(tmp_path, model, optimizer, TrainingState(step=1), name=BEST)
        names = {
            json.loads(line)["name"]
            for line in (tmp_path / MANIFEST).read_text(encoding="utf-8").splitlines()
        }
        assert names == {LATEST, BEST}
