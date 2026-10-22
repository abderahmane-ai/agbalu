"""Unit tests for Fadhma 2 training checkpointing, pruning, and best model tracking."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import Wav2Vec2BertConfig

from agbalu.speech.model import ConformerSpeechModel
from agbalu.speech.train import (
    CheckpointBundle,
    save_atomic_checkpoint,
    save_best,
    save_checkpoint,
)


def _make_dummy_model(vocab_size: int = 16) -> ConformerSpeechModel:
    config = Wav2Vec2BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        feat_proj_dropout=0.0,
    )
    config.feature_projection_input_dim = 160
    return ConformerSpeechModel(config, vocab_size=vocab_size)


class TestSpeechTrainingCheckpoints:
    def test_save_atomic_checkpoint(self, tmp_path: Path) -> None:
        target = tmp_path / "checkpoint.pt"
        payload: dict[str, object] = {"step": 10, "loss": 0.5}
        save_atomic_checkpoint(payload, target)

        assert target.is_file()
        loaded = torch.load(target, weights_only=False)
        assert loaded["step"] == 10
        assert loaded["loss"] == 0.5

    def test_save_best_provenance(self, tmp_path: Path) -> None:
        model = _make_dummy_model()
        save_best(tmp_path, model, step=150, dev_cer=7.85, dev_loss=0.312)

        best_dir = tmp_path / "best"
        assert best_dir.is_dir()
        assert (best_dir / "best.json").is_file()
        assert (best_dir / "best.pt").is_file()
        assert (best_dir / "model").is_dir()

        # Root files
        assert (tmp_path / "best.pt").is_file()
        assert (tmp_path / "best_model").is_dir()

        best_meta = json.loads((best_dir / "best.json").read_text(encoding="utf-8"))
        assert best_meta["step"] == 150
        assert best_meta["dev_cer_percent"] == 7.85
        assert best_meta["dev_loss"] == 0.312
        assert "lowest validation CER" in best_meta["selection_criterion"]

    def test_checkpoint_rotation_and_pruning(self, tmp_path: Path) -> None:
        model = _make_dummy_model()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=10)

        # Save 4 checkpoints with keep=2
        for step in (10, 20, 30, 40):
            bundle = CheckpointBundle(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=1,
                best_dev_cer=10.0 - (step * 0.1),
                best_dev_loss=1.0 - (step * 0.01),
            )
            save_checkpoint(tmp_path, bundle, keep=2)

        # Only the two most recent step directories (30 and 40) must remain
        existing = sorted([p.name for p in tmp_path.glob("step_*") if p.is_dir()])
        assert existing == ["step_0000030", "step_0000040"]

        # Each directory must hold model, optimizer, scheduler, rng, and state
        latest_dir = tmp_path / "step_0000040"
        assert (latest_dir / "model").is_dir()
        assert (latest_dir / "optimizer.pt").is_file()
        assert (latest_dir / "scheduler.pt").is_file()
        assert (latest_dir / "rng.pt").is_file()
        assert (latest_dir / "state.json").is_file()

        # latest_checkpoint.pt must point to step 40
        latest_chk = torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)
        assert latest_chk["step"] == 40
        assert "rng" in latest_chk
