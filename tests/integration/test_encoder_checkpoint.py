"""Verify recovery transactions and interrupted writes."""

from pathlib import Path

import pytest
import torch

from agbalu.encoder import checkpoint
from agbalu.encoder.configuration import Geometry, Recipe
from agbalu.encoder.model import Encoder
from agbalu.encoder.smoke import recovery
from agbalu.encoder.training import optimizer_for

pytestmark = pytest.mark.integration


def test_resumed_optimizer_matches_continuous_updates(tmp_path: Path) -> None:
    report = recovery(tmp_path, torch.device("cpu"))
    assert report["passed"] is True
    assert report["absolute_loss_difference"] == 0


def test_corruption_control_fires(tmp_path: Path) -> None:
    assert recovery(tmp_path, torch.device("cpu"), corrupt=True)["corruption_detected"] is True


def test_partial_write_keeps_the_committed_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = Geometry(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        layers=2,
        block_layers=1,
        heads=2,
        backend="reference",
    )
    model = Encoder(geometry)
    optimizer = optimizer_for(model, Recipe(geometry=geometry), torch.device("cpu"))
    saved = checkpoint.save(tmp_path, model, optimizer, checkpoint.State(step=1), "identity")

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        message = "injected disk-full failure"
        raise OSError(message)

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk-full"):
        checkpoint.save(tmp_path, model, optimizer, checkpoint.State(step=2), "identity")
    assert checkpoint.latest(tmp_path) == saved


def test_corrupt_latest_falls_back_and_changed_identity_is_refused(tmp_path: Path) -> None:
    geometry = Geometry(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        layers=2,
        block_layers=1,
        heads=2,
        backend="reference",
    )
    model = Encoder(geometry)
    optimizer = optimizer_for(model, Recipe(geometry=geometry), torch.device("cpu"))
    older = checkpoint.save(tmp_path, model, optimizer, checkpoint.State(step=1), "identity")
    newer = checkpoint.save(tmp_path, model, optimizer, checkpoint.State(step=2), "identity")
    (newer / "training.pt").write_bytes(b"corrupt")
    assert checkpoint.latest(tmp_path) == older
    with pytest.raises(ValueError, match="identity changed"):
        checkpoint.restore(older, model, optimizer, "other")
