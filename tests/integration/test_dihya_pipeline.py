"""Verify file boundaries and deterministic training recovery on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from agbalu.dihya.artifacts import atomic_file
from agbalu.dihya.cli import main
from agbalu.dihya.config import ModelConfig, TrainConfig
from agbalu.dihya.dataset import Example, TextDataset
from agbalu.dihya.train import train

pytestmark = pytest.mark.integration


def test_atomic_file_preserves_existing_contents_after_interruption(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_bytes(b"original")

    def interrupted_write() -> None:
        with atomic_file(target) as stream:
            stream.write(b"partial")
            msg = "interrupted"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted_write()
    assert target.read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data.json"]


def test_missing_training_data_fails_without_synthetic_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main(["train", "--run", "never-created", "--data-dir", str(tmp_path)])


def test_jsonl_rejects_script_contradiction_with_line_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"text": "ⴰⵣⵓⵍ", "label": "shi_Latn", "domain": "web", "source": "test"})
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:1:"):
        TextDataset.load_jsonl(path)


def test_resume_matches_uninterrupted_training_and_writes_first_best(tmp_path: Path) -> None:
    model_config = ModelConfig(
        embed_dim=16,
        conv_dim=8,
        num_layers=1,
        num_heads=2,
        intermediate_dim=32,
        projection_dim=8,
        max_length=32,
        classes=("kab_Latn", "rif_Latn"),
        dropout=0.1,
    )
    config = TrainConfig(
        epochs=2,
        groups_per_batch=2,
        samples_per_group=2,
        eval_batch_size=4,
        memory_bank_size=8,
        min_group_support=1,
    )
    dataset = TextDataset(
        [
            Example(f"Azul fell awen {index}", label, "web", "test")
            for label in model_config.classes
            for index in range(4)
        ]
    )
    dev = TextDataset(
        [
            Example(f"Different phrase {index}", label, "web", "test")
            for index, label in enumerate(model_config.classes)
        ]
    )
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    train(model_config, config, dataset, dev, whole, torch.device("cpu"))
    train(model_config, config, dataset, dev, resumed, torch.device("cpu"), stop_after_steps=1)
    assert (resumed / "best.pt").is_file()
    train(model_config, config, dataset, dev, resumed, torch.device("cpu"))
    expected = torch.load(whole / "final.pt", weights_only=True)
    actual = torch.load(resumed / "final.pt", weights_only=True)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    pointer = json.loads((resumed / "latest.json").read_text())
    snapshot = resumed / pointer["snapshot"]
    with snapshot.open("ab") as stream:
        stream.write(b"corruption-positive-control")
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        train(model_config, config, dataset, dev, resumed, torch.device("cpu"))
