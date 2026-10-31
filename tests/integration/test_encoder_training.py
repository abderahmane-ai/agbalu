"""Verify full trainer continuation on real artifact-shaped batches."""

from pathlib import Path

import numpy as np
import pytest
import torch

from agbalu.encoder.configuration import Geometry, Locations, Recipe
from agbalu.encoder.corpus import write_json
from agbalu.encoder.training import Trainer
from agbalu.llm.provenance import file_digest

pytestmark = pytest.mark.integration


def test_stopped_trainer_reproduces_continuous_training(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.tile(np.arange(5, 37, dtype=np.uint16), 20)
    (data / "tokens.bin").write_bytes(tokens.tobytes())
    (data / "groups.bin").write_bytes(np.arange(640, dtype=np.int32).tobytes())
    table = np.asarray([(i * 32, (i + 1) * 32, int(i % 5 == 0)) for i in range(20)])
    np.save(data / "documents.npy", table, allow_pickle=False)
    write_json(
        data / "corpus.stats.json",
        {
            "text_tokens": 640,
            "train_text_tokens": 512,
            "validation_text_tokens": 128,
            "files": {
                name: file_digest(data / name)
                for name in ("tokens.bin", "groups.bin", "documents.npy")
            },
        },
    )
    geometry = Geometry(
        hidden_size=16,
        intermediate_size=32,
        heads=2,
        layers=2,
        block_layers=1,
        backend="reference",
        dropout=0.1,
    )
    recipe = Recipe(
        geometry=geometry,
        total_tokens=3072,
        micro_tokens=1024,
        batch_tokens=1024,
        require_smoke=False,
        task_evaluation=False,
        workers=0,
        compile=False,
        validation_documents=4,
    )
    locations = Locations(data=data, runs=tmp_path / "runs")
    full = Trainer(recipe, locations, "continuous", torch.device("cpu"))
    full_report = full.run(install_signals=False)
    split = Trainer(recipe, locations, "resumed", torch.device("cpu"))
    split.run(stop_after=1, install_signals=False)
    resumed = Trainer(recipe, locations, "resumed", torch.device("cpu"))
    resumed_report = resumed.run(install_signals=False)
    assert full_report["tokens_total"] == resumed_report["tokens_total"] == 3072
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], atol=0, rtol=0)
    noop = Trainer(recipe, locations, "resumed", torch.device("cpu")).run(install_signals=False)
    assert noop["tokens_this_run"] == 0
    assert noop["steps_this_run"] == 0
    assert noop["tokens_per_second"] is None
