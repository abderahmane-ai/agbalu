"""Recover identical optimizer updates and all random streams."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from agbalu.codec_tts.adaptation import adapter_state, inject_adapter
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.checkpoint import RunState, latest_snapshot, restore_snapshot, save_snapshot
from agbalu.codec_tts.configuration import Geometry, Recipe
from agbalu.codec_tts.dataset import collate_cpu
from agbalu.codec_tts.layout import SPECIALS, prompt_layout, target_layout
from agbalu.codec_tts.training import perform_update

pytestmark = pytest.mark.integration


def test_interrupted_update_replays_optimizer_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(10)
    model = Generator(
        Geometry(
            hidden_size=16,
            intermediate_size=24,
            layers=1,
            heads=2,
            kv_heads=1,
            head_dim=8,
            text_vocabulary=32,
        )
    )
    inject_adapter(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    prompt = prompt_layout(dict(zip(SPECIALS, range(1, 6), strict=True)), [7])
    batch = collate_cpu([target_layout(prompt, torch.ones((3, 8), dtype=torch.long))])
    perform_update(model, optimizer, [batch], Recipe())
    state = RunState(step=1, cursor=1, targets=32, examples=1)
    saved = save_snapshot(tmp_path, model, optimizer, state, "fixture")
    expected_random = (torch.rand(3), random.random(), float(np.random.random()))
    perform_update(model, optimizer, [batch], Recipe())
    expected = adapter_state(model)
    restored = restore_snapshot(saved, model, optimizer, "fixture")
    assert restored == state
    actual_random = (torch.rand(3), random.random(), float(np.random.random()))
    assert torch.equal(actual_random[0], expected_random[0])
    assert actual_random[1:] == expected_random[1:]
    perform_update(model, optimizer, [batch], Recipe())
    for name, value in adapter_state(model).items():
        assert torch.equal(value, expected[name])
    (saved / "training.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="no valid recovery"):
        latest_snapshot(tmp_path, "fixture")
