"""Verify numeric update normalization and fixed schedule behavior."""

import torch

from agbalu.codec_tts.adaptation import adapter_state, inject_adapter
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.configuration import Geometry, Recipe
from agbalu.codec_tts.dataset import collate_cpu
from agbalu.codec_tts.layout import SPECIALS, prompt_layout, target_layout
from agbalu.codec_tts.training import perform_update, scheduled_rate


def test_complete_update_is_independent_of_microbatch_partition() -> None:
    geometry = Geometry(
        hidden_size=16,
        intermediate_size=24,
        layers=1,
        heads=2,
        kv_heads=1,
        head_dim=8,
        text_vocabulary=32,
    )
    tokens = dict(zip(SPECIALS, range(1, 6), strict=True))
    layouts = [
        target_layout(prompt_layout(tokens, [7]), torch.full((frames, 8), 3)) for frames in (2, 4)
    ]
    results: list[dict[str, torch.Tensor]] = []
    for grouped in (True, False):
        torch.manual_seed(9)
        model = Generator(geometry)
        inject_adapter(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        batches = (
            [collate_cpu(layouts)] if grouped else [collate_cpu([layout]) for layout in layouts]
        )
        metrics = perform_update(model, optimizer, batches, Recipe())
        assert metrics["supervised_tokens"] == 64
        results.append(adapter_state(model))
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], atol=1e-6, rtol=1e-4)


def test_schedule_uses_planned_updates_and_final_floor() -> None:
    recipe = Recipe()
    assert scheduled_rate(0, 1000, recipe) == recipe.peak_lr / 20
    assert scheduled_rate(19, 1000, recipe) == recipe.peak_lr
    assert scheduled_rate(999, 1000, recipe) == recipe.minimum_lr
