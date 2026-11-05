"""Verify merged projections preserve adapter computation."""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from agbalu.codec_tts.adaptation import AdaptedLinear
from agbalu.codec_tts.release import remote_sources


def test_effective_weight_does_not_mutate_frozen_parameters() -> None:
    torch.manual_seed(1)
    original = nn.Linear(7, 9, bias=False)
    layer = AdaptedLinear(original)
    with torch.no_grad():
        layer.factor_b.normal_(std=0.01)
    before = original.weight.detach().clone()
    inputs = torch.randn((2, 7))
    torch.testing.assert_close(layer(inputs), inputs @ layer.merged_weight().T)
    assert torch.equal(before, original.weight)


@pytest.mark.integration
def test_remote_dependencies_preserve_every_computation() -> None:
    root = Path(__file__).resolve().parents[3]
    exported = remote_sources(root)
    assert "modeling_speech.py" in exported
    assert "synthesis.py" in exported
    for name, content in exported.items():
        original_name = "transformers_model.py" if name == "modeling_speech.py" else name
        path = root / "src/agbalu/codec_tts" / original_name
        if not path.exists():
            path = root / "src/agbalu/llm" / original_name
        original = ast.parse(path.read_text())
        for node in ast.walk(original):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("agbalu.")
            ):
                node.module = node.module.rsplit(".", 1)[-1]
                node.level = 1
        actual = ast.parse(content)
        if name == "modeling_speech.py":
            actual.body = actual.body[:-2]
        assert ast.dump(actual) == ast.dump(original), name
