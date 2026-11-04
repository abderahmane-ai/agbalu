"""Verify adapter initialization, learning and merge parity."""

import torch
from torch import nn

from agbalu.codec_tts.adaptation import AdaptedLinear, inject_adapter
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.configuration import Geometry


def test_zero_update_and_gradient_asymmetry() -> None:
    torch.manual_seed(9)
    original = nn.Linear(7, 9, bias=False)
    adapter = AdaptedLinear(original)
    inputs = torch.randn((2, 7))
    assert torch.equal(adapter(inputs), original(inputs))
    adapter(inputs).square().sum().backward()
    assert adapter.factor_a.grad is not None
    assert adapter.factor_a.grad.eq(0).all()
    assert adapter.factor_b.grad is not None
    assert adapter.factor_b.grad.abs().sum() > 0
    assert original.weight.grad is None


def test_exact_selected_projection_and_parameter_count() -> None:
    geometry = Geometry(
        hidden_size=16,
        intermediate_size=24,
        layers=1,
        heads=2,
        kv_heads=1,
        head_dim=8,
        text_vocabulary=32,
    )
    generator = Generator(geometry)
    selected = inject_adapter(generator)
    assert len(selected) == 7
    assert (
        sum(p.numel() for p in generator.parameters() if p.requires_grad)
        == geometry.adapter_parameters
    )
    assert Geometry().adapter_parameters == 66060288
    assert not generator.audio_embedding.weight.requires_grad
