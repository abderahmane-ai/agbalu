"""Verify adapter initialization, learning and merge parity."""

import pytest
import torch
from torch import nn

from agbalu.codec_tts.adaptation import AdaptedLinear, inject_adapter, scaled_adapter
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


def test_scaling_is_reversible_and_does_not_mutate_weights() -> None:
    geometry = Geometry(
        hidden_size=8,
        intermediate_size=12,
        layers=1,
        heads=1,
        kv_heads=1,
        head_dim=8,
        text_vocabulary=16,
    )
    model = Generator(geometry)
    inject_adapter(model)
    layer = model.get_submodule("body.layers.0.self_attn.q_proj")
    assert isinstance(layer, AdaptedLinear)
    with torch.no_grad():
        layer.factor_b.fill_(0.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    inputs = torch.randn(2, 8)
    baseline = layer(inputs)
    with scaled_adapter(model, 0):
        assert torch.equal(layer(inputs), layer.original(inputs))
        with scaled_adapter(model, 0.5):
            torch.testing.assert_close(layer(inputs), (baseline + layer.original(inputs)) / 2)
        assert layer.inference_scale == 0
    assert torch.equal(layer(inputs), baseline)
    message = "interrupted"
    with pytest.raises(RuntimeError, match="interrupted"), scaled_adapter(model, 0.65):
        raise RuntimeError(message)
    assert layer.inference_scale == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_adapter_strength_fails(value: float) -> None:
    with (
        pytest.raises(ValueError, match="finite and nonnegative"),
        scaled_adapter(
            Generator(
                Geometry(
                    hidden_size=8,
                    intermediate_size=12,
                    layers=1,
                    heads=1,
                    kv_heads=1,
                    head_dim=8,
                    text_vocabulary=16,
                )
            ),
            value,
        ),
    ):
        pass
