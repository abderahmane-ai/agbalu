"""Prove gradient parity checks detect errors above precision noise."""

import pytest
import torch

from agbalu.encoder.configuration import Geometry
from agbalu.encoder.model import Encoder
from agbalu.encoder.smoke import compare_gradient, compare_model_gradients


def test_precision_noise_is_measured_from_a_float32_control() -> None:
    high = torch.tensor([1.0, 2.0, 3.0])
    reference = high + 0.01
    error, noise = compare_gradient(reference, reference + 0.001, high)
    assert 0 < error < noise


def test_deliberately_wrong_gradients_fail_even_with_a_precision_control() -> None:
    reference = torch.ones(8)
    with pytest.raises(ValueError, match="gradient L2 error"):
        compare_gradient(reference, torch.zeros(8), reference + 0.001)
    with pytest.raises(ValueError, match="gradient L2 error"):
        compare_gradient(reference, torch.full((8,), float("nan")), None)


def test_complete_gradient_control_rejects_wrong_bf16_updates() -> None:
    geometry = Geometry(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=8,
        heads=1,
        layers=1,
        block_layers=1,
        backend="reference",
    )
    exact, fast = Encoder(geometry), Encoder(geometry)
    high: dict[str, torch.Tensor] = {}
    for (name, left), right in zip(exact.named_parameters(), fast.parameters(), strict=True):
        left.grad = torch.ones_like(left)
        right.grad = torch.zeros_like(right)
        high[name] = torch.ones_like(left)
    with pytest.raises(ValueError, match="complete gradient error"):
        compare_model_gradients(exact, fast, high, is_bf16=True, per_parameter=False)
