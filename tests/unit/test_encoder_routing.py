"""Verify per-token depth aggregation and its gradients."""

import torch

from agbalu.encoder.routing import DepthAttention, RMSNorm


def test_uniform_initialization_and_partial_block() -> None:
    router = DepthAttention(4, 1e-6)
    first = torch.ones(2, 4)
    second = torch.ones(2, 4) * 3
    assert torch.equal(router.forward([first], None), first)
    assert torch.equal(router.forward([first], second), (first + second) / 2)


def test_router_attends_per_token_and_backpropagates() -> None:
    router = DepthAttention(4, 1e-6)
    values = torch.tensor(
        [[[1.0, 0, 0, 0], [0, 1.0, 0, 0]], [[0, 1.0, 0, 0], [1.0, 0, 0, 0]]], requires_grad=True
    )
    with torch.no_grad():
        router.query[0] = 1
    weights = router.weights(values)
    assert weights[0, 0] > weights[0, 1]
    torch.autograd.backward(router.forward([values[0]], values[1]).square().sum())
    assert router.query.grad is not None
    assert router.query.grad.abs().sum() > 0
    assert values.grad is not None


def test_norm_reduces_in_float32() -> None:
    normalized = RMSNorm(4, 1e-6).forward(torch.ones(2, 4, dtype=torch.bfloat16))
    assert normalized.dtype == torch.bfloat16
    assert torch.equal(normalized, torch.ones_like(normalized))
