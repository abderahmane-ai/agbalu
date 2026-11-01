"""Unit tests for LDAM margins, ConLID contrastive loss, and Group-DRO."""

from __future__ import annotations

import math

import pytest
import torch

from agbalu.dihya.losses import (
    GroupWeights,
    SupervisedContrastiveLoss,
    ldam_margins,
    logit_adjustment,
)


def test_ldam_margins_computation() -> None:
    """Verify LDAM margins scale with Rademacher complexity bound."""
    counts = [10000, 1000, 100, 0]
    margins = ldam_margins(counts, scale=0.5)

    assert margins.shape == (4,)
    assert margins[3] == 0.0  # Zero support gets zero margin
    assert margins[2] > margins[1] > margins[0]  # Smaller support gets larger margin
    assert torch.isclose(margins.max(), torch.tensor(0.5))


def test_logit_adjustment() -> None:
    """Verify logit adjustment scales correctly with prior distribution."""
    counts = [1000, 100]
    shift = logit_adjustment(counts, tau=0.5)

    assert shift.shape == (2,)
    assert shift[0] > shift[1]  # Higher prior gets larger adjustment subtracted


def test_group_weights_dro_update() -> None:
    """Verify Group-DRO updates weights in the direction of higher group loss."""
    gw = GroupWeights(num_groups=3, step=0.1)
    gw.train()

    initial_weights = gw.weights.clone()
    losses = torch.tensor([1.0, 5.0, 1.0])
    groups = torch.tensor([0, 1, 2])

    reweighted = gw(losses, groups)
    assert reweighted > 0.0

    # Group 1 suffered loss 5.0 vs 1.0, its weight must strictly increase
    assert gw.weights[1] > initial_weights[1]
    assert gw.weights[0] < initial_weights[0]
    assert torch.isclose(gw.weights.sum(), torch.tensor(1.0))


def test_conlid_contrastive_loss() -> None:
    """Verify ConLID supervised contrastive loss over projections."""
    dim = 16
    con_fn = SupervisedContrastiveLoss(dim=dim, capacity=32, temperature=0.1)

    projections = torch.nn.functional.normalize(torch.randn(4, dim), p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 0, 0, 0])

    loss = con_fn(projections, labels, domains)
    assert loss >= 0.0
    assert con_fn._filled() == 4


def test_group_dro_matches_exponentiated_update() -> None:
    objective = GroupWeights(2, 0.1)
    losses = torch.tensor([1.0, 1.0, 3.0], requires_grad=True)
    result = objective(losses, torch.tensor([0, 0, 1]))
    expected = torch.softmax(torch.tensor([0.1, 0.3]), 0)
    assert torch.allclose(objective.weights, expected)
    assert result.item() == pytest.approx(float(expected @ torch.tensor([1.0, 3.0])))
    torch.autograd.backward(result)
    assert losses.grad is not None
    assert losses.grad.tolist() == pytest.approx(
        [float(expected[0] / 2)] * 2 + [float(expected[1])]
    )


def test_subset_dro_uses_convex_normalization() -> None:
    objective = GroupWeights(2, 0.0).eval()
    objective.log_weights.copy_(torch.tensor([0.8, 0.2]).log())
    assert objective(torch.tensor([2.0]), torch.tensor([0])).item() == pytest.approx(2.0)


def test_large_group_loss_does_not_overflow() -> None:
    objective = GroupWeights(2, 1.0)
    value = objective(torch.tensor([1e6, 1.0]), torch.tensor([0, 1]))
    assert torch.isfinite(value)
    assert objective.weights.tolist() == [1.0, 0.0]


def test_contrastive_bank_fills_without_usable_pairs_and_has_zero_gradient() -> None:
    objective = SupervisedContrastiveLoss(3, 8, 0.1)
    projections = torch.eye(3, requires_grad=True)
    result = objective(projections, torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long))
    torch.autograd.backward(result)
    assert result.item() == 0
    assert objective._filled() == 3
    assert projections.grad is not None
    assert torch.equal(projections.grad, torch.zeros(3, 3))


def test_contrastive_loss_matches_uniform_pool_and_eval_is_read_only() -> None:
    objective = SupervisedContrastiveLoss(2, 8, 0.1).eval()
    projections = torch.tensor([[1.0, 0.0]] * 4, requires_grad=True)
    result = objective(projections, torch.tensor([0, 0, 1, 1]), torch.zeros(4, dtype=torch.long))
    assert result.item() == pytest.approx(math.log(3), abs=1e-6)
    torch.autograd.backward(result)
    assert projections.grad is not None
    assert torch.isfinite(projections.grad).all()
    assert objective._filled() == 0


def test_partial_usable_anchors_and_wrapped_bank_have_finite_gradients() -> None:
    objective = SupervisedContrastiveLoss(3, 3, 0.1)
    for _ in range(3):
        projections = torch.nn.functional.normalize(torch.randn(3, 3), dim=1).requires_grad_()
        result = objective(projections, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]))
        torch.autograd.backward(result)
        assert projections.grad is not None
        assert torch.isfinite(projections.grad).all()
