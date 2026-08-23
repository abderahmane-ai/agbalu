from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from agbalu.lid.losses import (
    GroupWeights,
    SupervisedContrastiveLoss,
    cross_entropy_per_sample,
    ldam_margins,
    logit_adjustment,
)


def unit(rows: int, dim: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return functional.normalize(torch.randn(rows, dim, generator=generator), dim=1)


class TestLdamMargins:
    def test_the_rare_class_gets_the_largest_margin(self) -> None:
        margins = ldam_margins([24_123, 22_698, 447], scale=0.5)
        assert margins[2] > margins[0]
        assert margins[2] == pytest.approx(0.5)

    def test_the_margin_follows_the_fourth_root_of_the_count(self) -> None:
        """`m_k = C / n_k^(1/4)`; a 16x rarer class earns exactly a 2x margin."""
        margins = ldam_margins([16_000, 1_000], scale=1.0)
        assert float(margins[1] / margins[0]) == pytest.approx(2.0, rel=1e-5)

    def test_a_class_with_no_examples_gets_no_margin_rather_than_an_infinite_one(self) -> None:
        margins = ldam_margins([100, 0, 50], scale=0.5)
        assert float(margins[1]) == 0.0
        assert torch.isfinite(margins).all()

    def test_an_all_empty_corpus_yields_zero_margins(self) -> None:
        assert torch.equal(ldam_margins([0, 0], scale=0.5), torch.zeros(2))

    def test_a_uniform_corpus_gives_every_class_the_same_margin(self) -> None:
        margins = ldam_margins([500] * 4, scale=0.35)
        assert torch.allclose(margins, torch.full((4,), 0.35))


class TestLogitAdjustment:
    def test_the_rare_class_is_shifted_furthest(self) -> None:
        shift = logit_adjustment([24_123, 447], tau=0.5)
        assert shift[1] < shift[0] < 0

    def test_tau_zero_disables_the_shift(self) -> None:
        assert torch.equal(logit_adjustment([10, 20], tau=0.0), torch.zeros(2))

    def test_a_zero_count_does_not_produce_negative_infinity(self) -> None:
        assert torch.isfinite(logit_adjustment([10, 0], tau=1.0)).all()

    def test_an_empty_corpus_gives_no_shift(self) -> None:
        assert torch.equal(logit_adjustment([0, 0], tau=1.0), torch.zeros(2))


class TestGroupWeights:
    def test_the_worst_group_gains_weight(self) -> None:
        weights = GroupWeights(4, step=0.05).train()
        before = weights.weights.clone()
        weights(torch.tensor([0.1, 0.1, 3.0, 3.0]), torch.tensor([0, 0, 2, 2]))
        assert weights.weights[2] > before[2]
        assert weights.weights[0] < before[0]

    def test_the_weights_stay_a_distribution(self) -> None:
        weights = GroupWeights(5, step=0.2).train()
        for _ in range(20):
            weights(torch.rand(10) * 5, torch.randint(0, 5, (10,)))
        assert float(weights.weights.sum()) == pytest.approx(1.0, abs=1e-5)
        assert bool((weights.weights > 0).all())

    def test_a_group_absent_from_the_batch_keeps_its_relative_weight(self) -> None:
        """A sampler cannot fit 21 cells into a batch of 12, so absent groups must be
        renormalised, not reset."""
        weights = GroupWeights(4, step=0.1).train()
        before = weights.weights.clone()
        weights(torch.tensor([2.0, 2.0]), torch.tensor([0, 0]))
        assert float(weights.weights[1] / weights.weights[3]) == pytest.approx(
            float(before[1] / before[3]), abs=1e-6
        )

    def test_a_zero_step_reproduces_the_group_mean(self) -> None:
        weights = GroupWeights(2, step=0.0).train()
        losses = torch.tensor([1.0, 3.0, 5.0, 7.0])
        groups = torch.tensor([0, 0, 1, 1])
        assert float(weights(losses, groups)) == pytest.approx(4.0)

    def test_evaluation_does_not_move_the_weights(self) -> None:
        weights = GroupWeights(3, step=0.5).eval()
        before = weights.weights.clone()
        weights(torch.tensor([9.0, 9.0]), torch.tensor([1, 1]))
        assert torch.equal(weights.weights, before)

    def test_the_weights_survive_a_state_dict_round_trip(self) -> None:
        """They are what the loss is taken against; a resumed run without them spends its
        first epoch back where it started."""
        source = GroupWeights(3, step=0.3).train()
        source(torch.tensor([5.0, 0.1]), torch.tensor([0, 1]))
        target = GroupWeights(3, step=0.3)
        target.load_state_dict(source.state_dict())
        assert torch.equal(target.weights, source.weights)

    def test_no_groups_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_groups must be at least 1"):
            GroupWeights(0, step=0.1)

    def test_the_loss_is_differentiable(self) -> None:
        weights = GroupWeights(2, step=0.1).train()
        losses = torch.tensor([1.0, 2.0], requires_grad=True)
        weights(losses, torch.tensor([0, 1])).backward()
        assert losses.grad is not None
        assert bool(torch.isfinite(losses.grad).all())


class TestSupervisedContrastiveLoss:
    def test_a_batch_with_no_positives_returns_zero_rather_than_nan(self) -> None:
        loss = SupervisedContrastiveLoss(4, capacity=0, temperature=0.05)
        value = loss(unit(3, 4), torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]))
        assert float(value) == 0.0

    def test_perfectly_clustered_projections_score_near_zero(self) -> None:
        loss = SupervisedContrastiveLoss(4, capacity=0, temperature=0.05)
        anchors = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        value = loss(anchors, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0]))
        assert float(value) < 0.1

    def test_a_language_split_across_two_domains_scores_worse_than_one_clustered(self) -> None:
        """The whole objective: the same language in two domains must be pulled together."""
        loss = SupervisedContrastiveLoss(2, capacity=0, temperature=0.1)
        labels = torch.tensor([0, 0, 1, 1])
        domains = torch.tensor([0, 1, 0, 1])
        clustered = functional.normalize(
            torch.tensor([[1.0, 0.02], [1.0, -0.02], [-1.0, 0.02], [-1.0, -0.02]]), dim=1
        )
        by_domain = functional.normalize(
            torch.tensor([[1.0, 0.02], [-1.0, 0.02], [1.0, -0.02], [-1.0, -0.02]]), dim=1
        )
        assert float(loss(clustered, labels, domains)) < float(loss(by_domain, labels, domains))

    def test_the_memory_bank_fills_and_wraps(self) -> None:
        loss = SupervisedContrastiveLoss(4, capacity=6, temperature=0.05)
        for seed in range(4):
            loss(unit(4, 4, seed), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
        assert int(loss.bank_fill) == 16
        assert loss.bank_projections.shape == (6, 4)
        assert bool((loss.bank_labels >= 0).all())

    def test_the_bank_supplies_negatives_a_single_language_batch_lacks(self) -> None:
        loss = SupervisedContrastiveLoss(4, capacity=8, temperature=0.05)
        loss(unit(4, 4, 1), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]))
        value = loss(unit(4, 4, 2), torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0, 0]))
        assert float(value) > 0.0

    def test_the_bank_survives_a_state_dict_round_trip(self) -> None:
        source = SupervisedContrastiveLoss(4, capacity=8, temperature=0.05)
        source(unit(4, 4, 3), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
        target = SupervisedContrastiveLoss(4, capacity=8, temperature=0.05)
        target.load_state_dict(source.state_dict())
        assert torch.equal(target.bank_projections, source.bank_projections)
        assert int(target.bank_fill) == int(source.bank_fill)

    def test_an_anchor_is_never_its_own_positive(self) -> None:
        """Self-similarity is 1/tau and would swamp every real pair."""
        loss = SupervisedContrastiveLoss(4, capacity=0, temperature=0.05)
        identical = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        pair = torch.cat([identical, torch.tensor([[0.0, 1.0, 0.0, 0.0]])])
        value = loss(pair, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
        assert torch.isfinite(value)

    def test_a_non_positive_temperature_is_refused(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            SupervisedContrastiveLoss(4, capacity=4, temperature=0.0)

    def test_the_loss_is_differentiable(self) -> None:
        loss = SupervisedContrastiveLoss(4, capacity=4, temperature=0.05)
        projections = unit(4, 4, 5).requires_grad_(True)
        loss(projections, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])).backward()
        assert projections.grad is not None
        assert bool(torch.isfinite(projections.grad).all())

    def test_the_bank_is_not_backpropagated_through(self) -> None:
        loss = SupervisedContrastiveLoss(4, capacity=8, temperature=0.05)
        loss(unit(4, 4, 6), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
        assert not loss.bank_projections.requires_grad


def test_cross_entropy_per_sample_is_unreduced_so_groups_can_weight_it() -> None:
    logits = torch.randn(6, 3)
    labels = torch.randint(0, 3, (6,))
    per_sample = cross_entropy_per_sample(logits, labels, smoothing=0.0)
    assert per_sample.shape == (6,)
    assert float(per_sample.mean()) == pytest.approx(
        float(functional.cross_entropy(logits, labels)), abs=1e-6
    )
