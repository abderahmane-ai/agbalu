"""The large-vocabulary training loss (task 11.6).

What these pin is the definition — the causal shift, the softcap, the ignored positions —
because the fused kernel is checked against `reference_loss` on the GPU and a reference that
is wrong makes that check pass for the wrong reason.
"""

from __future__ import annotations

import pytest
import torch

from agbalu.llm.loss import IGNORE_INDEX, LossError, logit_softcap, reference_loss, sequence_loss


def inputs(
    rows: int = 2, length: int = 6, width: int = 8, vocabulary: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(rows, length, width, generator=generator)
    classifier = torch.randn(vocabulary, width, generator=generator)
    targets = torch.randint(0, vocabulary, (rows, length), generator=generator)
    return hidden, classifier, targets


class TestShift:
    def test_position_t_predicts_position_t_plus_one(self) -> None:
        """The loss must depend on every target but the first, and on no other."""
        hidden, classifier, targets = inputs()
        before = reference_loss(hidden, classifier, targets)

        moved = targets.clone()
        moved[0, 0] = (moved[0, 0] + 1) % classifier.shape[0]
        assert torch.equal(reference_loss(hidden, classifier, moved), before), (
            "the first target is not predicted by anything and must not enter the loss"
        )

        moved = targets.clone()
        moved[0, -1] = (moved[0, -1] + 1) % classifier.shape[0]
        assert not torch.equal(reference_loss(hidden, classifier, moved), before)

    def test_the_last_hidden_state_predicts_nothing(self) -> None:
        hidden, classifier, targets = inputs()
        before = reference_loss(hidden, classifier, targets)
        hidden[:, -1, :] += 10.0
        assert torch.equal(reference_loss(hidden, classifier, targets), before)


class TestSoftcap:
    def test_it_is_applied_and_changes_the_answer(self) -> None:
        hidden, classifier, targets = inputs()
        assert not torch.equal(
            reference_loss(hidden, classifier, targets, softcap=30.0),
            reference_loss(hidden, classifier, targets),
        )

    def test_a_large_cap_converges_on_no_cap(self) -> None:
        """`softcap * tanh(x / softcap) -> x`, so the two must agree in the limit."""
        hidden, classifier, targets = inputs()
        capped = reference_loss(hidden, classifier, targets, softcap=1e6)
        assert torch.allclose(capped, reference_loss(hidden, classifier, targets), atol=1e-5)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_non_positive_cap_is_refused(self, value: float) -> None:
        hidden, classifier, targets = inputs()
        with pytest.raises(LossError, match="softcap must be positive"):
            reference_loss(hidden, classifier, targets, softcap=value)


class TestIgnoredPositions:
    def test_ignored_targets_leave_the_loss_alone(self) -> None:
        hidden, classifier, targets = inputs()
        targets[1, 3:] = IGNORE_INDEX
        kept = reference_loss(hidden[:1], classifier, targets[:1])
        both = reference_loss(hidden, classifier, targets)
        assert torch.isfinite(both)
        assert not torch.equal(both, kept)

    def test_every_target_ignored_is_not_silently_zero(self) -> None:
        """`cross_entropy` returns nan for an empty selection, and nan must reach the caller
        rather than be floored into a number a run would train on."""
        hidden, classifier, targets = inputs()
        targets[:] = IGNORE_INDEX
        assert torch.isnan(reference_loss(hidden, classifier, targets))


class TestShapes:
    def test_a_target_grid_of_the_wrong_shape_is_refused(self) -> None:
        hidden, classifier, targets = inputs()
        with pytest.raises(LossError, match="does not match targets"):
            reference_loss(hidden, classifier, targets[:, :-1])

    def test_a_classifier_of_the_wrong_width_is_refused(self) -> None:
        hidden, classifier, targets = inputs()
        with pytest.raises(LossError, match="classifier width"):
            reference_loss(hidden, classifier[:, :-1], targets)


class TestDispatch:
    def test_off_cuda_it_matches_the_reference(self) -> None:
        hidden, classifier, targets = inputs()
        assert torch.equal(
            sequence_loss(hidden, classifier, targets, softcap=30.0),
            reference_loss(hidden, classifier, targets, softcap=30.0),
        )


class TestSoftcapFromConfig:
    def test_a_config_declaring_one_is_read(self) -> None:
        assert logit_softcap(type("C", (), {"final_logit_softcapping": 30.0})()) == 30.0

    def test_qwen_carries_none(self) -> None:
        assert logit_softcap(type("C", (), {"final_logit_softcapping": None})()) is None

    def test_a_config_without_the_field_is_uncapped(self) -> None:
        assert logit_softcap(object()) is None
