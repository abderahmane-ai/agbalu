"""Check factorized likelihood and accumulation denominators against hand calculations."""

import math

import pytest
import torch

from agbalu.restoration.dataset import Batch, Example, collate
from agbalu.restoration.model import Scores
from agbalu.restoration.objective import loss_sums, target_counts


def test_uniform_logits_match_joint_likelihood(example: Example) -> None:
    batch = collate([example], 64)
    scores = Scores(
        torch.zeros_like(batch.marks, dtype=torch.float32),
        torch.zeros((*batch.marks.shape, 8)),
        torch.zeros((*batch.casing.shape, 3)),
        torch.zeros((*batch.structure.shape, 3)),
    )
    losses = loss_sums(scores, batch)
    assert float(losses.punctuation) == pytest.approx(4 * math.log(2) + 2 * math.log(8))
    assert float(losses.casing) == pytest.approx(3 * math.log(3))
    assert target_counts(batch).tolist() == [4, 3, 4]


def test_no_positive_marks_and_all_masked_auxiliaries_remain_finite(example: Example) -> None:
    batch = collate([example], 64)
    batch.marks.fill_(0)
    batch.casing.fill_(-100)
    batch.structure.fill_(-100)
    scores = Scores(
        torch.zeros_like(batch.marks, dtype=torch.float32),
        torch.zeros((*batch.marks.shape, 8)),
        torch.zeros((*batch.casing.shape, 3)),
        torch.zeros((*batch.structure.shape, 3)),
    )
    losses = loss_sums(scores, batch)
    assert float(losses.casing) == 0
    assert float(losses.structure) == 0
    assert bool(torch.isfinite(losses.normalized(target_counts(batch))))


def uniform_scores(batch: Batch) -> Scores:
    return Scores(
        torch.zeros_like(batch.marks, dtype=torch.float32),
        torch.zeros((*batch.marks.shape, 8)),
        torch.zeros((*batch.casing.shape, 3)),
        torch.zeros((*batch.structure.shape, 3)),
    )


def test_balanced_loss_has_explicit_class_denominators(example: Example) -> None:
    batch = collate([example], 64)
    counts = target_counts(batch, balanced=True, structure_enabled=False)
    losses = loss_sums(
        uniform_scores(batch), batch, denominators=counts, balanced=True, structure_enabled=False
    )
    assert float(losses.punctuation) == pytest.approx(math.log(2) + math.log(8))
    assert float(losses.casing) == pytest.approx(math.log(3))
    assert float(losses.structure) == 0
    assert losses.contributions is not None
    assert float(losses.contributions[:2].sum()) == pytest.approx(math.log(2))
    assert float(losses.contributions[2:10].sum()) == pytest.approx(math.log(8))


def test_balanced_loss_with_active_structure(example: Example) -> None:
    batch = collate([example], 64)
    counts = target_counts(batch, balanced=True)
    losses = loss_sums(uniform_scores(batch), batch, denominators=counts, balanced=True)
    assert float(losses.punctuation) == pytest.approx(math.log(2) + math.log(8))
    assert float(losses.casing) == pytest.approx(math.log(3))
    assert float(losses.structure) == pytest.approx(math.log(3))
    assert losses.contributions is not None
    assert float(losses.contributions[-1]) == pytest.approx(math.log(3))


def test_adding_none_targets_cannot_dilute_positive_type_loss(example: Example) -> None:
    batch = collate([example], 64)
    before = loss_sums(
        uniform_scores(batch),
        batch,
        denominators=target_counts(batch, balanced=True),
        balanced=True,
    )
    batch.marks[0, 4:] = 0
    after = loss_sums(
        uniform_scores(batch),
        batch,
        denominators=target_counts(batch, balanced=True),
        balanced=True,
    )
    assert before.contributions is not None
    assert after.contributions is not None
    torch.testing.assert_close(before.contributions[:10], after.contributions[:10])
    ordinary = loss_sums(uniform_scores(batch), batch)
    assert float(ordinary.punctuation / target_counts(batch)[0]) < float(before.punctuation)


def test_balanced_accumulation_uses_whole_update_counts(example: Example) -> None:
    first, second = collate([example], 64), collate([example], 64)
    second.marks.fill_(0)
    second.casing.fill_(0)
    counts = target_counts(first, balanced=True) + target_counts(second, balanced=True)
    parts = [
        loss_sums(uniform_scores(batch), batch, denominators=counts, balanced=True).normalized(
            counts
        )
        for batch in (first, second)
    ]
    combined = collate([example, example], 64)
    combined.marks[1] = second.marks[0]
    combined.casing[1] = second.casing[0]
    expected = loss_sums(
        uniform_scores(combined), combined, denominators=counts, balanced=True
    ).normalized(counts)
    torch.testing.assert_close(sum(parts), expected)
