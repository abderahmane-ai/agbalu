"""Verify coordinated masks and protected control tokens."""

import torch

from agbalu.encoder.masking import mask_groups


def test_specials_are_protected_and_groups_are_atomic() -> None:
    ids = torch.tensor([2, 6, 7, 8, 9, 3, 0])
    groups = torch.tensor([-1, 1, 1, 3, 3, -1, -1])
    for seed in range(30):
        masked = mask_groups(ids, groups, 0.25, seed, 32)
        assert torch.equal(masked.labels[1:3] >= 0, (masked.labels[1] >= 0).expand(2))
        assert masked.targets >= 1
        assert masked.labels[[0, 5, 6]].tolist() == [-100, -100, -100]
        assert torch.equal(masked.ids[[0, 5, 6]], ids[[0, 5, 6]])


def test_same_seed_reproduces_masks() -> None:
    ids = torch.arange(5, 25)
    groups = torch.arange(20)
    left = mask_groups(ids, groups, 0.3, 6, 32)
    right = mask_groups(ids, groups, 0.3, 6, 32)
    assert torch.equal(left.ids, right.ids)
    assert torch.equal(left.labels, right.labels)


def test_all_special_batch_has_zero_targets() -> None:
    result = mask_groups(torch.tensor([2, 3]), torch.tensor([-1, -1]), 1, 0, 32)
    assert result.targets.item() == 0
