"""Verify signed relative position boundaries."""

import torch

from agbalu.encoder.positions import bucket_indices


def test_exact_and_logarithmic_positions() -> None:
    offsets = torch.tensor([-1023, -17, -16, -15, 0, 15, 16, 17, 1023])
    result = bucket_indices(offsets, 32, 1024)
    assert result.tolist() == [0, 14, 15, 16, 31, 46, 47, 48, 62]
    assert torch.equal(result + result.flip(0), torch.full_like(result, 62))
