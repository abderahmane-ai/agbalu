"""Verify merged projections preserve adapter computation."""

import torch
from torch import nn

from agbalu.codec_tts.adaptation import AdaptedLinear


def test_effective_weight_does_not_mutate_frozen_parameters() -> None:
    torch.manual_seed(1)
    original = nn.Linear(7, 9, bias=False)
    layer = AdaptedLinear(original)
    with torch.no_grad():
        layer.factor_b.normal_(std=0.01)
    before = original.weight.detach().clone()
    inputs = torch.randn((2, 7))
    torch.testing.assert_close(layer(inputs), inputs @ layer.merged_weight().T)
    assert torch.equal(before, original.weight)
