"""Aggregate residual blocks with per-token attention over depth."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, True) + self.epsilon
        )
        result: Tensor = normalized.to(value.dtype) * self.weight.to(value.dtype)
        return result


class DepthAttention(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width, epsilon)

    def weights(self, values: Tensor) -> Tensor:
        keys = self.norm.forward(values)
        return (keys.float() * self.query.float()).sum(-1).softmax(0)

    def forward(self, blocks: Sequence[Tensor], partial: Tensor | None) -> Tensor:
        values = torch.stack([*blocks, *([] if partial is None else [partial])])
        weights = self.weights(values).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(0)
