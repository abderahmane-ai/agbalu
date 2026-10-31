"""Map signed relative distances to logarithmic buckets."""

import math

import torch
from torch import Tensor


def bucket_indices(offsets: Tensor, buckets: int, context_tokens: int) -> Tensor:
    middle = buckets // 2
    distance = offsets.abs().clamp(min=middle, max=context_tokens - 1)
    logarithmic = (
        middle
        + torch.ceil(
            torch.log(distance.float() / middle)
            / math.log((context_tokens - 1) / middle)
            * (middle - 1)
        ).long()
    )
    signed = torch.where(offsets.abs() <= middle, offsets, logarithmic * offsets.sign())
    return signed + buckets - 1
