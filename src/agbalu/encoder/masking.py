"""Mask whole token groups reproducibly on the selected device."""

from dataclasses import dataclass

import torch
from torch import Tensor

from agbalu.encoder.configuration import SPECIAL_TOKENS

RANDOM_SHARE = 0.1
UNCHANGED_BOUNDARY = 0.2


@dataclass(frozen=True)
class Masked:
    ids: Tensor
    labels: Tensor
    targets: Tensor


def mask_groups(
    ids: Tensor, groups: Tensor, probability: float, seed: int, vocab_size: int
) -> Masked:
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    size = ids.numel()
    group = groups.clamp(min=0)
    eligible = (groups >= 0) & (ids >= SPECIAL_TOKENS)
    selected_groups = torch.rand(size, device=ids.device, generator=generator) < probability
    selected = selected_groups[group] & eligible
    # Ensure a nonempty real batch has supervision without a device-to-host branch.
    first = torch.where(eligible, torch.arange(size, device=ids.device), size).amin()
    first_group = groups[first.clamp(max=size - 1)]
    selected = selected | ((~selected.any()) & eligible & (groups == first_group))
    draw = torch.rand(size, device=ids.device, generator=generator)[group]
    random_ids = torch.randint(
        SPECIAL_TOKENS, vocab_size, (size,), device=ids.device, generator=generator
    )
    replaced = torch.where(selected & (draw < RANDOM_SHARE), random_ids, ids)
    replaced = torch.where(selected & (draw >= UNCHANGED_BOUNDARY), 4, replaced)
    labels = torch.where(selected, ids, -100)
    return Masked(replaced, labels, selected.sum())
