"""Sum masked multitask losses with explicit supervision denominators."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from agbalu.restoration.annotations import IGNORE
from agbalu.restoration.dataset import Batch
from agbalu.restoration.model import Scores


@dataclass(frozen=True)
class Losses:
    punctuation: Tensor
    casing: Tensor
    structure: Tensor
    is_normalized: bool = False
    contributions: Tensor | None = None

    def normalized(self, counts: Tensor) -> Tensor:
        if self.is_normalized:
            return self.punctuation + self.casing + self.structure
        return (
            torch.stack((self.punctuation, self.casing, self.structure)) / counts.clamp(min=1)
        ).sum()


def target_counts(
    batch: Batch, *, balanced: bool = False, structure_enabled: bool = True
) -> Tensor:
    base = torch.stack(
        tuple((value != IGNORE).sum() for value in (batch.marks, batch.casing, batch.structure))
    )
    if not structure_enabled:
        base[2] = 0
    if not balanced:
        return base
    return torch.cat(
        (
            base,
            torch.stack(
                (
                    (batch.marks > 0).sum(),
                    (batch.marks == 0).sum(),
                    *((batch.marks == label).sum() for label in range(1, 9)),
                    *((batch.casing == label).sum() for label in range(3)),
                )
            ),
        )
    )


def loss_sums(
    scores: Scores,
    batch: Batch,
    *,
    denominators: Tensor | None = None,
    balanced: bool = False,
    structure_enabled: bool = True,
) -> Losses:
    valid = batch.marks != IGNORE
    positive = batch.marks > 0
    presence = functional.binary_cross_entropy_with_logits(
        scores.presence.float(), positive.float(), reduction="none"
    )
    types = functional.cross_entropy(
        scores.punctuation.float().transpose(1, 2), (batch.marks - 1).clamp(min=0), reduction="none"
    )
    punctuation = (presence * valid + types * positive).sum()
    cases = functional.cross_entropy(
        scores.casing.float().transpose(1, 2), batch.casing, ignore_index=IGNORE, reduction="none"
    )
    structure = (
        functional.cross_entropy(
            scores.structure.float().transpose(1, 2),
            batch.structure,
            ignore_index=IGNORE,
            reduction="sum",
        )
        if structure_enabled
        else scores.structure.sum() * 0
    )
    if not balanced:
        return Losses(punctuation, cases.sum(), structure)
    if denominators is None:
        message = "balanced loss requires complete-update class denominators"
        raise ValueError(message)
    presence_parts = (
        torch.stack(((presence * positive).sum(), (presence * (valid & ~positive)).sum()))
        / denominators[3:5].clamp(min=1)
        / (denominators[3:5] > 0).sum().clamp(min=1)
    )
    type_sums = torch.zeros(8, device=types.device).scatter_add(
        0, (batch.marks - 1).clamp(min=0).flatten(), (types * positive).flatten()
    )
    type_parts = (
        type_sums / denominators[5:13].clamp(min=1) / (denominators[5:13] > 0).sum().clamp(min=1)
    )
    case_sums = torch.zeros(3, device=cases.device).scatter_add(
        0, batch.casing.clamp(min=0).flatten(), cases.flatten()
    )
    case_parts = (
        case_sums / denominators[13:16].clamp(min=1) / (denominators[13:16] > 0).sum().clamp(min=1)
    )
    structural_part = structure / denominators[2].clamp(min=1)
    parts = torch.cat((presence_parts, type_parts, case_parts, structural_part.reshape(1)))
    return Losses(
        presence_parts.sum() + type_parts.sum(), case_parts.sum(), structural_part, True, parts
    )
