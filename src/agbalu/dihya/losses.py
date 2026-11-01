"""Loss objectives for Dihya-22M Pan-Amazigh LID.

Implements LDAM margins, ConLID domain-aware supervised contrastive learning with
memory bank, Group-DRO online re-weighting, and auxiliary multi-task heads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn

LDAM_EXPONENT: Final = 0.25
EPSILON: Final = 1e-12


def ldam_margins(counts: Sequence[int], scale: float) -> Tensor:
    """Calculate per-class additive margins C / n_k^(1/4) normalized to max scale.

    Args:
        counts: Class support counts.
        scale: Maximum margin scale constant C.

    Returns:
        Tensor of per-class margins.
    """
    counted = torch.tensor([max(int(n), 0) for n in counts], dtype=torch.float)
    margins = torch.where(
        counted > 0, counted.clamp(min=1.0).pow(-LDAM_EXPONENT), torch.zeros_like(counted)
    )
    largest = margins.max()
    if largest <= 0:
        return torch.zeros_like(margins)
    return margins / largest * scale


def logit_adjustment(counts: Sequence[int], tau: float) -> Tensor:
    """Compute inference prior shift tau * log(prior).

    Args:
        counts: Class support counts.
        tau: Adjustment temperature.

    Returns:
        Tensor of shifts to subtract from inference logits.
    """
    counted = torch.tensor([max(int(n), 0) for n in counts], dtype=torch.float)
    total = counted.sum()
    if total <= 0 or tau == 0.0:
        return torch.zeros_like(counted)
    return tau * torch.log(counted / total + EPSILON)


class GroupWeights(nn.Module):
    """Online group-DRO exponentiated-gradient weights over (variety, domain, script) cells."""

    log_weights: Tensor

    def __init__(self, num_groups: int, step: float) -> None:
        super().__init__()
        if num_groups < 1:
            msg = f"num_groups must be at least 1, got {num_groups}"
            raise ValueError(msg)
        self.step = step
        self.register_buffer("log_weights", torch.full((num_groups,), -math.log(num_groups)))

    @property
    def weights(self) -> Tensor:
        """Expose normalized probabilities without losing small weights in stored state."""
        return self.log_weights.softmax(dim=0)

    def forward(self, losses: Tensor, groups: Tensor) -> Tensor:
        """Reweight per-sample losses across groups and update group distribution."""
        present, inverse = torch.unique(groups, return_inverse=True)
        totals = torch.zeros(present.numel(), device=losses.device, dtype=torch.float32)
        totals.index_add_(0, inverse, losses.float())
        sizes = torch.zeros_like(totals).index_add_(0, inverse, torch.ones_like(losses).float())
        per_group = totals / sizes.clamp(min=1.0)

        if self.training and self.step > 0:
            with torch.no_grad():
                log_weights = self.log_weights.clone()
                # Uniform cell sampling observes each cell with probability k / G.
                log_weights[present] += (
                    self.step * per_group.detach() * (self.weights.numel() / present.numel())
                )
                self.log_weights.copy_(log_weights - torch.logsumexp(log_weights, dim=0))

        active: Tensor = self.weights[present].to(per_group)
        return (active * per_group).sum() / active.sum().clamp(min=1e-12)


class SupervisedContrastiveLoss(nn.Module):
    """ConLID supervised contrastive loss with memory bank and domain hard negatives."""

    bank_projections: Tensor
    bank_labels: Tensor
    bank_domains: Tensor
    bank_fill: Tensor

    def __init__(self, dim: int, capacity: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)
        self.capacity = max(int(capacity), 0)
        self.temperature = temperature
        self.register_buffer("bank_projections", torch.zeros(self.capacity, dim))
        self.register_buffer("bank_labels", torch.full((self.capacity,), -1, dtype=torch.long))
        self.register_buffer("bank_domains", torch.full((self.capacity,), -1, dtype=torch.long))
        self.register_buffer("bank_fill", torch.zeros((), dtype=torch.long))

    def _filled(self) -> int:
        return min(int(self.bank_fill.item()), self.capacity)

    @torch.no_grad()
    def enqueue(self, projections: Tensor, labels: Tensor, domains: Tensor) -> None:
        if not self.capacity:
            return
        taken = projections[-self.capacity :].detach()
        count = taken.size(0)
        positions = (torch.arange(count, device=taken.device) + self.bank_fill) % self.capacity
        self.bank_projections.index_copy_(0, positions, taken.to(self.bank_projections))
        self.bank_labels.index_copy_(0, positions, labels[-count:].to(self.bank_labels))
        self.bank_domains.index_copy_(0, positions, domains[-count:].to(self.bank_domains))
        self.bank_fill += count

    def forward(self, projections: Tensor, labels: Tensor, domains: Tensor) -> Tensor:
        pool = torch.cat([projections, self.bank_projections.to(projections)])
        pool_labels = torch.cat([labels, self.bank_labels.to(labels)])
        pool_domains = torch.cat([domains, self.bank_domains.to(domains)])
        valid = pool_labels.unsqueeze(0) >= 0

        similarity = projections.float() @ pool.float().t() / self.temperature
        anchors = projections.size(0)
        identity = torch.zeros_like(similarity, dtype=torch.bool)
        identity[:, :anchors] = torch.eye(anchors, dtype=torch.bool, device=similarity.device)

        same_language = labels.unsqueeze(1) == pool_labels.unsqueeze(0)
        positives = same_language & ~identity & valid
        same_domain = domains.unsqueeze(1) == pool_domains.unsqueeze(0)
        hard = ~same_language & same_domain & valid
        negatives = torch.where(hard.any(dim=1, keepdim=True), hard, ~same_language & valid)

        usable = positives.any(dim=1) & negatives.any(dim=1)
        if self.training:
            self.enqueue(projections, labels, domains)
        # A self entry gives unused anchors a finite denominator and exactly zero gradient.
        considered = ((positives | negatives) & usable.unsqueeze(1)) | (
            identity & ~usable.unsqueeze(1)
        )
        selected_positives = positives & usable.unsqueeze(1)
        log_denominator = torch.logsumexp(
            similarity.masked_fill(~considered, -torch.inf), dim=1, keepdim=True
        )
        log_probabilities = similarity - log_denominator
        positive_count = selected_positives.sum(dim=1).clamp(min=1)
        per_anchor = -(
            log_probabilities.masked_fill(~selected_positives, 0.0).sum(dim=1) / positive_count
        )
        result: Tensor = per_anchor.sum() / usable.sum().clamp(min=1)
        return result


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Detailed breakdown of multi-objective components."""

    total_loss: Tensor
    classification_loss: Tensor
    contrastive_loss: Tensor
    script_loss: Tensor
    filter_loss: Tensor
