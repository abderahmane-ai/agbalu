"""The three objectives that put gradient on every dialect rather than on the average.

Each answers a different failure the first classifier showed on the held-out set:

- **LDAM margins** answer the long tail. Shawiya has 447 training lines against
  Tashelhit's 24,123, and a shared margin gives the rare class the same slack as the
  common one. `m_k = C / n_k^(1/4)` is the Rademacher-complexity bound from Cao et
  al., NeurIPS 2019 (arXiv 1906.07413).
- **Group-DRO** answers the domain confound. Weighting the worst `(language, domain)`
  cell up, rather than averaging the batch, is what stops Tamazight-on-the-web being
  traded for Tamazight-in-a-phrasebook: its recall was 23.2% while the mean looked
  healthy. Sagawa et al., ICLR 2020 (arXiv 1911.08731).
- **Domain-aware supervised contrastive learning** answers domain entanglement
  directly. Negatives are drawn first from a *different language in the same domain*,
  so the only way to separate them is the language. Khosla et al. NeurIPS 2020 for the
  loss; the hierarchical hard-negative scheme and the memory bank are ConLID's, which
  measured +3.23 F1 on low-resource languages and +5.41 on multi-domain ones
  (arXiv 2506.15304).

`logit_adjustment` is the inference-time counterpart of the margins: the shift
`logit_k - tau * log(prior_k)` from Menon et al., ICLR 2021 (arXiv 2007.07314), which
makes the argmax minimise balanced error instead of the training prior's error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional

LDAM_EXPONENT: Final = 0.25
EPSILON: Final = 1e-12


def ldam_margins(counts: Sequence[int], scale: float) -> Tensor:
    """Per-class additive margins `C / n_k^(1/4)`, normalised so the largest is `C`.

    A class with no training examples gets no margin rather than an infinite one: it
    cannot be learned, and an infinite margin would make its logit unreachable instead
    of merely unlearned.
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
    """`tau * log(prior)`, to be subtracted from the logits at inference."""
    counted = torch.tensor([max(int(n), 0) for n in counts], dtype=torch.float)
    total = counted.sum()
    if total <= 0 or tau == 0.0:
        return torch.zeros_like(counted)
    return tau * torch.log(counted / total + EPSILON)


class GroupWeights(nn.Module):
    """Online group-DRO: exponentiated-gradient weights over `(language, domain)` cells.

    The weights are model state, not statistics — they are what the loss is taken
    against — so they are registered as a buffer and travel in the checkpoint. A run
    resumed with uniform weights re-learns them from scratch and spends the first
    epoch back where it started.
    """

    weights: Tensor

    def __init__(self, num_groups: int, step: float) -> None:
        super().__init__()
        if num_groups < 1:
            msg = f"num_groups must be at least 1, got {num_groups}"
            raise ValueError(msg)
        self.step = step
        self.register_buffer("weights", torch.full((num_groups,), 1.0 / num_groups))

    def forward(self, losses: Tensor, groups: Tensor) -> Tensor:
        """Mean loss per group, reweighted; updates the weights when training.

        `losses` is per sample, `groups` the group index of each sample. Groups absent
        from the batch keep their weight and contribute nothing, which is what makes
        this usable with a sampler that cannot fit every group in one batch.
        """
        present, inverse = torch.unique(groups, return_inverse=True)
        totals = torch.zeros(present.numel(), device=losses.device, dtype=losses.dtype)
        totals.index_add_(0, inverse, losses)
        sizes = torch.zeros_like(totals).index_add_(0, inverse, torch.ones_like(losses))
        per_group = totals / sizes.clamp(min=1.0)

        if self.training and self.step > 0:
            with torch.no_grad():
                updated = self.weights.clone()
                updated[present] *= torch.exp(self.step * per_group.detach()).to(updated)
                self.weights.copy_(updated / updated.sum().clamp(min=EPSILON))

        active = self.weights[present].to(per_group)
        return (active * per_group).sum() / active.sum().clamp(min=EPSILON)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss with a memory bank and domain-aware hard negatives.

    The bank holds the most recent projections with their language and domain, so the
    negative pool is `batch + bank` rather than `batch`. Selection is hierarchical and
    falls back rather than failing: different language in the same domain first, then
    different language in any domain. A batch that offers no same-domain contrast is
    the normal case for a language that owns its domain outright, and refusing to
    train on it would drop exactly the classes that need the term.
    """

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
        start = int(self.bank_fill.item()) % self.capacity
        positions = (torch.arange(count, device=taken.device) + start) % self.capacity
        self.bank_projections.index_copy_(0, positions, taken.to(self.bank_projections))
        self.bank_labels.index_copy_(0, positions, labels[-count:].to(self.bank_labels))
        self.bank_domains.index_copy_(0, positions, domains[-count:].to(self.bank_domains))
        self.bank_fill += count

    def forward(self, projections: Tensor, labels: Tensor, domains: Tensor) -> Tensor:
        filled = self._filled()
        if filled:
            pool = torch.cat([projections, self.bank_projections[:filled].to(projections)])
            pool_labels = torch.cat([labels, self.bank_labels[:filled].to(labels)])
            pool_domains = torch.cat([domains, self.bank_domains[:filled].to(domains)])
        else:
            pool, pool_labels, pool_domains = projections, labels, domains

        similarity = projections @ pool.t() / self.temperature
        anchors = projections.size(0)
        identity = torch.zeros_like(similarity, dtype=torch.bool)
        identity[:, :anchors] = torch.eye(anchors, dtype=torch.bool, device=similarity.device)
        similarity = similarity.masked_fill(identity, -torch.inf)

        same_language = labels.unsqueeze(1) == pool_labels.unsqueeze(0)
        positives = same_language & ~identity
        same_domain = domains.unsqueeze(1) == pool_domains.unsqueeze(0)
        hard = ~same_language & same_domain
        negatives = torch.where(hard.any(dim=1, keepdim=True), hard, ~same_language)

        usable = positives.any(dim=1) & negatives.any(dim=1)
        if not bool(usable.any()):
            return projections.new_zeros(())

        considered = positives | negatives
        log_denominator = torch.logsumexp(
            similarity.masked_fill(~considered, -torch.inf), dim=1, keepdim=True
        )
        log_probabilities = similarity - log_denominator
        positive_count = positives.sum(dim=1).clamp(min=1)
        per_anchor = -(log_probabilities.masked_fill(~positives, 0.0).sum(dim=1) / positive_count)
        self.enqueue(projections, labels, domains)
        return per_anchor[usable].mean()


def cross_entropy_per_sample(logits: Tensor, labels: Tensor, smoothing: float) -> Tensor:
    """Unreduced cross-entropy, so group-DRO can weight it before it collapses."""
    return functional.cross_entropy(logits, labels, reduction="none", label_smoothing=smoothing)
