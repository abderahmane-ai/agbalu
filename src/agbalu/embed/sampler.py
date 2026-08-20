"""Cluster-aware batch sampling for contrastive sentence representation learning.

In InfoNCE and MultipleNegativesRankingLoss, every non-target passage in a mini-batch
acts as an in-batch negative. When two training pairs belong to the same semantic
cluster (e.g., two distinct paraphrases from the same TaPaCo cluster), standard random
batching places them in the same mini-batch with non-negligible probability.

When this happens:
1. The loss computes contrastive similarity between anchor $A_i$ and negative $P_j$.
2. Because $P_j$ is a true paraphrase of $A_i$, the loss actively penalises the model
   for placing their embeddings close together (false-negative gradient penalty).
3. The embedding space fragments semantic clusters rather than aligning them.

`ClusterAwareBatchSampler` guarantees that every yielded batch contains examples from
$K$ strictly distinct `cluster_id`s, eliminating in-batch false negative collisions
without sacrificing batch diversity or epoch coverage.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ClusterAwareBatchSampler(Sampler[list[int]]):
    """Yields mini-batches of dataset indices ensuring at most one item per cluster per batch."""

    def __init__(
        self,
        cluster_ids: Sequence[str],
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            message = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(message)
        self.cluster_ids = tuple(cluster_ids)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self._cluster_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, cid in enumerate(self.cluster_ids):
            self._cluster_to_indices[cid].append(idx)
        self._clusters = tuple(sorted(self._cluster_to_indices.keys()))

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic seeded shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)  # noqa: S311
        cluster_queues: dict[str, list[int]] = {
            cid: list(indices) for cid, indices in self._cluster_to_indices.items()
        }
        for indices in cluster_queues.values():
            rng.shuffle(indices)

        available_clusters = [cid for cid in self._clusters if cluster_queues[cid]]
        rng.shuffle(available_clusters)

        while len(available_clusters) >= self.batch_size:
            # Pick batch_size distinct clusters
            batch_clusters = available_clusters[: self.batch_size]
            batch = [cluster_queues[cid].pop() for cid in batch_clusters]

            # Remove clusters whose items are exhausted
            available_clusters = [cid for cid in available_clusters if cluster_queues[cid]]
            rng.shuffle(available_clusters)
            yield batch

        if not self.drop_last and available_clusters:
            batch = [cluster_queues[cid].pop() for cid in available_clusters if cluster_queues[cid]]
            if batch:
                yield batch

    def __len__(self) -> int:
        total_items = len(self.cluster_ids)
        if self.drop_last:
            return total_items // self.batch_size
        return (total_items + self.batch_size - 1) // self.batch_size
