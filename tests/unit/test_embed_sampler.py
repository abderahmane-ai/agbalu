"""Unit tests for ClusterAwareBatchSampler."""

from __future__ import annotations

import pytest

from agbalu.embed.sampler import ClusterAwareBatchSampler


class TestClusterAwareBatchSampler:
    def test_refuses_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ClusterAwareBatchSampler(["c1", "c2"], batch_size=0)

    def test_no_cluster_collisions_in_any_batch(self) -> None:
        # Create 20 clusters with 3 items each = 60 items
        cluster_ids = [f"cluster_{i // 3}" for i in range(60)]
        sampler = ClusterAwareBatchSampler(cluster_ids, batch_size=8, seed=42)

        batches = list(sampler)
        assert len(batches) > 0

        for batch in batches:
            assert len(batch) == 8
            # Verify all items in the batch have distinct cluster IDs
            batch_clusters = [cluster_ids[idx] for idx in batch]
            assert len(set(batch_clusters)) == len(batch)

    def test_epoch_shuffling_changes_order_deterministically(self) -> None:
        cluster_ids = [f"cluster_{i // 2}" for i in range(40)]
        sampler = ClusterAwareBatchSampler(cluster_ids, batch_size=4, seed=123)

        sampler.set_epoch(0)
        epoch_0 = list(sampler)

        sampler.set_epoch(1)
        epoch_1 = list(sampler)

        assert epoch_0 != epoch_1

        # Re-running epoch 0 returns the exact same batches
        sampler.set_epoch(0)
        assert list(sampler) == epoch_0

    def test_handles_skewed_cluster_sizes(self) -> None:
        # One large cluster (10 items) and 10 small clusters (1 item each)
        cluster_ids = ["dominant"] * 10 + [f"small_{i}" for i in range(10)]
        sampler = ClusterAwareBatchSampler(cluster_ids, batch_size=4, drop_last=False, seed=42)

        batches = list(sampler)
        for batch in batches:
            batch_clusters = [cluster_ids[idx] for idx in batch]
            # No batch may contain more than one instance of 'dominant'
            assert batch_clusters.count("dominant") <= 1
            assert len(set(batch_clusters)) == len(batch)
