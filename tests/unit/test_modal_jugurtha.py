"""The CPT trainer's pure parts: where it reads and writes, and how it shards a step.

Nothing here touches a GPU. What it pins is the arithmetic that decides which blocks a rank
trains on — two ranks sharing a block train the same tokens twice and report a step count
that says otherwise, and neither the loss curve nor the summary would show it.
"""

from __future__ import annotations

import numpy as np
import pytest
from modal_app.common import CHECKPOINT_PATH, DATA_PATH
from modal_app.jugurtha import (
    ACCUMULATION,
    DEFAULT_RUN,
    GPU,
    GPUS,
    MICRO_BATCH,
    _rows,
    blocks_path,
    order,
    run_directory,
)

from agbalu.llm.recipe import SHUFFLE_SEED


class TestPaths:
    def test_the_blocks_live_on_the_data_volume(self) -> None:
        assert blocks_path().is_relative_to(DATA_PATH)

    def test_a_run_writes_under_the_checkpoint_volume_in_its_own_directory(self) -> None:
        assert run_directory(DEFAULT_RUN).is_relative_to(CHECKPOINT_PATH)
        assert run_directory(DEFAULT_RUN).name == DEFAULT_RUN

    def test_two_runs_do_not_share_a_directory(self) -> None:
        """Resume reads the step counter out of the directory, so a collision restarts one
        run from the other's checkpoint."""
        assert run_directory("jugurtha-v1") != run_directory("jugurtha-smoke")

    def test_the_default_run_is_not_the_encoders(self) -> None:
        assert DEFAULT_RUN == "jugurtha-v1"


class TestGpuRequest:
    def test_the_request_names_the_count_it_shards_across(self) -> None:
        """FSDP2 shards to `GPUS`; asking Modal for a different number leaves the run either
        short of memory or paying for a card it never uses."""
        assert GPU.split(":") == ["A10", str(GPUS)]

    def test_it_is_two_cards_because_one_cannot_hold_the_optimizer(self) -> None:
        assert GPUS == 2


class TestOrder:
    def test_it_is_a_permutation_and_loses_no_block(self) -> None:
        permutation = order(1000, SHUFFLE_SEED)
        assert sorted(permutation.tolist()) == list(range(1000))

    def test_the_same_seed_gives_the_same_order_so_a_resume_continues_it(self) -> None:
        assert np.array_equal(order(500, SHUFFLE_SEED), order(500, SHUFFLE_SEED))

    def test_a_different_seed_gives_a_different_order(self) -> None:
        assert not np.array_equal(order(500, SHUFFLE_SEED), order(500, SHUFFLE_SEED + 1))

    def test_it_actually_shuffles_rather_than_returning_file_order(self) -> None:
        """The corpus is written source by source, so file order makes every batch one
        source and the loss tracks the source rather than the model."""
        permutation = order(1000, SHUFFLE_SEED)
        assert not np.array_equal(permutation, np.arange(1000))

    def test_no_blocks_is_an_empty_order_not_an_error(self) -> None:
        assert order(0, SHUFFLE_SEED).tolist() == []

    def test_a_single_block_is_its_own_order(self) -> None:
        assert order(1, SHUFFLE_SEED).tolist() == [0]


class TestRows:
    @staticmethod
    def _blocks(step: int, world: int, count: int) -> list[list[int]]:
        permutation = order(count, SHUFFLE_SEED)
        return [
            batch.tolist()
            for rank in range(world)
            for batch in _rows(step, rank, world, permutation, count)
        ]

    def test_a_rank_gets_one_micro_batch_per_accumulation_step(self) -> None:
        permutation = order(4096, SHUFFLE_SEED)
        batches = _rows(0, 0, GPUS, permutation, 4096)
        assert len(batches) == ACCUMULATION
        assert all(len(batch) == MICRO_BATCH for batch in batches)

    def test_two_ranks_never_train_the_same_block_in_one_step(self) -> None:
        """The whole point of the second card: overlapping ranks train the same tokens twice
        and the step count says they did not."""
        seen = [block for batch in self._blocks(0, GPUS, 4096) for block in batch]
        assert len(seen) == len(set(seen))
        assert len(seen) == ACCUMULATION * MICRO_BATCH * GPUS

    def test_consecutive_steps_do_not_revisit_the_same_blocks(self) -> None:
        first = {block for batch in self._blocks(0, GPUS, 4096) for block in batch}
        second = {block for batch in self._blocks(1, GPUS, 4096) for block in batch}
        assert first.isdisjoint(second)

    def test_the_step_consumes_the_leading_window_of_the_permutation(self) -> None:
        """Rank-interleaved rather than rank-major, so the window is what both ranks cover
        between them and not the order either one reads it in."""
        permutation = order(4096, SHUFFLE_SEED)
        window = ACCUMULATION * MICRO_BATCH * GPUS
        seen = {block for batch in self._blocks(0, GPUS, 4096) for block in batch}
        assert seen == set(permutation[:window].tolist())

    def test_it_wraps_rather_than_running_off_the_end_of_a_short_corpus(self) -> None:
        """A smoke packs far fewer blocks than a step consumes; indexing past the end would
        raise on the GPU after the image build is paid for."""
        count = 8
        seen = [block for batch in self._blocks(0, GPUS, count) for block in batch]
        assert len(seen) == ACCUMULATION * MICRO_BATCH * GPUS
        assert set(seen) <= set(range(count))

    def test_a_single_rank_takes_every_block_of_the_step(self) -> None:
        seen = [block for batch in self._blocks(0, 1, 4096) for block in batch]
        assert len(seen) == len(set(seen))
        assert len(seen) == ACCUMULATION * MICRO_BATCH

    @pytest.mark.parametrize("rank", [0, 1])
    def test_each_rank_reads_only_inside_the_corpus(self, rank: int) -> None:
        permutation = order(64, SHUFFLE_SEED)
        for batch in _rows(7, rank, GPUS, permutation, 64):
            assert all(0 <= int(block) < 64 for block in batch)
