"""The CPT recipe: the schedule and the replay budget the stream is sized by."""

from __future__ import annotations

import itertools
import math

import pytest

from agbalu.llm.recipe import (
    ALIGNED_PASSES,
    BLOCKS_PER_STEP,
    FINAL_LR_FRACTION,
    KABYLE_PASSES,
    MICRO_BLOCKS,
    PEAK_LR,
    REPLAY_SHARE,
    SEQUENCE_LENGTH,
    TOKENS_PER_STEP,
    WARMUP_FRACTION,
    RecipeError,
    learning_rate,
    replay_budget,
)


class TestGeometry:
    def test_a_step_is_524288_tokens(self) -> None:
        assert TOKENS_PER_STEP == SEQUENCE_LENGTH * BLOCKS_PER_STEP == 524_288

    def test_micro_batches_divide_a_step(self) -> None:
        assert BLOCKS_PER_STEP % MICRO_BLOCKS == 0


class TestLearningRate:
    def test_step_zero_is_not_zero(self) -> None:
        assert learning_rate(0, 1000) > 0

    def test_warmup_reaches_the_peak(self) -> None:
        warmup = int(1000 * WARMUP_FRACTION)
        assert learning_rate(warmup - 1, 1000) == pytest.approx(PEAK_LR)

    def test_it_decays_to_the_floor_at_the_last_step(self) -> None:
        last = learning_rate(999, 1000)
        assert last == pytest.approx(PEAK_LR * FINAL_LR_FRACTION, rel=1e-3)

    def test_it_never_rises_after_warmup(self) -> None:
        values = [learning_rate(step, 500) for step in range(5, 500)]
        assert all(a >= b for a, b in itertools.pairwise(values))

    def test_a_one_step_run_trains_at_the_peak(self) -> None:
        assert learning_rate(0, 1) == pytest.approx(PEAK_LR)

    @pytest.mark.parametrize(("step", "total"), [(-1, 10), (10, 10), (0, 0)])
    def test_out_of_range_steps_are_refused(self, step: int, total: int) -> None:
        with pytest.raises(RecipeError):
            learning_rate(step, total)


class TestReplayBudget:
    def test_replay_is_its_share_of_the_whole_stream(self) -> None:
        kabyle, aligned = 90_000_000, 30_000_000
        replay = replay_budget(kabyle, aligned)
        stream = KABYLE_PASSES * kabyle + ALIGNED_PASSES * aligned + replay
        assert math.isclose(replay / stream, REPLAY_SHARE, rel_tol=1e-6)

    def test_no_kabyle_is_refused(self) -> None:
        with pytest.raises(RecipeError):
            replay_budget(0, 10)

    def test_no_aligned_text_is_allowed(self) -> None:
        assert replay_budget(3, 0) == round(4 * 3 * REPLAY_SHARE / (1 - REPLAY_SHARE))
