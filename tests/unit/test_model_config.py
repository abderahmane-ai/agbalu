from __future__ import annotations

from dataclasses import replace

import pytest

from agbalu.model.config import PRESETS, ModelConfig, ModelError, RunConfig, TrainConfig


class TestModelConfig:
    def test_rejects_a_head_count_that_does_not_divide_the_width(self) -> None:
        with pytest.raises(ModelError, match="not divisible"):
            ModelConfig(hidden_size=384, num_attention_heads=5)

    @pytest.mark.parametrize(
        "field",
        ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers"],
    )
    def test_rejects_non_positive_dimensions(self, field: str) -> None:
        with pytest.raises(ModelError, match=field):
            ModelConfig(**{field: 0})

    def test_head_size_divides_the_width(self) -> None:
        config = ModelConfig(hidden_size=384, num_attention_heads=6)
        assert config.head_size == 64

    def test_embedding_share_is_what_the_budget_argument_rests_on(self) -> None:
        """`docs/architecture_design.md` §2: 32k at d=384 is 33.0%, 8k is 11.0%.

        This test exists because the doc's first table was hand-computed and wrong.
        The doc now quotes these numbers, so it cannot drift from the code again."""
        wide = replace(PRESETS["kab"], vocab_size=32_000)
        narrow = replace(PRESETS["kab"], vocab_size=8_000)
        assert wide.embedding_parameters / wide.parameters == pytest.approx(0.330, abs=0.002)
        assert narrow.embedding_parameters / narrow.parameters == pytest.approx(0.110, abs=0.002)

    def test_the_shipped_preset_is_about_thirty_million(self) -> None:
        assert 30_000_000 < PRESETS["kab"].parameters < 32_000_000

    def test_presets_carry_the_reference_geometry(self) -> None:
        for name in ("small", "kab"):
            config = PRESETS[name]
            assert (config.hidden_size, config.intermediate_size) == (384, 1_280)
            assert (config.num_attention_heads, config.num_hidden_layers) == (6, 12)
        assert PRESETS["base"].hidden_size == 768
        assert all(config.position_bucket_size == 32 for config in PRESETS.values())


class TestTrainConfigValidation:
    def test_rejects_a_non_positive_learning_rate(self) -> None:
        with pytest.raises(ModelError, match="learning_rate"):
            TrainConfig(learning_rate=0.0)

    def test_rejects_a_global_batch_that_is_not_a_multiple_of_the_local_one(self) -> None:
        with pytest.raises(ModelError, match="not divisible"):
            TrainConfig(global_batch_size=100, local_batch_size=32)

    def test_rejects_a_hybrid_ratio_above_one(self) -> None:
        with pytest.raises(ModelError, match="not a proportion"):
            TrainConfig(hybrid_numerator=17, hybrid_denominator=16)

    def test_rejects_replacement_probabilities_that_exceed_one(self) -> None:
        with pytest.raises(ModelError, match="proportion"):
            TrainConfig(mask_random_p=0.7, mask_keep_p=0.7)

    def test_rejects_warmup_plus_cooldown_covering_the_whole_run(self) -> None:
        with pytest.raises(ModelError, match="exceed the whole run"):
            TrainConfig(warmup_proportion=0.7, cooldown_proportion=0.5)

    def test_rejects_a_run_of_no_steps(self) -> None:
        with pytest.raises(ModelError, match="max_steps"):
            TrainConfig(max_steps=0)


class TestDerivedQuantities:
    def test_accumulation_reaches_the_global_batch(self) -> None:
        config = TrainConfig(global_batch_size=32_768, local_batch_size=256)
        assert config.accumulation_steps == 128
        assert config.accumulation_steps * config.local_batch_size == config.global_batch_size

    def test_tokens_per_step_is_the_reference_ramp_floor(self) -> None:
        """1.05M, the batch the reference ramp starts at — not its 4.19M ceiling, which
        would buy 1,425 optimiser steps inside 24 hours instead of 4,500."""
        assert TrainConfig().tokens_per_step == 8_192 * 128

    def test_the_default_run_fits_modals_ceiling(self) -> None:
        """The budget is wall clock, not steps: at the 70,200 tok/s measured eager on an
        A10 the 24-hour ceiling is 6.07B tokens, and `max_steps` is chosen against it.
        Sized against the eager path because that is what `compile=False` selects."""
        config = TrainConfig()
        seconds = config.max_steps * config.tokens_per_step / 70_200
        assert seconds < 24 * 60 * 60
        assert seconds > 12 * 60 * 60, "a run this far under the ceiling is leaving budget unspent"

    def test_masked_ratio_is_the_hybrid_fraction(self) -> None:
        assert TrainConfig().masked_ratio == 15 / 16


class TestSchedules:
    def test_masking_anneals_from_start_to_end(self) -> None:
        config = TrainConfig(max_steps=1_000)
        assert config.mask_probability(0) == pytest.approx(config.mask_p_start)
        assert config.mask_probability(1_000) == pytest.approx(config.mask_p_end)
        assert config.mask_probability(500) == pytest.approx(0.225)

    def test_masking_is_clamped_outside_the_run(self) -> None:
        config = TrainConfig(max_steps=100)
        assert config.mask_probability(-5) == pytest.approx(config.mask_p_start)
        assert config.mask_probability(10_000) == pytest.approx(config.mask_p_end)

    def test_masking_is_monotonic(self) -> None:
        config = TrainConfig(max_steps=100)
        values = [config.mask_probability(step) for step in range(101)]
        assert values == sorted(values, reverse=True)

    def test_learning_rate_warms_up_from_zero(self) -> None:
        config = TrainConfig(max_steps=1_000)
        assert config.learning_rate_at(0) == 0.0
        assert config.learning_rate_at(8) < config.learning_rate_at(16)


class TestContinuation:
    """A finished run is extended by moving `schedule_start`, never by raising `max_steps`
    alone: the schedule is a shape over the span, so moving only the end would drop a run
    that is already past its cooldown back into the cosine body at full rate."""

    def test_a_continuation_warms_up_again_from_its_own_start(self) -> None:
        config = TrainConfig(max_steps=9_000, schedule_start=4_500)
        assert config.learning_rate_at(4_500) == 0.0
        assert config.learning_rate_at(4_508) < config.learning_rate_at(4_516)

    def test_it_reaches_the_same_peak_and_ends_at_zero(self) -> None:
        config = TrainConfig(max_steps=9_000, schedule_start=4_500)
        peak = max(config.learning_rate_at(s) for s in range(4_500, 9_001))
        assert peak == pytest.approx(config.learning_rate, rel=0.02)
        assert config.learning_rate_at(9_000) == pytest.approx(0.0, abs=1e-9)

    def test_raising_max_steps_alone_would_restart_the_body_at_full_rate(self) -> None:
        """The failure the span exists to prevent: at step 4,500 of a 9,000-step run with
        no `schedule_start`, the rate is back near peak instead of annealed."""
        naive = TrainConfig(max_steps=9_000)
        continued = TrainConfig(max_steps=9_000, schedule_start=4_500)
        assert naive.learning_rate_at(4_500) > 0.5 * naive.learning_rate
        assert continued.learning_rate_at(4_500) == 0.0

    def test_the_span_is_what_remains(self) -> None:
        assert TrainConfig(max_steps=9_000, schedule_start=4_500).scheduled_steps == 4_500
        assert TrainConfig(max_steps=4_500).scheduled_steps == 4_500

    def test_a_continuation_holds_the_mask_rate_where_the_last_run_left_it(self) -> None:
        """Set both ends to `mask_p_end` and the inverse schedule stays put rather than
        climbing back to 30% and re-hardening the task."""
        config = TrainConfig(max_steps=9_000, schedule_start=4_500, mask_p_start=0.15)
        assert config.mask_probability(4_500) == pytest.approx(0.15)
        assert config.mask_probability(6_750) == pytest.approx(0.15)
        assert config.mask_probability(9_000) == pytest.approx(0.15)

    def test_a_start_at_or_past_the_end_is_refused(self) -> None:
        with pytest.raises(ModelError, match="leaves no steps"):
            TrainConfig(max_steps=4_500, schedule_start=4_500)

    def test_a_negative_start_is_refused(self) -> None:
        with pytest.raises(ModelError, match="must not be negative"):
            TrainConfig(schedule_start=-1)

    def test_learning_rate_peaks_after_warmup(self) -> None:
        config = TrainConfig(max_steps=1_000)
        peak = max(config.learning_rate_at(step) for step in range(1_000))
        assert peak == pytest.approx(config.learning_rate, rel=0.02)

    def test_learning_rate_decays_to_almost_nothing(self) -> None:
        config = TrainConfig(max_steps=1_000)
        assert config.learning_rate_at(1_000) == pytest.approx(0.0, abs=1e-9)

    def test_learning_rate_never_exceeds_the_configured_peak(self) -> None:
        config = TrainConfig(max_steps=500)
        assert all(
            config.learning_rate_at(step) <= config.learning_rate * 1.0001 for step in range(501)
        )

    def test_learning_rate_is_never_negative(self) -> None:
        config = TrainConfig(max_steps=500)
        assert all(config.learning_rate_at(step) >= 0.0 for step in range(600))


class TestRunConfig:
    def test_defaults_to_the_shipped_preset(self) -> None:
        assert RunConfig().model == PRESETS["kab"]

    def test_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            RunConfig().name = "other"  # type: ignore[misc]
