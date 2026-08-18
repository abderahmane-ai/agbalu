"""Architecture and training configuration for the encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal


class ModelError(Exception):
    """A model or training configuration is not constructible."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture. Field names follow the reference configs so the JSON is portable."""

    vocab_size: int = 16_000
    hidden_size: int = 384
    intermediate_size: int = 1_280
    num_attention_heads: int = 6
    num_hidden_layers: int = 12
    max_position_embeddings: int = 512
    position_bucket_size: int = 32
    """Log-bucketed relative positions, not RoPE. 32 at every reference model size."""
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            msg = (
                f"hidden_size {self.hidden_size} is not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
            raise ModelError(msg)
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("max_position_embeddings", self.max_position_embeddings),
            ("position_bucket_size", self.position_bucket_size),
        ):
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ModelError(msg)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def embedding_parameters(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def parameters(self) -> int:
        """Analytic count, for budgeting before anything is allocated.

        Tied classifier, so the output head adds only its own projection. Verified
        against the built module in `tests/unit/test_model_modeling.py`.
        """
        attention = 4 * self.hidden_size * self.hidden_size + 4 * self.hidden_size
        feed_forward = 3 * self.hidden_size * self.intermediate_size
        blocks = self.num_hidden_layers * (attention + feed_forward)
        relative = (2 * self.position_bucket_size - 1) * self.hidden_size
        classifier = self.hidden_size * self.hidden_size + self.hidden_size + self.vocab_size
        return self.embedding_parameters + blocks + relative + classifier + 2 * self.hidden_size


Preset = Literal["small", "kab", "base"]

PRESETS: Final[dict[Preset, ModelConfig]] = {
    # ltgoslo/gpt-bert configs/small.json, the BabyLM 10M-track winner, with its own
    # 8,192 vocabulary.
    "small": ModelConfig(
        vocab_size=8_192,
        hidden_size=384,
        intermediate_size=1_280,
        num_attention_heads=6,
        num_hidden_layers=12,
    ),
    # Ours: the same architecture carrying our 16,000 vocabulary. 31.1M parameters,
    # 20% of them the embedding table (`docs/architecture_design.md` §2).
    "kab": ModelConfig(
        vocab_size=16_000,
        hidden_size=384,
        intermediate_size=1_280,
        num_attention_heads=6,
        num_hidden_layers=12,
    ),
    # configs/base.json, the BabyLM 100M-track winner. Three times our data.
    "base": ModelConfig(
        vocab_size=16_000,
        hidden_size=768,
        intermediate_size=2_560,
        num_attention_heads=12,
        num_hidden_layers=12,
    ),
}


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation. Defaults interpolate the reference 10M and 100M recipes.

    The two recipes differ in exactly four places — config, learning rate, local batch
    size and step count — so interpolating is well defined rather than a guess.
    """

    seq_length: int = 128
    global_batch_size: int = 8_192
    """Sequences per optimiser step, reached by gradient accumulation. 8,192 × 128 = 1.05M
    tokens per step — the *starting* batch of the reference ramp (1M → 4M tokens), held flat.
    The ramp's final 4.19M buys only 1,425 optimiser steps inside 24 hours at our measured
    throughput, against the reference's 7,812; the same token budget spent at 1.05M gives
    4,500. LAMB at 1.2e-2 was validated by the reference at this batch."""
    local_batch_size: int = 256

    learning_rate: float = 1.2e-2
    """Between the references' 1.41e-2 at 10M words and 1.0e-2 at 100M. LAMB, not AdamW:
    the reference uses LAMB at this learning rate and it does not transfer to Adam."""
    optimizer: Literal["lamb", "adamw"] = "lamb"
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_gradient: float = 2.0

    max_steps: int = 4_500
    """4.72B tokens at 1.05M per step: 67 epochs over the 70,184,279 training tokens in
    `artifacts/model-data/tokenised.stats.json`, and 18.7 hours at the 70,200 tok/s measured
    eager on an A10 — inside Modal's 24-hour ceiling with margin. Compiled it is 10.4 hours,
    but the ceiling is sized against the slower path, which is the default.

    Fewer epochs than either reference (strict-small ~1,200, strict ~240). That is the
    ceiling, not a choice: 24 hours at this throughput is 6.07B tokens, and reaching the
    reference's 15.6B inside the same window needs 180,556 tok/s.

    A finished run is extended by `schedule_start`, never by raising this alone: the
    schedule is a shape over `[schedule_start, max_steps]`, so moving only the end would
    put a run that is already past its cooldown back into the cosine body at full rate."""
    schedule_start: int = 0
    """Where this run's schedule begins. 0 for a fresh run; the previous `max_steps` for a
    continuation, which then gets its own warmup, cosine and cooldown over what remains."""
    warmup_proportion: float = 0.016
    cooldown_proportion: float = 0.016

    mask_p_start: float = 0.3
    mask_p_end: float = 0.15
    """Inverse masking schedule: hard early, standard late. Independently arrived at by
    mmBERT, which anneals 30% → 15% → 5%."""
    mask_random_p: float = 0.1
    mask_keep_p: float = 0.1
    max_span_length: int = 3

    hybrid_numerator: int = 15
    hybrid_denominator: int = 16
    """Masked:causal mixing from GPT-BERT. 15/16 of each batch is masked-LM, 1/16 is
    causal. The paper's abstract quotes 1:7; both shipped recipes use 15:16."""

    z_loss_weight: float = 1e-4
    mixed_precision: bool = True
    compile: bool = False
    """`torch.compile` the forward pass: 1.79x on an A10, 70,200 to 125,700 tok/s at
    1,048,576 tokens per step. Off by default because compilation costs ~90 s per
    container — once for the training graph and once at the first evaluation — which a run
    shorter than a few hundred steps does not earn back."""

    seed: int = 20260807
    validate_every: int = 250
    """18 evaluations across the run, each carrying masked-prediction previews, for ~2
    minutes of the wall clock in total."""
    validation_batches: int = 20
    log_every: int = 50
    save_every: int = 100
    """Each save writes weights plus both LAMB moments — ~356 MiB at this size, to a
    network volume, measured at ~2.9 s. 45 saves across the run costs ~2 minutes and caps
    what an interruption can destroy at ~25 minutes; at 500 the first cancelled run lost
    150 steps and could have lost 500. `lock.STALE_AFTER` must stay above this interval,
    because the checkpoint is what refreshes the heartbeat."""

    def __post_init__(self) -> None:
        if not self.learning_rate > 0.0:
            msg = f"learning_rate must be positive, got {self.learning_rate}"
            raise ModelError(msg)
        if self.global_batch_size % self.local_batch_size:
            msg = (
                f"global_batch_size {self.global_batch_size} is not divisible by "
                f"local_batch_size {self.local_batch_size}"
            )
            raise ModelError(msg)
        if not 0 <= self.hybrid_numerator <= self.hybrid_denominator:
            msg = (
                f"hybrid ratio {self.hybrid_numerator}/{self.hybrid_denominator} "
                "is not a proportion"
            )
            raise ModelError(msg)
        if self.hybrid_denominator < 1:
            msg = "hybrid_denominator must be positive"
            raise ModelError(msg)
        if not 0.0 <= self.mask_random_p + self.mask_keep_p <= 1.0:
            msg = "mask_random_p + mask_keep_p must be a proportion"
            raise ModelError(msg)
        if self.warmup_proportion + self.cooldown_proportion > 1.0:
            msg = "warmup and cooldown together exceed the whole run"
            raise ModelError(msg)
        if self.max_steps < 1:
            msg = f"max_steps must be positive, got {self.max_steps}"
            raise ModelError(msg)
        if self.schedule_start < 0:
            msg = f"schedule_start must not be negative, got {self.schedule_start}"
            raise ModelError(msg)
        if self.schedule_start >= self.max_steps:
            msg = (
                f"schedule_start {self.schedule_start} leaves no steps before "
                f"max_steps {self.max_steps}"
            )
            raise ModelError(msg)

    @property
    def masked_ratio(self) -> float:
        return self.hybrid_numerator / self.hybrid_denominator

    @property
    def accumulation_steps(self) -> int:
        return self.global_batch_size // self.local_batch_size

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_size * self.seq_length

    @property
    def scheduled_steps(self) -> int:
        """Steps this run's schedule spans. A continuation shapes only its own span."""
        return max(self.max_steps - self.schedule_start, 1)

    def mask_probability(self, step: int) -> float:
        """Linear from `mask_p_start` to `mask_p_end` across the run.

        A continuation sets both ends to `mask_p_end`, so the rate stays where the
        previous run left it rather than climbing back to 30%.
        """
        elapsed = min(max(step - self.schedule_start, 0), self.scheduled_steps)
        progress = elapsed / self.scheduled_steps
        return self.mask_p_start + (self.mask_p_end - self.mask_p_start) * progress

    def learning_rate_at(self, step: int) -> float:
        """Linear warmup, cosine body, linear cooldown — the reference schedule."""
        span_steps = self.scheduled_steps
        elapsed = min(max(step - self.schedule_start, 0), span_steps)
        warmup = self.warmup_proportion * span_steps
        cooldown = self.cooldown_proportion * span_steps
        if elapsed < warmup:
            return self.learning_rate * elapsed / max(warmup, 1.0)
        if elapsed > span_steps - cooldown:
            remaining = max(span_steps - elapsed, 0)
            return self.learning_rate * 0.1 * remaining / max(cooldown, 1.0)
        body = span_steps - warmup - cooldown
        progress = (elapsed - warmup) / max(body, 1.0)
        return self.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


DEFAULT_RUN_NAME: Final = "agbalu-encoder-v1"
"""Names the run directory on both the Modal checkpoint volume and `artifacts/runs`, so
the two layouts stay interchangeable and a pulled-down checkpoint needs no renaming."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    model: ModelConfig = field(default_factory=lambda: PRESETS["kab"])
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = DEFAULT_RUN_NAME
