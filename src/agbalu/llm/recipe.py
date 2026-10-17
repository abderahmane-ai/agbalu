"""The continued-pretraining recipe: block geometry, step size and schedule.

Full continued pretraining of every text-tower parameter, one stage. LoRA is refused on
measurement: in CPT at rank 256 it needs 20B tokens to match full fine-tuning at 4B
(arXiv 2405.09673, TMLR 2024), a five-fold sample inefficiency this corpus cannot pay.
The peak rate sits between AfriqueLLM's two optima on a 4B base (arXiv 2601.06395,
Table 10): 5e-5 was best for translation, 1e-5 for knowledge retention.
"""

from __future__ import annotations

import math
from typing import Final

SEQUENCE_LENGTH: Final = 8192
"""Tokens per packed block. AfriqueLLM measured 16k ahead of 4k on a 4B base; documents
here are mostly short, so a block holds many and attention is masked at each boundary."""

MICRO_BLOCKS: Final = 2
"""Blocks per forward pass, flattened into one variable-length row of 16,384 tokens."""

BLOCKS_PER_STEP: Final = 64
"""524,288 tokens an optimizer step."""

TOKENS_PER_STEP: Final = SEQUENCE_LENGTH * BLOCKS_PER_STEP

PEAK_LR: Final = 2e-5
WARMUP_FRACTION: Final = 0.01
FINAL_LR_FRACTION: Final = 0.1
WEIGHT_DECAY: Final = 0.1
BETAS: Final = (0.9, 0.95)
GRADIENT_CLIP: Final = 1.0

SHUFFLE_SEED: Final = 20260811
"""Seeds the document order the packer writes, which is the order the trainer reads."""

KABYLE_PASSES: Final = 4
"""Four epochs is where repetition stops being free (Muennighoff et al., JMLR 2025)."""

ALIGNED_PASSES: Final = 2
"""96% of the aligned Kabyle side repeats the monolingual text, so the pairs are seen half
as often: the signal they add is the alignment, not the Kabyle."""

REPLAY_SHARE: Final = 1 / 3
"""Replay's share of the token stream, read once."""

SFT_PEAK_LR: Final = 5e-6
SFT_WARMUP_FRACTION: Final = 0.03
SFT_WEIGHT_DECAY: Final = 0.01
SFT_ROW_TOKENS: Final = 8192
SFT_ROWS_PER_STEP: Final = 2
"""16,384 tokens an update. The set was cut to 812 conversations, 61 rows of 8,192
(2026-10-17); at 4 rows an epoch was 15 updates (`docs/llm_post_training_design.md` §5)."""

SFT_REPLAY_SHARE: Final = 0.10
"""Raw-text replay's share of the *supervised* tokens. Replay is mixed into the same rows as
the conversations and every supervised token weighs the same, so the loss is 90% conversation
and 10% replay in expectation in every update, including updates no replay row would reach
if replay were packed apart."""

SFT_SEED: Final = 20261016
"""Seeds the row packing and the per-epoch row order."""

SFT_EPOCHS: Final = 3
"""91 updates over the 61 rows. Small curated sets take several epochs (LIMA, arXiv
2305.11206: 1,000 examples, 15 epochs); the best checkpoint is still chosen on validation
loss, which catches the epoch where it starts to overfit."""

DPO_PEAK_LR: Final = 5e-7
DPO_BETA: Final = 0.1
DPO_PAIRS_PER_STEP: Final = 16
DPO_MAX_TOKENS: Final = 4096


class RecipeError(Exception):
    """A schedule cannot be built."""


def replay_budget(kabyle_tokens: int, aligned_tokens: int) -> int:
    """Replay tokens that make `REPLAY_SHARE` of the stream the Kabyle passes define."""
    if kabyle_tokens < 1 or aligned_tokens < 0:
        message = f"cannot size replay from {kabyle_tokens} Kabyle and {aligned_tokens} aligned"
        raise RecipeError(message)
    target = KABYLE_PASSES * kabyle_tokens + ALIGNED_PASSES * aligned_tokens
    return round(target * REPLAY_SHARE / (1 - REPLAY_SHARE))


def learning_rate(
    step: int, total: int, *, peak: float = PEAK_LR, warmup_fraction: float = WARMUP_FRACTION
) -> float:
    """Linear warmup then cosine decay to `FINAL_LR_FRACTION` of peak.

    `step` is zero-based and step 0 is not zero: a first step at lr 0 wastes an optimizer
    step, and on a short run a measurable share of the budget.
    """
    if total < 1:
        message = f"total steps must be positive, got {total}"
        raise RecipeError(message)
    if not 0 <= step < total:
        message = f"step {step} is outside a run of {total}"
        raise RecipeError(message)
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine)
