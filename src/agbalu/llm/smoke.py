"""The HPC smoke that gates the full run (task 11.19).

Four parts, each one invocation of under fifteen minutes:

- `throughput`: the 4B at the real row shape, one micro-batch a step, for step time, tokens
  per second, model-FLOP utilisation, peak memory and data-loader wait, then a checkpoint.
- `parity`: the 0.8B pilot trained 200 steps with the 8-bit optimizer and with fp32 AdamW on
  the same blocks. The 4B cannot hold an fp32 reference, so the pilot is where the 8-bit
  arm is shown to train like it.
- `resume`: the pilot run straight through, and the same run stopped at a checkpoint and
  restarted. The restarted losses must match the continuous ones, and must not match them
  shifted by a step, which is the control that proves the comparison can fail.
- `evaluate`: every evaluation part on the 4B with small limits, and batched generation
  checked against each prompt generated alone.

Torch and the trainer are imported when a part runs, so the CLI and the launcher read
`SMOKE_PARTS` from here on a machine without a GPU stack.
"""

from __future__ import annotations

import gc
import statistics
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from agbalu.llm.bases import BASE, PILOT
from agbalu.llm.evaluate import (
    PARTS,
    Tokenizer,
    first_line,
    flores_side,
    greedy,
    run,
    translation_prompt,
)
from agbalu.llm.prompting import SHOTS, shots
from agbalu.llm.recipe import BLOCKS_PER_STEP, MICRO_BLOCKS

if TYPE_CHECKING:
    import torch

    from agbalu.llm.forward import LanguageModel

    Loader = Callable[[str | Path, torch.dtype], LanguageModel]

Part = Literal["throughput", "parity", "resume", "evaluate"]
SMOKE_PARTS: Final[tuple[Part, ...]] = ("throughput", "parity", "resume", "evaluate")

THROUGHPUT_STEPS: Final = 20
PARITY_STEPS: Final = 200
PARITY_WINDOW: Final = 50
PARITY_TOLERANCE: Final = 0.01
"""Largest relative gap in mean loss over the last `PARITY_WINDOW` steps the 8-bit arm may
show against fp32 AdamW."""

RESUME_STEPS: Final = 30
RESUME_AT: Final = 15
RESUME_TOLERANCE: Final = 0.01
"""Nats. Larger than the run-to-run noise of non-deterministic kernels over fifteen steps,
far smaller than the step-to-step change the shifted control measures."""

CONTROL_RATIO: Final = 5.0
"""The shifted comparison must differ at least this many times more than the aligned one."""

MIN_COMPARED: Final = 2
"""The shifted control needs one step beyond the first compared."""

EVALUATION_LIMIT: Final = 16
GENERATION_PROMPTS: Final = 6
GENERATION_TOKENS: Final = 48
HELDOUT_LIMIT: Final = 200


class SmokeError(Exception):
    """A smoke part cannot be judged from what it produced."""


def parity_verdict(
    eight_bit: Mapping[int, float], reference: Mapping[int, float], window: int = PARITY_WINDOW
) -> dict[str, object]:
    """Mean loss over the last `window` steps of each arm, and the reference's own noise."""
    if set(eight_bit) != set(reference) or len(reference) < window:
        message = f"arms logged {len(eight_bit)} and {len(reference)} steps; need {window} shared"
        raise SmokeError(message)
    tail = sorted(reference)[-window:]
    eight = statistics.fmean(eight_bit[s] for s in tail)
    full = statistics.fmean(reference[s] for s in tail)
    relative = abs(eight - full) / full
    return {
        "steps": len(reference),
        "window": window,
        "adamw8bit_mean": round(eight, 5),
        "adamw_mean": round(full, 5),
        "relative_gap": round(relative, 5),
        "reference_step_stdev": round(statistics.stdev(reference[s] for s in tail), 5),
        "tolerance": PARITY_TOLERANCE,
        "passed": relative <= PARITY_TOLERANCE,
    }


def resume_verdict(
    continuous: Mapping[int, float], resumed: Mapping[int, float], after: int = RESUME_AT
) -> dict[str, object]:
    """Restarted losses against continuous ones, aligned and shifted by one step."""
    steps = [s for s in sorted(continuous) if s > after]
    missing = [s for s in steps if s not in resumed]
    if len(steps) < MIN_COMPARED or missing:
        message = f"the resumed run lacks steps {missing[:5]} after {after}"
        raise SmokeError(message)
    aligned = [abs(continuous[s] - resumed[s]) for s in steps]
    shifted = [abs(continuous[s] - resumed[s + 1]) for s in steps[:-1]]
    aligned_mean = statistics.fmean(aligned)
    shifted_mean = statistics.fmean(shifted)
    return {
        "after": after,
        "compared": len(steps),
        "aligned_max": round(max(aligned), 6),
        "aligned_mean": round(aligned_mean, 6),
        "shifted_mean": round(shifted_mean, 6),
        "tolerance": RESUME_TOLERANCE,
        "passed": max(aligned) <= RESUME_TOLERANCE and shifted_mean >= CONTROL_RATIO * aligned_mean,
    }


def release() -> None:
    """Return a finished model's memory before the next one loads."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def throughput(
    run_id: str, blocks: Path, load: Loader, tokenizer: Tokenizer, device: torch.device
) -> dict[str, object]:
    from agbalu.llm.train import RunConfig, block_shape, train

    config = RunConfig(
        run=f"{run_id}/throughput",
        model=BASE,
        blocks=blocks,
        blocks_per_step=MICRO_BLOCKS,
        total_steps=THROUGHPUT_STEPS,
        eval_every=THROUGHPUT_STEPS // 2,
        heldout_limit=HELDOUT_LIMIT,
        checkpoint_every=THROUGHPUT_STEPS,
    )
    result = train(config, load, tokenizer, device)
    full_steps = block_shape(blocks)[0] // BLOCKS_PER_STEP
    median = result["step_seconds_median"]
    projected = (
        full_steps * median * (BLOCKS_PER_STEP // MICRO_BLOCKS) / 3600
        if isinstance(median, float)
        else None
    )
    return {
        **result,
        "full_run_steps": full_steps,
        "projected_full_run_hours": round(projected, 1) if projected is not None else None,
        "projection": "median micro-step time x micro-steps per real step x real steps",
    }


def parity(
    run_id: str, blocks: Path, load: Loader, tokenizer: Tokenizer, device: torch.device
) -> dict[str, object]:
    from agbalu.llm.train import RunConfig, losses, train

    curves: dict[str, dict[int, float]] = {}
    for arm in ("adamw8bit", "adamw"):
        config = RunConfig(
            run=f"{run_id}/parity-{arm}",
            model=PILOT,
            blocks=blocks,
            optimizer=arm,
            blocks_per_step=MICRO_BLOCKS,
            total_steps=PARITY_STEPS,
            eval_every=0,
            probe=False,
            checkpoint_every=PARITY_STEPS,
        )
        train(config, load, tokenizer, device)
        curves[arm] = losses(config.directory / "log.jsonl")
        release()
    return parity_verdict(curves["adamw8bit"], curves["adamw"])


def resume(
    run_id: str, blocks: Path, load: Loader, tokenizer: Tokenizer, device: torch.device
) -> dict[str, object]:
    from agbalu.llm.train import RunConfig, losses, train

    continuous = RunConfig(
        run=f"{run_id}/resume-continuous",
        model=PILOT,
        blocks=blocks,
        blocks_per_step=MICRO_BLOCKS,
        total_steps=RESUME_STEPS,
        eval_every=0,
        probe=False,
        checkpoint_every=RESUME_STEPS,
    )
    train(continuous, load, tokenizer, device)
    release()
    restarted = replace(continuous, run=f"{run_id}/resume-restarted", stop_after=RESUME_AT)
    first = train(restarted, load, tokenizer, device)
    release()
    second = train(replace(restarted, stop_after=None), load, tokenizer, device)
    release()
    if first["ended"] != "stop_after" or second["step"] != RESUME_STEPS:
        message = f"the restart drill ended {first['ended']} then at step {second['step']}"
        raise SmokeError(message)
    return resume_verdict(
        losses(continuous.directory / "log.jsonl"), losses(restarted.directory / "log.jsonl")
    )


def generation_parity(
    model: LanguageModel, tokenizer: Tokenizer, device: torch.device
) -> dict[str, object]:
    """Greedy output of a left-padded batch against each prompt generated alone."""
    examples = shots(flores_side("eng", "dev"), flores_side("kab", "dev"), SHOTS)
    sources = flores_side("eng", "devtest")[:GENERATION_PROMPTS]
    prompts = [translation_prompt(examples, "eng_Latn", "kab_Latn", s) for s in sources]
    budgets = [GENERATION_TOKENS] * len(prompts)
    batched = greedy(model, tokenizer, prompts, budgets, device=device, batch_size=len(prompts))
    alone = [
        greedy(model, tokenizer, [prompt], [GENERATION_TOKENS], device=device, batch_size=1)[0]
        for prompt in prompts
    ]
    equal = sum(first_line(a) == first_line(b) for a, b in zip(batched, alone, strict=True))
    return {
        "prompts": len(prompts),
        "identical_first_lines": equal,
        "passed": equal == len(prompts),
        "pairs": [{"batched": a, "alone": b} for a, b in zip(batched, alone, strict=True)],
    }


def evaluation(load: Loader, tokenizer: Tokenizer, device: torch.device) -> dict[str, object]:
    import torch

    model = load(BASE, torch.bfloat16)
    report = run(model, tokenizer, PARTS, device=device, limit=EVALUATION_LIMIT)
    report["generation_parity"] = generation_parity(model, tokenizer, device)
    return report
