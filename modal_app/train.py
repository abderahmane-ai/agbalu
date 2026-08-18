"""Pretrain the encoder on a Modal GPU."""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from agbalu.model.config import DEFAULT_RUN_NAME
from agbalu.tokenizer.spec import CLS_ID, PAD_ID, SEP_ID, UNK_ID
from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    RETRIES,
    TRAIN_TIMEOUT,
    VOLUMES,
    app,
    checkpoint_volume,
    data_volume,
    image,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from agbalu.model.checkpoint import ResumeFrom
    from agbalu.model.config import TrainConfig
    from agbalu.model.data import PackedDataset
    from agbalu.model.preview import Previewer

GPU: Final = "A10"

N_SPECIAL_TOKENS: Final = 5
"""[PAD] [UNK] [CLS] [SEP] [MASK] — ids 0-4, never masked and never drawn as replacements."""
MASK_TOKEN_ID: Final = 4

LOCAL_DATA: Final = Path("artifacts/model-data")
LOCAL_TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")

_SPECIAL_IDS: Final = (PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_TOKEN_ID)
if sorted(_SPECIAL_IDS) != list(range(N_SPECIAL_TOKENS)):  # pragma: no cover - import-time guard
    message = f"special ids {_SPECIAL_IDS} are not the first {N_SPECIAL_TOKENS} slots"
    raise AssertionError(message)


def _load_split(split: str) -> NDArray[np.uint16]:
    from agbalu.model.data import TokenisedCorpus

    corpus = TokenisedCorpus(
        train=Path(DATA_PATH) / "train.bin",
        validation=Path(DATA_PATH) / "validation.bin",
        stats=Path(DATA_PATH) / "tokenised.stats.json",
    )
    return corpus.load_split(split)


PREVIEW_SENTENCES: Final = 3

SMOKE_ACCUMULATION: Final = 4
"""The full run accumulates 32 micro-batches per optimiser step — 1.05M tokens, 15 s. A
smoke at that batch would spend a minute per step and measure almost nothing."""

SMOKE_CHECKPOINT: Final = 10
SMOKE_VALIDATION_BATCHES: Final = 4
"""The first smoke saved 8 checkpoints (2.8 GiB to a network volume) and ran as many
validation micro-batches as training ones, in 60 seconds. Both dominated its throughput
number and neither is representative of the real run, which validates every 250 steps."""


def _reset(out_dir: Path) -> None:
    """Clear a run directory so the smoke starts at step 0.

    A smoke exists to measure throughput, and resuming its own finished checkpoint trains
    nothing while still reporting the previous run's counters. Restarting on a preemption
    retry is the right trade at 20 steps; `pretrain` never resets, so the real run keeps
    resume as its preemption cover.
    """
    if not out_dir.is_dir():
        return
    removed = sorted(p.name for p in out_dir.iterdir())
    shutil.rmtree(out_dir)
    logging.getLogger("agbalu.model").info(
        "reset %s | removed %s", out_dir, ", ".join(removed) or "nothing"
    )


def _previewer(validation: PackedDataset, train: TrainConfig) -> Previewer:
    """The same held-out sentences at every evaluation, so the previews are comparable."""
    from agbalu.model.data import iter_batches
    from agbalu.model.preview import Previewer as Build
    from agbalu.tokenizer.evaluate import load

    batch = next(iter_batches(validation, PREVIEW_SENTENCES, seed=train.seed, epoch=0))
    return Build(
        batch,
        load(Path(DATA_PATH) / LOCAL_TOKENIZER.name),
        stop_ids=frozenset({SEP_ID, PAD_ID}),
        skip_ids=frozenset({CLS_ID}),
    )


def _run(
    preset: str,
    steps: int | None,
    run_name: str,
    *,
    smoke: bool = False,
    force: bool = False,
    compile_model: bool = False,
    schedule_start: int = 0,
    resume_from: ResumeFrom = "latest",
) -> dict[str, Any]:
    import torch

    from agbalu.model.config import PRESETS, RunConfig, TrainConfig
    from agbalu.model.data import PackedDataset
    from agbalu.model.trainer import Trainer

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    log = logging.getLogger("agbalu.model")
    if torch.cuda.is_available():
        log.info("gpu=%s torch=%s", torch.cuda.get_device_name(0), torch.__version__)

    if preset not in PRESETS:
        message = f"unknown preset {preset!r}; choose from {sorted(PRESETS)}"
        raise ValueError(message)

    train = TrainConfig()
    if steps is not None:
        train = replace(train, max_steps=steps)
    if compile_model:
        train = replace(train, compile=True)
    if schedule_start:
        # A continuation holds the masking rate where the finished run left it instead of
        # letting the inverse schedule climb back to 30%.
        train = replace(
            train,
            schedule_start=schedule_start,
            mask_p_start=train.mask_p_end,
        )
    if smoke:
        train = replace(
            train,
            global_batch_size=train.local_batch_size * SMOKE_ACCUMULATION,
            save_every=SMOKE_CHECKPOINT,
            validate_every=SMOKE_CHECKPOINT,
            validation_batches=SMOKE_VALIDATION_BATCHES,
            log_every=1,
        )
    config = RunConfig(model=PRESETS[preset], train=train, name=run_name)

    def dataset(split: str) -> PackedDataset:
        return PackedDataset(
            _load_split(split),
            config.train,
            config.model.vocab_size,
            mask_token_id=MASK_TOKEN_ID,
            n_special_tokens=N_SPECIAL_TOKENS,
        )

    out_dir = Path(CHECKPOINT_PATH) / run_name
    if smoke:
        _reset(out_dir)

    validation = dataset("validation")
    trainer = Trainer(
        config,
        dataset("train"),
        validation,
        out_dir,
        previewer=_previewer(validation, config.train),
        force=force,
        resume_from=resume_from,
    )
    state = trainer.train()
    checkpoint_volume.commit()

    summary = trainer.summary
    training = summary.training_tokens_per_second
    wall = summary.wall_clock_tokens_per_second
    full = TrainConfig()
    return {
        "run": run_name,
        "preset": preset,
        "parameters": trainer.model.parameter_count(),
        "step": state.step,
        "steps_this_run": summary.steps,
        "tokens_this_run": summary.tokens,
        "masked_this_run": summary.masked,
        "tokens_seen": state.tokens_seen,
        "masked_seen": state.masked_seen,
        "best_validation_loss": round(state.best_validation_loss, 4),
        "training_tokens_per_second": None if training is None else round(training, 1),
        "wall_clock_tokens_per_second": None if wall is None else round(wall, 1),
        "full_run_hours": None
        if training is None
        else round(full.max_steps * full.tokens_per_step / training / 3600, 2),
    }


@app.function(image=image, volumes=VOLUMES, timeout=15 * 60)
def inventory() -> dict[str, list[str]]:
    """What is on the volumes. CPU-only, and the first thing to run after an upload."""
    data = Path(DATA_PATH)
    checkpoints = Path(CHECKPOINT_PATH)
    return {
        "data": sorted(p.name for p in data.iterdir()) if data.is_dir() else [],
        "checkpoints": sorted(
            str(p.relative_to(checkpoints)) for p in checkpoints.rglob("*") if p.is_file()
        )
        if checkpoints.is_dir()
        else [],
    }


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60, retries=RETRIES)
def pretrain_smoke(
    steps: int = 20, run_name: str = "smoke", compile_model: bool = False
) -> dict[str, Any]:
    return _run("kab", steps, run_name, smoke=True, compile_model=compile_model)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=TRAIN_TIMEOUT, retries=RETRIES)
def pretrain(
    preset: str = "kab",
    steps: int | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    force: bool = False,
    compile_model: bool = False,
    schedule_start: int = 0,
    resume_from: ResumeFrom = "latest",
) -> dict[str, Any]:
    """Started by `modal_app.launch`, never by a blocking local entrypoint.

    `schedule_start` continues a finished run: pass the previous `max_steps` and a larger
    `steps`, and the new span gets its own warmup, cosine and cooldown.
    """
    return _run(
        preset,
        steps,
        run_name,
        force=force,
        compile_model=compile_model,
        schedule_start=schedule_start,
        resume_from=resume_from,
    )


@app.local_entrypoint()
def upload_corpus() -> None:
    """Push the tokenised corpus and the tokenizer to the data volume."""
    if not (LOCAL_DATA / "train.bin").is_file():
        message = f"{LOCAL_DATA}/train.bin missing; run `make model-data` first"
        raise SystemExit(message)
    with data_volume.batch_upload(force=True) as batch:
        for name in ("train.bin", "validation.bin", "tokenised.stats.json"):
            batch.put_file(LOCAL_DATA / name, f"/{name}")
        batch.put_file(LOCAL_TOKENIZER, f"/{LOCAL_TOKENIZER.name}")
    print("uploaded ->", inventory.remote())


@app.local_entrypoint()
def smoke(steps: int = 20, compare_compile: bool = False) -> None:
    """End-to-end proof on real hardware before committing to the full run.

    `--compare-compile` runs it twice, eager then compiled, and prints the ratio — the
    measurement that decides whether a continuation is worth 27 hours or 18.
    """
    eager = pretrain_smoke.remote(steps=steps, run_name="smoke")
    for key, value in eager.items():
        print(f"  {key}: {value}")
    if not compare_compile:
        return

    print("\n-- compiled --")
    compiled = pretrain_smoke.remote(steps=steps, run_name="smoke-compiled", compile_model=True)
    for key, value in compiled.items():
        print(f"  {key}: {value}")

    before = eager["training_tokens_per_second"]
    after = compiled["training_tokens_per_second"]
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
        print(f"\nspeedup {after / before:.2f}x  ({before:,.0f} -> {after:,.0f} tok/s)")
    else:
        print("\nno throughput from one of the runs; nothing to compare")
