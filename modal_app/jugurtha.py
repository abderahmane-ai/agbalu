"""Continued pretraining of the base on Kabyle (task 11.6).

Two functions, and the split is deliberate. `jugurtha_pack` tokenises and packs on a CPU
container, so every failure that is knowable before training — a missing corpus, a tokenizer
that cannot be resolved, a corpus shorter than one step — happens where a GPU is not billed
for it, five times over `RETRIES`. `jugurtha_train` reads the packed blocks and does nothing
but train.

Full fine-tuning of all 1,881,825,088 text-tower parameters. AdamW over them needs 30.1 GB —
bf16 weights, bf16 gradients, fp32 master, fp32 moments — which does not fit one A10G, so the
run is `A10:2` with FSDP2 sharding gradients and optimizer state to 16.9 GiB a card. The
second GPU is what makes the method possible; the throughput is the bonus.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from agbalu.llm.bases import BASE, TEXT_TOWER_PARAMETERS
from agbalu.llm.recipe import SEQUENCE_LENGTH, SHUFFLE_SEED, WEIGHT_DECAY

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch
    from torch import nn
    from torch.optim import Optimizer

from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    JUGURTHA_PACK_CPU,
    JUGURTHA_TIMEOUT,
    JUGURTHA_VOLUMES,
    RETRIES,
    app,
    call_owner,
    checkpoint_volume,
    data_volume,
    jugurtha_image,
    llm_image,
)

GPUS: Final = 2
GPU: Final = f"A10:{GPUS}"

REMOTE_LLM: Final = "llm"
CORPUS_FILE: Final = "cpt.jsonl"
BLOCKS_FILE: Final = "cpt-blocks.u32"
BLOCKS_STATS: Final = "cpt-blocks.stats.json"

DEFAULT_RUN: Final = "jugurtha-v1"
MICRO_BATCH: Final = 2
ACCUMULATION: Final = 32
"""2 x 1024 x 32 x 2 GPUs = 131,072 tokens an optimizer step. Micro-batch 2 halves the
transient logit tensor for the 248,320-token head (1.9 GiB vs 3.8 GiB), keeping peak VRAM
well within the 22.06 GiB A10G limit while preserving the exact 131k token step geometry."""

TOKENS_PER_STEP: Final = MICRO_BATCH * SEQUENCE_LENGTH * ACCUMULATION * GPUS

ENCODE_BATCH: Final = 1000

LOG_EVERY: Final = 50
CHECKPOINT_EVERY: Final = 500
COMPILE: Final = True
"""Wrap every decoder layer in `torch.compile` before FSDP2 sharding. The compile graph is
per-layer rather than whole-model so FSDP2's per-layer `fully_shard` sees a static graph
boundary and the Triton kernel cache is warm from layer 0 on the first forward."""

MIN_TOKENS_PER_SECOND: Final = 2500.0
"""Floor raised to match the compiled baseline. Two A10s at ~30% MFU carry ~6,500–7,800
tok/s compiled; half of that still indicates a broken kernel or data path, not variance."""

log: Final = logging.getLogger("agbalu.jugurtha")


class Settings(TypedDict):
    """What each FSDP rank needs. A TypedDict because it crosses `mp.spawn`."""

    world: int
    run: str
    epochs: int
    steps: int


class PackReport(TypedDict):
    blocks: int
    tokens: int
    documents: int
    sequence_length: int
    base: str


class TrainReport(TypedDict):
    run: str
    resumed: bool
    epochs_target: int
    steps_this_run: int
    tokens_this_run: int
    steps_total: int
    tokens_total: int
    final_loss: float | None
    tokens_per_second: float | None
    peak_gib: float
    refusal: str | None


def blocks_path() -> Path:
    return Path(DATA_PATH) / REMOTE_LLM / BLOCKS_FILE


def run_directory(run: str) -> Path:
    return Path(CHECKPOINT_PATH) / "llm" / run


@app.function(
    image=llm_image,
    cpu=JUGURTHA_PACK_CPU,
    volumes=JUGURTHA_VOLUMES,
    timeout=JUGURTHA_TIMEOUT,
)
def jugurtha_pack(force: bool = False) -> PackReport:
    """Tokenise the corpus once and write fixed-length blocks the trainer memory-maps.

    Packed here rather than in the training container because the tokenizer is a CPU job of
    5,254,022 documents: paying for it on an A10, once per epoch and again on every
    preemption, is the same defect as building a batch row by row.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    import numpy as np
    from transformers import AutoTokenizer

    from agbalu.llm.recipe import documents, pack

    target = blocks_path()
    stats = target.with_name(BLOCKS_STATS)
    if target.is_file() and stats.is_file() and not force:
        existing: PackReport = json.loads(stats.read_text(encoding="utf-8"))
        log.info("blocks already packed | %d blocks | force=True rebuilds", existing["blocks"])
        return existing

    corpus = Path(DATA_PATH) / REMOTE_LLM / CORPUS_FILE
    if not corpus.is_file():
        message = f"{corpus} absent; run `make llm TASK=mixture` then `make modal-upload TASK=llm`"
        raise FileNotFoundError(message)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    separator = tokenizer.eos_token_id
    if separator is None:
        message = f"{BASE} has no eos token to separate packed documents with"
        raise RuntimeError(message)

    def encoded() -> Iterator[list[int]]:
        batch: list[str] = []
        for text in documents(corpus):
            batch.append(text)
            if len(batch) == ENCODE_BATCH:
                yield from tokenizer(batch, add_special_tokens=False)["input_ids"]
                batch = []
        if batch:
            yield from tokenizer(batch, add_special_tokens=False)["input_ids"]

    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = 0
    with target.open("wb") as handle:
        for block in pack(encoded(), SEQUENCE_LENGTH, int(separator)):
            handle.write(np.asarray(block, dtype=np.uint32).tobytes())
            written += 1
            if written % 50_000 == 0:
                log.info("packed %d blocks", written)
    seen = written * SEQUENCE_LENGTH

    if written < 1:
        message = f"{corpus} yielded no full block of {SEQUENCE_LENGTH} tokens"
        raise RuntimeError(message)

    report: PackReport = {
        "blocks": written,
        "tokens": seen,
        "documents": sum(1 for _ in documents(corpus)),
        "sequence_length": SEQUENCE_LENGTH,
        "base": BASE,
    }
    stats.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    log.info("packed %d blocks | %d tokens | -> %s", written, seen, target)
    return report


def order(count: int, seed: int) -> npt.NDArray[np.int64]:
    """A fixed permutation of the block indices.

    The corpus is written source by source and packing preserves that, so without this every
    batch would come from one source and the loss would track the source rather than the
    model. Fixed rather than drawn so a resumed run sees the same order it left.
    """
    import numpy as np

    generator = np.random.default_rng(seed)
    return generator.permutation(count)


def _setup(rank: int, world: int) -> tuple[nn.Module, Optimizer, torch.device]:
    """The sharded model, its optimizer, and this rank's device."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoModelForCausalLM

    from agbalu.llm.recipe import PEAK_LR

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    # bf16 parameters with fp32 reduction: the all-reduce is where a bf16 sum loses the
    # small updates that a 2e-5 schedule is made of.
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95), fused=True
    )
    return model, optimizer, device


def _step(
    model: nn.Module,
    optimizer: Optimizer,
    batch_rows: Sequence[npt.NDArray[np.int64]],
    blocks: npt.NDArray[np.uint32],
    device: torch.device,
) -> tuple[float, float]:
    """One optimizer step over `ACCUMULATION` micro-batches. Returns loss and gradient norm."""
    import numpy as np
    import torch

    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for rows in batch_rows:
        ids = torch.from_numpy(blocks[rows].astype(np.int64)).to(device, non_blocking=True)
        out = model(input_ids=ids, labels=ids)
        (out.loss / ACCUMULATION).backward()
        loss_value = float(out.loss.detach())
    # Read before the step, not after: a non-finite check that runs afterwards reports the
    # damage it caused rather than preventing it.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(norm):
        optimizer.step()
    return loss_value, float(norm)


def _rows(
    step: int, rank: int, world: int, permutation: npt.NDArray[np.int64], count: int
) -> list[npt.NDArray[np.int64]]:
    """This rank's micro-batches for one optimizer step, disjoint from every other rank's."""
    batches = []
    for micro in range(ACCUMULATION):
        index = (step * ACCUMULATION + micro) * MICRO_BATCH * world + rank * MICRO_BATCH
        batches.append(permutation[[(index + i) % count for i in range(MICRO_BATCH)]])
    return batches


def _worker(rank: int, settings: Settings) -> None:
    """One FSDP rank. Nothing here touches CUDA before `spawn` returns in the parent."""
    import numpy as np
    import torch
    import torch.distributed as dist

    from agbalu.llm.recipe import learning_rate

    world = settings["world"]
    run = settings["run"]
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{rank}] %(message)s",
        force=True,
    )

    blocks = np.memmap(blocks_path(), dtype=np.uint32, mode="r").reshape(-1, SEQUENCE_LENGTH)
    permutation = order(len(blocks), SHUFFLE_SEED)
    model, optimizer, device = _setup(rank, world)

    directory = run_directory(run)
    start_step = _load(directory, model, optimizer)
    total = (len(blocks) * settings["epochs"]) // (MICRO_BATCH * ACCUMULATION * world)
    cap = settings["steps"]
    if cap:
        total = min(total, start_step + cap)

    if start_step >= total:
        if rank == 0:
            log.info("already at %d of %d steps; nothing to train", start_step, total)
        dist.destroy_process_group()
        return

    if rank == 0:
        log.info(
            "run=%s resumed=%s steps %d -> %d | %d tokens/step | %d parameters",
            run,
            start_step > 0,
            start_step,
            total,
            TOKENS_PER_STEP,
            TEXT_TOWER_PARAMETERS,
        )

    torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    tokens = 0
    loss_value = 0.0
    model.train()

    for step in range(start_step, total):
        rate = learning_rate(step, total)
        for group in optimizer.param_groups:
            group["lr"] = rate
        loss_value, norm = _step(
            model, optimizer, _rows(step, rank, world, permutation, len(blocks)), blocks, device
        )
        tokens += TOKENS_PER_STEP
        first_or_last = step == start_step or step + 1 == total
        if rank == 0 and (first_or_last or (step + 1) % LOG_EVERY == 0):
            log.info(
                "step %d/%d loss %.4f lr %.2e grad %.2f %.0f tok/s peak %.2f GiB",
                step + 1,
                total,
                loss_value,
                rate,
                norm,
                tokens / max(time.monotonic() - began, 1e-9),
                torch.cuda.max_memory_allocated(device) / 2**30,
            )
        if (step + 1) % CHECKPOINT_EVERY == 0 or step + 1 == total:
            _save(directory, model, optimizer, step + 1, rank)

    dist.barrier()
    if rank == 0:
        _summarise(directory, run, start_step, total, tokens, loss_value, began, device)
    dist.destroy_process_group()


def _summarise(
    directory: Path,
    run: str,
    start_step: int,
    total: int,
    tokens: int,
    loss_value: float,
    began: float,
    device: torch.device,
) -> None:
    import torch

    elapsed = time.monotonic() - began
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run": run,
                "steps_this_run": total - start_step,
                "tokens_this_run": tokens,
                "steps_total": total,
                "final_loss": round(loss_value, 6),
                "tokens_per_second": round(tokens / max(elapsed, 1e-9), 1),
                "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()


def _save(directory: Path, model: object, optimizer: object, step: int, rank: int) -> None:
    """Sharded, atomic, and committed before the step counter is believed."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    directory.mkdir(parents=True, exist_ok=True)
    weights, states = get_state_dict(model, optimizer)
    dcp.save({"model": weights, "optimizer": states}, checkpoint_id=str(directory / "state"))
    if rank == 0:
        (directory / "step.json").write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
        checkpoint_volume.commit()


def _load(directory: Path, model: nn.Module, optimizer: Optimizer) -> int:
    """The step to resume from, or 0. A checkpoint without its counter is not resumed."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    marker = directory / "step.json"
    if not (directory / "state").is_dir() or not marker.is_file():
        return 0
    weights, states = get_state_dict(model, optimizer)
    payload = {"model": weights, "optimizer": states}
    dcp.load(payload, checkpoint_id=str(directory / "state"))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=payload["model"],
        optim_state_dict=payload["optimizer"],
    )
    return int(json.loads(marker.read_text(encoding="utf-8"))["step"])


@app.function(
    image=jugurtha_image,
    gpu=GPU,
    volumes=JUGURTHA_VOLUMES,
    timeout=JUGURTHA_TIMEOUT,
    retries=RETRIES,
)
def jugurtha_train(
    epochs: int = 1, steps: int = 0, run: str = DEFAULT_RUN, force: bool = False
) -> TrainReport:
    """One epoch of full continued pretraining, resumable.

    `steps` caps the run for a smoke and changes nothing else, so the smoke exercises the
    path the real run takes rather than a shorter one beside it. `epochs` is the target the
    schedule is drawn against, not a number of passes to add: relaunching at a higher value
    continues the same cosine rather than restarting it.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    import torch.multiprocessing as mp

    from agbalu.model import lock

    if not blocks_path().is_file():
        message = "packed blocks absent; run `make modal-jugurtha TASK=pack` first"
        return _refusal(run, message)

    directory = run_directory(run)
    directory.mkdir(parents=True, exist_ok=True)
    owner = call_owner()
    try:
        lock.acquire(directory, owner, force=force)
    except lock.RunLockedError as error:
        return _refusal(run, str(error))

    try:
        mp.spawn(
            _worker,
            args=(Settings(world=GPUS, run=run, epochs=epochs, steps=steps),),
            nprocs=GPUS,
            join=True,
        )
    finally:
        lock.release(directory, owner)

    checkpoint_volume.reload()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rate = summary["tokens_per_second"]
    if rate is not None and rate < MIN_TOKENS_PER_SECOND and not steps:
        log.warning("%.0f tok/s is below the %.0f floor", rate, MIN_TOKENS_PER_SECOND)
    return {
        "run": run,
        "resumed": summary["steps_total"] > summary["steps_this_run"],
        "epochs_target": epochs,
        "steps_this_run": summary["steps_this_run"],
        "tokens_this_run": summary["tokens_this_run"],
        "steps_total": summary["steps_total"],
        "tokens_total": summary["steps_total"] * TOKENS_PER_STEP,
        "final_loss": summary["final_loss"],
        "tokens_per_second": rate,
        "peak_gib": summary["peak_gib"],
        "refusal": None,
    }


def _refusal(run: str, message: str) -> TrainReport:
    """A knowable failure returned rather than raised.

    `RETRIES` is five and cannot tell a bug from a preemption, so raising here buys the same
    diagnosis five times on a two-GPU container.
    """
    log.error("refusing to train: %s", message)
    return {
        "run": run,
        "resumed": False,
        "epochs_target": 0,
        "steps_this_run": 0,
        "tokens_this_run": 0,
        "steps_total": 0,
        "tokens_total": 0,
        "final_loss": None,
        "tokens_per_second": None,
        "peak_gib": 0.0,
        "refusal": message,
    }


@app.local_entrypoint()
def pack(force: bool = False) -> None:
    """Build the packed blocks on the volume. CPU, and nothing waits on the GPU for it."""
    report = jugurtha_pack.remote(force)
    print(f"{report['blocks']:,} blocks of {report['sequence_length']}")
    print(f"{report['tokens']:,} tokens from {report['documents']:,} documents")
    print(f"tokenised by {report['base']}")
