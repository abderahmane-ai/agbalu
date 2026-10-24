"""Continued pretraining and supervised fine-tuning on one GPU (tasks 11.18 and 11.23).

For CPT the trainer reads `pack`'s blocks in file order: the shuffle happened there, so a
step is a fixed slice of the file and a resumed run reads exactly the blocks an uninterrupted
one would. Each forward takes `micro_blocks` blocks as one variable-length row, and the loss
of a step is the summed loss of its rows divided by the step's scored tokens, so every target
token weighs the same however the documents fell into rows.

For SFT the rows come from `agbalu.llm.sft_rows`, already packed and masked, in an order
seeded per epoch; the step is the same function, scoring only the supervised tokens. The best
checkpoint is the one with the lowest loss on the SFT validation rows, and held-out bits per
character are logged beside it as a drift diagnostic.

A run directory holds `config.json`, `log.jsonl`, up to `keep` checkpoints (`step-NNNNNNN/`:
weights, `optimizer.pt`, `rng.pt`, `state.json`, written to a hidden partial directory and
renamed), `best/` and `summary.json`. Starting a run whose directory already has a
checkpoint resumes it; a config that differs from the stored one is refused.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import socket
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType, TracebackType
from typing import Final, Literal, Self

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agbalu.llm.bases import TEXT_TOWER_PARAMETERS
from agbalu.llm.evaluate import (
    HELDOUT_LANGUAGES,
    Scorer,
    Tokenizer,
    greedy,
    held_out,
    summed_nll,
)
from agbalu.llm.forward import LanguageModel, hidden_states, to_device
from agbalu.llm.holdout import read as read_heldout
from agbalu.llm.loss import logit_softcap, sequence_loss
from agbalu.llm.packing import BLOCK_AXES, PackedRow, pack_row
from agbalu.llm.progress import Heartbeat, Tracker, progress, stage
from agbalu.llm.provenance import code_stamp, config_hash, file_digest
from agbalu.llm.recipe import (
    BETAS,
    BLOCKS_PER_STEP,
    GRADIENT_CLIP,
    LARGE_MODEL_PARAMETERS,
    MICRO_BLOCKS,
    PEAK_LR,
    SFT_EPOCHS,
    SFT_SEED,
    SFT_WARMUP_FRACTION,
    SFT_WEIGHT_DECAY,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    learning_rate,
)
from agbalu.llm.sft_rows import Rows

Optimizer = Literal["adamw8bit", "adamw"]
Objective = Literal["cpt", "sft"]

RUNS_ROOT: Final = Path("artifacts/runs")
HELDOUT_DIR: Final = Path("data/processed/llm")

PEAK_FLOPS: Final = 250e12
"""bf16 dense matmul throughput measured on the HPC's MIG 3g.47gb slice, 2026-10-14
(`~/.claude/standards/hpc.md` §1). Model-FLOP utilisation is reported against it."""

MAX_SKIPPED: Final = 10
"""Steps with a non-finite gradient norm a run may skip before it stops."""

DRIFT_GUARD: Final = 1.05
"""A checkpoint is eligible as best only while English and French held-out bits per
character stay within 5% of the run's own step-0 values."""

TORCH_SEED: Final = 20261014
"""Seeds torch's generators, which stochastic rounding in the 8-bit optimizer draws from."""

MATRIX_AXES: Final = 2
"""Parameters with at least this many axes take weight decay."""

PREFETCH: Final = 4
ETA_WINDOW: Final = 20
"""Recent steps the time-left projection is taken over."""

PROBES: Final = (
    "Tamurt n Leqbayel",
    "Deg useggas-nni, imezdaɣ n taddart",
    "eng_Latn: The children went to school early this morning.\nkab_Latn:",
)
PROBE_TOKENS: Final = 48

STATE_FILE: Final = "state.json"
OPTIMIZER_FILE: Final = "optimizer.pt"
RNG_FILE: Final = "rng.pt"
CHECKPOINT_PREFIX: Final = "step-"


class TrainError(Exception):
    """A run cannot start, resume or continue as configured."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    run: str
    model: str
    blocks: Path
    optimizer: Optimizer = "adamw8bit"
    peak_lr: float = PEAK_LR
    blocks_per_step: int = BLOCKS_PER_STEP
    micro_blocks: int = MICRO_BLOCKS
    total_steps: int | None = None
    """The schedule's length; by default every whole step the blocks file holds."""

    stop_after: int | None = None
    """End this invocation at this step with a checkpoint. Not part of the run's identity."""

    checkpoint_every: int = 20
    eval_every: int = 50
    heldout_limit: int | None = None
    probe: bool = True
    keep: int = 2
    workers: int = 8
    runs_root: Path = RUNS_ROOT
    heldout_dir: Path = HELDOUT_DIR
    objective: Objective = "cpt"
    validation: Path | None = None
    """SFT validation rows, the loss the best checkpoint is chosen on; SFT only."""

    expected_parameters: int | None = None
    """The text-tower count a model loaded from disk must have; a published repository is
    held to `TEXT_TOWER_PARAMETERS` instead."""

    def __post_init__(self) -> None:
        if self.objective == "sft" and self.validation is None:
            message = f"{self.run}: an SFT run needs validation rows"
            raise TrainError(message)
        if self.micro_blocks < 1 or self.blocks_per_step % self.micro_blocks:
            message = f"{self.blocks_per_step} blocks a step do not split into {self.micro_blocks}"
            raise TrainError(message)
        if self.keep < 1 or self.checkpoint_every < 1 or self.eval_every < 0:
            message = (
                f"keep {self.keep}, checkpoint_every {self.checkpoint_every} and eval_every "
                f"{self.eval_every} must be positive (eval_every may be 0)"
            )
            raise TrainError(message)

    @property
    def micros_per_step(self) -> int:
        return self.blocks_per_step // self.micro_blocks

    @property
    def directory(self) -> Path:
        return self.runs_root / self.run

    def identity(self) -> dict[str, object]:
        """What a resumed run must share with the run it continues.

        A CPT run's identity omits the fields added for SFT, which did not exist when
        `jugurtha-4b-v1` stored its config: adding them must not refuse its resume.
        """
        values = asdict(self)
        for transient in ("stop_after", "workers", "runs_root", "heldout_dir"):
            values.pop(transient)
        values["blocks"] = str(self.blocks)
        if self.objective == "cpt":
            for added in ("objective", "validation", "expected_parameters"):
                values.pop(added)
        else:
            values["validation"] = str(self.validation)
        return values

    @property
    def warmup_fraction(self) -> float:
        return WARMUP_FRACTION if self.objective == "cpt" else SFT_WARMUP_FRACTION

    @property
    def weight_decay(self) -> float:
        return WEIGHT_DECAY if self.objective == "cpt" else SFT_WEIGHT_DECAY


@dataclass(slots=True)
class RunState:
    step: int = 0
    tokens: int = 0
    targets: int = 0
    skipped: int = 0
    reference: dict[str, float] = field(default_factory=dict)
    """Held-out bits per character at step 0, the drift guard's baseline."""

    best_bpc: float | None = None
    best_step: int | None = None
    best_validation: float | None = None
    """SFT: the lowest validation loss so far, nats per supervised token."""

    @classmethod
    def read(cls, path: Path) -> RunState:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            message = f"{path} is not a JSON object"
            raise TrainError(message)
        try:
            return cls(**payload)
        except TypeError as error:
            message = f"{path} does not match RunState: {error}"
            raise TrainError(message) from error


class EventLog:
    """One JSON object per line, to `log.jsonl` and to stdout, which the launcher captures."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: str, **values: object) -> None:
        line = json.dumps({"event": event, "time": round(time.time(), 3), **values}, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


BOOT_ID: Final = Path("/proc/sys/kernel/random/boot_id")


def boot_id() -> str:
    """This boot's id on Linux, empty elsewhere. A pod restart reuses process ids, so a lock
    is matched on the boot as well as the pid."""
    return BOOT_ID.read_text(encoding="utf-8").strip() if BOOT_ID.is_file() else ""


class RunLock:
    """One process per run directory. A lock left by a dead process, or written before the
    pod last restarted, is taken."""

    def __init__(self, path: Path, log: EventLog) -> None:
        self.path = path
        self.log = log

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "boot": boot_id()})
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._clear_stale()
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            return self
        message = f"could not take {self.path}"
        raise TrainError(message)

    def _clear_stale(self) -> None:
        holder = json.loads(self.path.read_text(encoding="utf-8"))
        pid, host = int(holder["pid"]), str(holder["host"])
        if host != socket.gethostname():
            message = f"{self.path} is held by pid {pid} on {host}"
            raise TrainError(message)
        if holder.get("boot") != boot_id():
            self.log.write("stale_lock", pid=pid, reason="written before a restart")
            self.path.unlink()
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.log.write("stale_lock", pid=pid)
            self.path.unlink()
            return
        message = f"{self.path} is held by running pid {pid}"
        raise TrainError(message)

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.path.unlink(missing_ok=True)


class StopRequest:
    """Set by SIGTERM or SIGINT; the loop checkpoints at the next step boundary."""

    def __init__(self) -> None:
        self.signal: int | None = None

    def install(self) -> None:
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, self.handle)

    def handle(self, number: int, frame: FrameType | None) -> None:
        del frame
        self.signal = number


class Blocks(Dataset[PackedRow]):
    """Micro-batches of `pack`'s blocks as packed rows, the memmap opened once per worker."""

    def __init__(self, path: Path, micro_blocks: int, eos: int, start: int, stop: int) -> None:
        self.path = path
        self.micro_blocks = micro_blocks
        self.eos = eos
        self.start = start
        self.stop = stop
        self.array: np.memmap[tuple[int, ...], np.dtype[np.uint32]] | None = None

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int) -> PackedRow:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r")
        first = (self.start + index) * self.micro_blocks
        return pack_row(np.asarray(self.array[first : first + self.micro_blocks]), self.eos)


def block_shape(path: Path) -> tuple[int, int]:
    """`(blocks, length)` from the file's header, without reading the data."""
    if not path.is_file():
        message = f"{path} not found; build it with `make data TASK=pack`"
        raise TrainError(message)
    array = np.load(path, mmap_mode="r")
    if array.ndim != BLOCK_AXES or array.dtype != np.uint32:
        message = f"{path}: expected (blocks, length) uint32, got {array.shape} {array.dtype}"
        raise TrainError(message)
    return int(array.shape[0]), int(array.shape[1])


def stream(config: RunConfig, eos: int, first_step: int, total: int) -> Iterator[PackedRow]:
    """Every micro-batch from `first_step` to the end of the schedule, in file order."""
    dataset = Blocks(
        config.blocks,
        config.micro_blocks,
        eos,
        first_step * config.micros_per_step,
        total * config.micros_per_step,
    )
    loader: DataLoader[PackedRow] = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=config.workers,
        prefetch_factor=PREFETCH if config.workers else None,
        persistent_workers=config.workers > 0,
    )
    yield from loader


def sft_order(rows: int, epochs: int = SFT_EPOCHS, seed: int = SFT_SEED) -> list[int]:
    """Row indices for the whole schedule, each epoch its own seeded permutation."""
    order: list[int] = []
    for epoch in range(epochs):
        order.extend(int(i) for i in np.random.default_rng(seed + epoch).permutation(rows))
    return order


def sft_stream(
    rows: Rows, micros_per_step: int, first_step: int, total: int
) -> Iterator[PackedRow]:
    """The rows from `first_step` to the end of the schedule; a resume reads the same rows."""
    order = sft_order(len(rows))
    for index in order[first_step * micros_per_step : total * micros_per_step]:
        yield rows.packed(index)


def parameter_groups(model: LanguageModel, weight_decay: float) -> list[dict[str, object]]:
    """Weight decay on matrices only; norms, biases and the delta-rule scalars are exempt."""
    decay: list[torch.nn.Parameter] = []
    exempt: list[torch.nn.Parameter] = []
    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= MATRIX_AXES else exempt).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": exempt, "weight_decay": 0.0},
    ]


def build_optimizer(model: LanguageModel, config: RunConfig) -> torch.optim.Optimizer:
    groups = parameter_groups(model, config.weight_decay)
    if config.optimizer == "adamw8bit":
        from torchao.optim import AdamW8bit

        optimizer: torch.optim.Optimizer = AdamW8bit(
            groups, lr=config.peak_lr, betas=BETAS, bf16_stochastic_round=True
        )
        return optimizer
    return torch.optim.AdamW(groups, lr=config.peak_lr, betas=BETAS)


def weight_dtype(optimizer: Optimizer) -> torch.dtype:
    """bf16 weights with the 8-bit arm, fp32 master weights with the reference arm."""
    return torch.bfloat16 if optimizer == "adamw8bit" else torch.float32


def checkpoints(directory: Path) -> list[Path]:
    """Complete checkpoints, oldest first. A partial one is hidden and never listed."""
    found = [
        path
        for path in directory.glob(f"{CHECKPOINT_PREFIX}*")
        if path.is_dir() and (path / STATE_FILE).is_file()
    ]
    return sorted(found, key=lambda path: int(path.name.removeprefix(CHECKPOINT_PREFIX)))


def write_json(path: Path, payload: object) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    partial.replace(path)


def save_checkpoint(
    directory: Path,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    state: RunState,
    keep: int,
) -> Path:
    """Weights, optimizer, generators and state, all or nothing; the oldest beyond `keep` go."""
    with stage("save checkpoint", step=state.step):
        return write_checkpoint(directory, model, optimizer, state, keep)


def write_checkpoint(
    directory: Path,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    state: RunState,
    keep: int,
) -> Path:
    final = directory / f"{CHECKPOINT_PREFIX}{state.step:07d}"
    partial = directory / f".{final.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    model.save_pretrained(partial)
    torch.save(optimizer.state_dict(), partial / OPTIMIZER_FILE)
    generators: dict[str, object] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        generators["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(generators, partial / RNG_FILE)
    write_json(partial / STATE_FILE, asdict(state))
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    for old in checkpoints(directory)[:-keep]:
        shutil.rmtree(old)
    return final


def restore(path: Path, optimizer: torch.optim.Optimizer) -> RunState:
    """Optimizer state and generators from a checkpoint this trainer wrote."""
    optimizer.load_state_dict(torch.load(path / OPTIMIZER_FILE, weights_only=False))
    generators = torch.load(path / RNG_FILE, weights_only=False)
    torch.set_rng_state(generators["cpu"])
    if "cuda" in generators and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(generators["cuda"])
    return RunState.read(path / STATE_FILE)


def save_best(directory: Path, model: LanguageModel, state: RunState) -> None:
    chosen_on = (
        {"kab_bpc": state.best_bpc}
        if state.best_validation is None
        else {"validation_loss": state.best_validation}
    )
    progress("save_best", step=state.step, **chosen_on)
    partial = directory / ".best.partial"
    if partial.exists():
        shutil.rmtree(partial)
    model.save_pretrained(partial)
    write_json(partial / "best.json", {"step": state.step, **chosen_on})
    final = directory / "best"
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)


def guard_passes(scores: dict[str, float], reference: dict[str, float]) -> bool:
    """English and French within `DRIFT_GUARD` of their step-0 values."""
    return all(scores[language] <= reference[language] * DRIFT_GUARD for language in ("eng", "fra"))


def heldout_bpc(
    scorer: Scorer, tokenizer: Tokenizer, limit: int | None, root: Path = HELDOUT_DIR
) -> dict[str, float]:
    return {
        language: held_out(
            scorer,
            tokenizer,
            read_heldout(root / f"heldout-{language}.jsonl")[:limit],
            language=language,
        ).bits_per_character
        for language in HELDOUT_LANGUAGES
    }


def check_identity(config: RunConfig, log: EventLog) -> None:
    """Write the run's identity on a fresh start; refuse a resume that changes it."""
    path = config.directory / "config.json"
    identity = config.identity()
    if not path.is_file():
        write_json(path, identity)
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    keys = identity.keys() | stored.keys()
    differs = sorted(key for key in keys if identity.get(key) != stored.get(key))
    if differs:
        message = f"{config.run}: config differs from the stored run in {differs}"
        raise TrainError(message)
    log.write("config_matches", hash=config_hash(identity))


def count_parameters(model: LanguageModel, config: RunConfig) -> int:
    """Trainable parameters, held to the published count where the repository has one, or
    to `expected_parameters` for a model on disk."""
    total = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    expected = TEXT_TOWER_PARAMETERS.get(config.model, config.expected_parameters)
    if expected is not None and total != expected:
        message = f"{config.model}: {total:,} trainable parameters, the text tower has {expected:,}"
        raise TrainError(message)
    return total


def validation_loss(scorer: Scorer, rows: Rows) -> float:
    """Nats per supervised token over every validation row."""
    nats, targets = summed_nll(scorer, [rows.packed(index) for index in range(len(rows))])
    if targets < 1:
        message = "the validation rows supervise no token"
        raise TrainError(message)
    return nats / targets


def peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


class GradientAccumulator:
    """Accumulates gradients in host memory (unpinned to spare BAR1) to keep VRAM below 38GB."""

    def __init__(self, model: LanguageModel, device: torch.device) -> None:
        self.device = device
        self.params: list[torch.nn.Parameter] = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        self.cpu_grads: dict[torch.nn.Parameter, torch.Tensor] = {
            p: torch.zeros_like(p, device="cpu", pin_memory=False) for p in self.params
        }
        for p in self.params:
            p.register_post_accumulate_grad_hook(self._hook)  # type: ignore[no-untyped-call]

    def _hook(self, param: torch.nn.Parameter) -> None:
        if param.grad is not None:
            self.cpu_grads[param].add_(param.grad.to("cpu"))
            param.grad = None

    def norm_and_clip(self, clip_norm: float) -> tuple[float, float, bool]:
        """Global gradient norm and scaling coefficient."""
        torch.cuda.synchronize()
        total_sq = 0.0
        for g in self.cpu_grads.values():
            total_sq += float(torch.sum(g.float() ** 2).item())
        norm = math.sqrt(total_sq)
        finite = math.isfinite(norm)
        coef = min(1.0, clip_norm / (norm + 1e-6)) if finite else 0.0
        return norm, coef, finite

    def step(self, optimizer: torch.optim.Optimizer, clip_coef: float) -> None:
        """Stream gradients parameter-by-parameter into optimizer to keep peak VRAM minimal."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                g_cpu = self.cpu_grads[p]
                scaled = g_cpu if clip_coef >= 1.0 else g_cpu * clip_coef
                p.grad = scaled.to(self.device, non_blocking=True)
                optimizer.step()
                p.grad = None
                g_cpu.zero_()
        torch.cuda.synchronize()

    def zero_grad(self) -> None:
        for g in self.cpu_grads.values():
            g.zero_()
        for p in self.params:
            p.grad = None


@dataclass(slots=True)
class Session:
    """Everything one invocation of the loop works on."""

    config: RunConfig
    model: LanguageModel
    optimizer: torch.optim.Optimizer
    state: RunState
    tokenizer: Tokenizer
    device: torch.device
    total: int
    parameters: int
    log: EventLog
    stop: StopRequest | None
    rows: Rows | None = None
    validation: Rows | None = None
    accumulator: GradientAccumulator | None = None
    step_seconds: list[float] = field(default_factory=list)
    step_tokens: list[int] = field(default_factory=list)
    wait_seconds: list[float] = field(default_factory=list)
    tokens_this_run: int = 0
    micro: int = 0
    phase: str = "starting"

    @property
    def autocast(self) -> bool:
        """fp32 master weights compute in bf16; bf16 weights need no autocast."""
        return self.device.type == "cuda" and weight_dtype(self.config.optimizer) == torch.float32


def summary(session: Session, ended: str) -> dict[str, object]:
    """Throughput of this invocation, the first step excluded as warm-up when there are more."""
    config, state = session.config, session.state
    first = 1 if len(session.step_seconds) > 1 else 0
    steady = session.step_seconds[first:]
    tokens_per_second = (
        math.fsum(session.step_tokens[first:]) / math.fsum(steady) if steady else None
    )
    return {
        "run": config.run,
        "ended": ended,
        "step": state.step,
        "steps_this_run": len(session.step_seconds),
        "tokens_total": state.tokens,
        "tokens_this_run": session.tokens_this_run,
        "skipped": state.skipped,
        "best_step": state.best_step,
        "best_kab_bpc": state.best_bpc,
        "best_validation_loss": state.best_validation,
        "step_seconds_median": statistics.median(steady) if steady else None,
        "tokens_per_second": tokens_per_second,
        "mfu": (
            6 * session.parameters * tokens_per_second / PEAK_FLOPS if tokens_per_second else None
        ),
        "mfu_counts": "6N per token; no attention, no recomputation",
        "data_wait_share": (
            math.fsum(session.wait_seconds) / math.fsum(session.step_seconds)
            if session.step_seconds
            else None
        ),
        "peak_memory_bytes": peak_memory(session.device),
        "parameters": session.parameters,
    }


def evaluate_now(session: Session) -> None:
    """Held-out bits per character, the best checkpoint under the drift guard, the probe."""
    with stage("in-training evaluation", step=session.state.step):
        evaluate_model(session)


def evaluate_model(session: Session) -> None:
    config, state, model = session.config, session.state, session.model
    model.eval()
    scorer = Scorer.of(model, session.tokenizer.eos, session.device)
    scores = heldout_bpc(scorer, session.tokenizer, config.heldout_limit, config.heldout_dir)
    if not state.reference:
        state.reference = scores
    if session.validation is not None:
        choose_on_validation(session, scorer, scores)
    else:
        eligible = guard_passes(scores, state.reference)
        session.log.write(
            "eval", step=state.step, bpc=scores, reference=state.reference, eligible=eligible
        )
        if eligible and (state.best_bpc is None or scores["kab"] < state.best_bpc):
            state.best_bpc, state.best_step = scores["kab"], state.step
            save_best(config.directory, model, state)
            session.log.write("best", step=state.step, kab_bpc=scores["kab"])
    if config.probe:
        budgets = [PROBE_TOKENS] * len(PROBES)
        texts = greedy(model, session.tokenizer, PROBES, budgets, device=session.device)
        session.log.write("probe", step=state.step, texts=dict(zip(PROBES, texts, strict=True)))
    model.train()


def scheduled_rate(session: Session) -> float:
    config = session.config
    return learning_rate(
        session.state.step,
        session.total,
        peak=config.peak_lr,
        warmup_fraction=config.warmup_fraction,
    )


def choose_on_validation(session: Session, scorer: Scorer, bpc: dict[str, float]) -> None:
    """SFT: the lowest validation loss is the best checkpoint; bits per character are logged
    beside it as a drift diagnostic and choose nothing."""
    state, validation = session.state, session.validation
    if validation is None:
        message = "no validation rows to choose the best checkpoint on"
        raise TrainError(message)
    loss = validation_loss(scorer, validation)
    session.log.write(
        "eval", step=state.step, validation_loss=loss, bpc=bpc, reference=state.reference
    )
    if state.best_validation is None or loss < state.best_validation:
        state.best_validation, state.best_step = loss, state.step
        save_best(session.config.directory, session.model, state)
        session.log.write("best", step=state.step, validation_loss=loss)


def step(session: Session, batch: list[PackedRow]) -> tuple[float, float, bool]:
    """One optimizer step over `batch`: mean loss per target, gradient norm, whether taken."""
    model, device = session.model, session.device
    softcap = logit_softcap(model.config)
    step_targets = sum(row.targets for row in batch)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    tracker = Tracker(f"step {session.state.step + 1} micro-batches", len(batch))
    for micro, packed in enumerate(batch, start=1):
        session.micro = micro
        row = to_device(packed, device, pinned=device.type == "cuda")
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=session.autocast):
            hidden = hidden_states(model.model, row)
        classifier = model.lm_head.weight
        if device.type == "cuda":
            hidden, classifier = hidden.bfloat16(), classifier.bfloat16()
        loss = sequence_loss(hidden, classifier, row.labels, softcap=softcap, reduction="sum")
        torch.autograd.backward(loss / step_targets)
        total_loss += loss.detach().float() / step_targets
        tracker.advance()
    if session.accumulator is not None:
        norm_val, clip_coef, finite = session.accumulator.norm_and_clip(GRADIENT_CLIP)
        if finite:
            lr = scheduled_rate(session)
            for group in session.optimizer.param_groups:
                current = group["lr"]
                if isinstance(current, torch.Tensor):
                    current.fill_(lr)
                else:
                    group["lr"] = lr
            session.accumulator.step(session.optimizer, clip_coef)
        else:
            session.accumulator.zero_grad()
        return float(total_loss.item()), norm_val, finite

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    norm_tensor = torch.nn.utils.clip_grad_norm_(params, GRADIENT_CLIP)
    finite = bool(torch.isfinite(norm_tensor).item())
    if finite:
        lr = scheduled_rate(session)
        for group in session.optimizer.param_groups:
            current = group["lr"]
            if isinstance(current, torch.Tensor):
                current.fill_(lr)
            else:
                group["lr"] = lr
        session.optimizer.step()
    session.optimizer.zero_grad(set_to_none=True)
    return float(total_loss.item()), float(norm_tensor.item()), finite


def loop(session: Session) -> dict[str, object]:
    config, state = session.config, session.state

    def status() -> dict[str, object]:
        return {
            "phase": session.phase,
            "step": state.step + 1,
            "micro": session.micro,
            "micros_per_step": config.micros_per_step,
        }

    with Heartbeat("train", status=status):
        ended = steps(session)
    session.phase = "checkpointing"
    path = save_checkpoint(config.directory, session.model, session.optimizer, state, config.keep)
    session.log.write("checkpoint", step=state.step, path=str(path))
    result = summary(session, ended)
    write_json(config.directory / "summary.json", result)
    session.log.write("end", **result)
    return result


def eta_hours(session: Session) -> float | None:
    """Steps left times the median of the last `ETA_WINDOW` step times; a projection."""
    recent = session.step_seconds[-ETA_WINDOW:]
    if not recent:
        return None
    left = session.total - session.state.step
    return round(left * statistics.median(recent) / 3600, 2)


def steps(session: Session) -> str:
    """Steps until the schedule ends, `stop_after` is reached, or a signal arrives."""
    config, state = session.config, session.state
    if config.eval_every and state.step == 0:
        session.phase = "evaluating"
        evaluate_now(session)
    session.model.train()
    micro_batches = (
        stream(config, session.tokenizer.eos, state.step, session.total)
        if session.rows is None
        else sft_stream(session.rows, config.micros_per_step, state.step, session.total)
    )
    ended = "schedule"
    while state.step < session.total:
        session.phase = "training"
        if config.stop_after is not None and state.step >= config.stop_after:
            ended = "stop_after"
            break
        if session.stop is not None and session.stop.signal is not None:
            ended = f"signal {session.stop.signal}"
            break
        started = time.perf_counter()
        session.phase = "loading data"
        batch: list[PackedRow] = []
        for _ in range(config.micros_per_step):
            waited = time.perf_counter()
            batch.append(next(micro_batches))
            session.wait_seconds.append(time.perf_counter() - waited)
        session.phase = "training"
        lr = scheduled_rate(session)
        loss, norm, taken = step(session, batch)
        elapsed = time.perf_counter() - started
        processed = sum(int(row.input_ids.size) for row in batch)
        state.step += 1
        state.tokens += processed
        state.targets += sum(row.targets for row in batch)
        session.tokens_this_run += processed
        session.step_seconds.append(elapsed)
        session.step_tokens.append(processed)
        session.log.write(
            "step",
            step=state.step,
            loss=loss,
            lr=lr,
            grad_norm=norm,
            taken=taken,
            tokens=state.tokens,
            step_s=round(elapsed, 3),
            wait_s=round(math.fsum(session.wait_seconds[-config.micros_per_step :]), 4),
            tokens_per_s=round(processed / elapsed, 1),
            mfu=round(6 * session.parameters * processed / elapsed / PEAK_FLOPS, 4),
            memory=peak_memory(session.device),
            total_steps=session.total,
            eta_h=eta_hours(session),
        )
        if not taken:
            state.skipped += 1
            if state.skipped > MAX_SKIPPED:
                message = f"{state.skipped} steps had a non-finite gradient norm"
                raise TrainError(message)
        elif not math.isfinite(loss):
            message = f"step {state.step}: loss {loss} with a finite gradient norm"
            raise TrainError(message)
        if config.eval_every and (
            state.step % config.eval_every == 0 or state.step == session.total
        ):
            session.phase = "evaluating"
            evaluate_now(session)
        if state.step % config.checkpoint_every == 0 and state.step < session.total:
            session.phase = "checkpointing"
            path = save_checkpoint(
                config.directory, session.model, session.optimizer, state, config.keep
            )
            session.log.write("checkpoint", step=state.step, path=str(path))
    return ended


def train(
    config: RunConfig,
    load: Callable[[str | Path, torch.dtype], LanguageModel],
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    stop: StopRequest | None = None,
) -> dict[str, object]:
    """Run or resume `config` to the end of its schedule, `stop_after`, or a signal."""
    log = EventLog(config.directory / "log.jsonl")
    with RunLock(config.directory / "lock", log):
        check_identity(config, log)
        rows = validation = None
        if config.objective == "cpt":
            blocks, _ = block_shape(config.blocks)
            available = blocks // config.blocks_per_step
        else:
            rows = Rows.read(config.blocks)
            validation = Rows.read(config.validation) if config.validation else None
            available = len(sft_order(len(rows))) // config.micros_per_step
        total = config.total_steps if config.total_steps is not None else available
        if not 1 <= total <= available:
            message = f"{total} steps requested, {config.blocks} holds {available}"
            raise TrainError(message)

        found = checkpoints(config.directory)
        latest = found[-1] if found else None
        source = latest if latest is not None else config.model
        with stage("load model", source=str(source)):
            model = load(source, weight_dtype(config.optimizer))
        parameters = count_parameters(model, config)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = build_optimizer(model, config)
        if latest is not None:
            with stage("restore optimizer and generators", checkpoint=str(latest)):
                state = restore(latest, optimizer)
        else:
            torch.manual_seed(TORCH_SEED)
            state = RunState()
        with stage("digest blocks", path=str(config.blocks)):
            digest = file_digest(config.blocks)
        log.write(
            "start",
            resumed_from=str(latest) if latest else None,
            step=state.step,
            total_steps=total,
            parameters=parameters,
            config=config.identity(),
            config_hash=config_hash(config.identity()),
            code=code_stamp(),
            blocks_sha256=digest,
        )
        accumulator = (
            GradientAccumulator(model, device)
            if parameters > LARGE_MODEL_PARAMETERS and device.type == "cuda"
            else None
        )
        session = Session(
            config,
            model,
            optimizer,
            state,
            tokenizer,
            device,
            total,
            parameters,
            log,
            stop,
            rows,
            validation,
            accumulator,
        )
        return loop(session)


def losses(log_path: Path) -> dict[int, float]:
    """The last logged loss per step. A resumed run logs the steps after its checkpoint again."""
    pairs: dict[int, float] = {}
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "step":
                pairs[int(event["step"])] = float(event["loss"])
    return pairs
