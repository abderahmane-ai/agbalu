"""The training loop: device-agnostic, resumable, structured logging."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from agbalu.model.checkpoint import (
    BEST,
    LATEST,
    ResumeFrom,
    TrainingState,
    load,
    resume_point,
    save,
)
from agbalu.model.config import RunConfig
from agbalu.model.data import PackedDataset, iter_batches
from agbalu.model.lock import acquire, default_owner, refresh, release
from agbalu.model.modeling import Encoder, LossOutput
from agbalu.model.optim import Lamb
from agbalu.model.preview import Previewer

log: Final = logging.getLogger("agbalu.model")

MAX_NON_FINITE: Final = 8
"""Consecutive non-finite micro-batches tolerated before rolling back. Isolated NaNs come
from a single pathological batch under fp16; a run of them means the weights are gone."""

NO_DECAY: Final = ("bias", "norm", "relative_embedding")

PERPLEXITY_CEILING: Final = 1e6
"""Above this the number carries no information; at init, loss ln(16000) is already 16k."""


def _perplexity(loss: float) -> str:
    if not math.isfinite(loss):
        return "n/a"
    value = math.exp(min(loss, math.log(PERPLEXITY_CEILING)))
    return f"{value:,.0f}" if value < PERPLEXITY_CEILING else ">1e6"


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ForwardPass(Protocol):
    """What the training step calls. `torch.compile` returns a callable rather than a
    `Module`, so the compiled and eager paths meet at this signature."""

    def __call__(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> LossOutput: ...


def quieten_dynamo() -> None:
    """Silence the `Tensor.item()` graph-break explainer, ten lines of it.

    `Encoder.forward` reads `selected.any()` into Python to guard the all-unmasked batch,
    so dynamo splits the graph and warns once per process. The break is already inside the
    measured compiled throughput. Only the one logger that carries it is raised, so a
    recompilation storm — `torch._dynamo.convert_frame` — still reports itself.
    """
    logging.getLogger("torch._dynamo.variables.tensor").setLevel(logging.ERROR)


def _compile(model: nn.Module, requested: bool, device: torch.device) -> ForwardPass:
    """The callable the forward pass uses: compiled on CUDA when asked, else the model.

    Only on CUDA. On CPU and MPS the backend either falls back or is slower, and the
    failure drills in the suite run on CPU — compiling them would test the compiler.
    """
    eager: ForwardPass = model
    if not requested or device.type != "cuda":
        return eager
    log.info("compiling the model")
    quieten_dynamo()
    compiled: ForwardPass = torch.compile(model)
    return compiled


def build_optimizer(model: nn.Module, config: RunConfig) -> Optimizer:
    """Weights decay; biases, norms and the relative-position table do not."""
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if any(tag in name for tag in NO_DECAY) else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=config.train.learning_rate,
            betas=(config.train.beta1, config.train.beta2),
            eps=config.train.eps,
        )
    return Lamb(
        groups,
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        eps=config.train.eps,
    )


@dataclass(frozen=True, slots=True)
class StepReport:
    step: int
    loss: float
    accuracy: float
    learning_rate: float
    mask_probability: float
    grad_norm: float
    tokens_seen: int
    seconds: float


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What *this process* did, as opposed to the cumulative counters in `TrainingState`.

    A resumed run inherits `tokens_seen` from the checkpoint, so dividing it by this
    process's elapsed time reports a rate the hardware never reached.
    """

    steps: int
    tokens: int
    masked: int
    training_seconds: float
    """Wall clock inside the training loop, less validation and checkpoint writes."""
    wall_seconds: float

    @property
    def training_tokens_per_second(self) -> float | None:
        """None, never 0.0: a run that trained nothing has no rate, and 0.0 reads as a
        measured stall and propagates into any projection built on it."""
        if self.steps < 1 or self.training_seconds <= 0.0:
            return None
        return self.tokens / self.training_seconds

    @property
    def wall_clock_tokens_per_second(self) -> float | None:
        if self.steps < 1 or self.wall_seconds <= 0.0:
            return None
        return self.tokens / self.wall_seconds


class _Accumulator:
    """One optimiser step's loss and accuracy, gathered across its micro-batches.

    Loss is weighted by scored tokens, which is what `Trainer.validate` returns, so the
    two numbers in the log are the same statistic and their gap means something. Both
    stay on device until `means`, so accumulating costs no synchronisation.
    """

    def __init__(self, device: torch.device) -> None:
        self.loss = torch.zeros((), device=device)
        self.accuracy = torch.zeros((), device=device)
        self.masked = 0

    def add(self, loss: Tensor, accuracy: Tensor, masked: int) -> None:
        self.loss += loss.detach() * masked
        self.accuracy += accuracy
        self.masked += masked

    def means(self, micro_batches: int) -> tuple[float, float]:
        if not self.masked or micro_batches < 1:
            return 0.0, 0.0
        pair = torch.stack((self.loss / self.masked, self.accuracy / micro_batches))
        loss, accuracy = pair.tolist()
        return float(loss), float(accuracy)


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        train_data: PackedDataset,
        validation_data: PackedDataset,
        out_dir: Path,
        *,
        device: torch.device | None = None,
        previewer: Previewer | None = None,
        owner: str | None = None,
        force: bool = False,
        resume_from: ResumeFrom = "latest",
    ) -> None:
        self.owner = owner or default_owner()
        self.force = force
        self.resume_from = resume_from
        self.config = config
        self.train_data = train_data
        self.validation_data = validation_data
        self.out_dir = out_dir
        self.device = device or select_device()
        self.model = Encoder(config.model).to(self.device)
        self.compiled = _compile(self.model, config.train.compile, self.device)
        """What the forward pass calls. `self.model` stays the eager module, because a
        compiled wrapper prefixes every state-dict key with `_orig_mod.` and a checkpoint
        written through it cannot be loaded by a run that has compilation off."""
        self.optimizer = build_optimizer(self.model, config)
        self.state = TrainingState()
        self.metrics_path = out_dir / "metrics.jsonl"
        self.previewer = previewer
        self.use_amp = config.train.mixed_precision and self.device.type == "cuda"
        """bfloat16, so no GradScaler: bf16 carries fp32's exponent range, and gradient
        scaling exists only to stop fp16 gradients underflowing."""
        self._window_started = time.monotonic()
        self._window_tokens = 0
        self._run_steps = 0
        self._run_tokens = 0
        self._run_masked = 0
        self._side_seconds = 0.0
        self._training_seconds = 0.0
        self._wall_seconds = 0.0

    @property
    def summary(self) -> RunSummary:
        return RunSummary(
            steps=self._run_steps,
            tokens=self._run_tokens,
            masked=self._run_masked,
            training_seconds=self._training_seconds,
            wall_seconds=self._wall_seconds,
        )

    def maybe_resume(self) -> str | None:
        """The checkpoint file resumed from, or None if the run started fresh."""
        chosen = resume_point(self.out_dir, prefer=self.resume_from)
        if chosen is None:
            return None
        self.state = load(self.out_dir, self.model, self.optimizer, name=chosen.name)
        return chosen.name

    @contextmanager
    def _off_clock(self) -> Iterator[None]:
        """Validation and checkpoint writes are not training; their time is excluded from
        `training_seconds` so the reported rate describes the training loop."""
        started = time.monotonic()
        try:
            yield
        finally:
            self._side_seconds += time.monotonic() - started

    def _emit(self, event: str, **fields: float | int | str | None) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **fields}) + "\n")

    def _forward(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, int, int]:
        """Returns `(loss, accuracy, masked_tokens, tokens)`. `accuracy` stays on device."""
        ids, attention, labels = (t.to(self.device, non_blocking=True) for t in batch)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            output = self.compiled(ids, attention, labels)
        loss = output.loss + self.config.train.z_loss_weight * output.z_loss
        return loss, output.accuracy, output.num_tokens, int(ids.numel())

    def _rollback(self) -> None:
        log.error("non-finite loss %d times consecutively; rolling back", MAX_NON_FINITE)
        self._emit("rollback", step=self.state.step)
        chosen = resume_point(self.out_dir, prefer=self.resume_from)
        if chosen is None:
            msg = "loss diverged before the first checkpoint; nothing to roll back to"
            raise RuntimeError(msg)
        self.state = load(self.out_dir, self.model, self.optimizer, name=chosen.name)
        self.state.consecutive_non_finite = 0

    @torch.no_grad()
    def validate(self, batches: int | None = None) -> float:
        batches = self.config.train.validation_batches if batches is None else batches
        self.model.eval()
        total, seen = 0.0, 0
        for batch in iter_batches(
            self.validation_data,
            self.config.train.local_batch_size,
            seed=self.config.train.seed,
            epoch=0,
            drop_last=False,
        ):
            loss, _, masked, _tokens = self._forward(batch)
            if math.isfinite(float(loss)) and masked:
                total += float(loss) * masked
                seen += masked
            batches -= 1
            if batches <= 0:
                break
        self.model.train()
        if not seen:
            msg = "validation produced no scored tokens; the held-out split is empty or unmasked"
            raise RuntimeError(msg)
        return total / seen

    def _announce(self) -> int:
        """Log the run and its plan, and return the steps left to train — 0 for a resume
        that is already at `max_steps`, where the loop would be a no-op."""
        config = self.config.train
        resumed = self.maybe_resume()
        log.info(
            "run %s | %s | %s parameters | %s vocab | %s",
            self.config.name,
            self.device,
            f"{self.model.parameter_count():,}",
            f"{self.config.model.vocab_size:,}",
            f"resumed {resumed} at step {self.state.step}" if resumed else "from scratch",
        )

        remaining = config.max_steps - self.state.step
        if remaining < 1:
            log.warning(
                "nothing to train: resumed at step %d of %d. Raise max_steps, or clear %s "
                "to run again — every rate this run could report would describe an empty loop",
                self.state.step,
                config.max_steps,
                self.out_dir,
            )
            self._emit("noop", step=self.state.step, max_steps=config.max_steps)
            return 0

        log.info(
            "plan step %d -> %d | %s steps x %d micro-batches of %d x %d | %s tokens"
            " | lr %.2e %s | mask %.2f->%.2f",
            self.state.step,
            config.max_steps,
            f"{remaining:,}",
            config.accumulation_steps,
            config.local_batch_size,
            config.seq_length,
            f"{remaining * config.tokens_per_step:,}",
            config.learning_rate,
            config.optimizer,
            config.mask_p_start,
            config.mask_p_end,
        )
        self._emit(
            "start",
            device=str(self.device),
            parameters=self.model.parameter_count(),
            step=self.state.step,
            max_steps=config.max_steps,
            remaining_steps=remaining,
            tokens_per_step=config.tokens_per_step,
            resumed_from=resumed,
        )
        return remaining

    def train(self) -> TrainingState:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        acquire(self.out_dir, self.owner, force=self.force)
        try:
            return self._train_locked()
        finally:
            release(self.out_dir, self.owner)

    def _train_locked(self) -> TrainingState:
        config = self.config.train
        self.model.train()
        if self._announce() < 1:
            return self.state

        started = self._window_started = time.monotonic()
        self._window_tokens = self.state.tokens_seen
        micro = 0
        accumulated = _Accumulator(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        while self.state.step < config.max_steps:
            self.train_data.set_step(self.state.step)
            batches = iter_batches(
                self.train_data,
                config.local_batch_size,
                seed=config.seed,
                epoch=self.state.epoch,
                start_batch=self.state.batch_in_epoch,
            )
            exhausted = True
            for batch in batches:
                exhausted = False
                loss, accuracy, masked, tokens = self._forward(batch)
                if not torch.isfinite(loss):
                    self.state.consecutive_non_finite += 1
                    self._emit("non_finite", step=self.state.step)
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.state.consecutive_non_finite >= MAX_NON_FINITE:
                        self._rollback()
                    micro = 0
                    accumulated = _Accumulator(self.device)
                    continue

                self.state.consecutive_non_finite = 0
                torch.autograd.backward(loss / config.accumulation_steps)
                accumulated.add(loss, accuracy, masked)
                self.state.tokens_seen += tokens
                self.state.masked_seen += masked
                self._run_tokens += tokens
                self._run_masked += masked
                self.state.batch_in_epoch += 1
                micro += 1

                if micro < config.accumulation_steps:
                    continue

                mean_loss, mean_accuracy = accumulated.means(micro)
                report = self._optimiser_step(
                    loss=mean_loss,
                    accuracy=mean_accuracy,
                    started=started,
                )
                micro = 0
                accumulated = _Accumulator(self.device)
                self.train_data.set_step(self.state.step)
                self._after_step(report)
                if self.state.step >= config.max_steps:
                    break

            if exhausted or self.state.batch_in_epoch >= len(self.train_data):
                self.state.epoch += 1
                self.state.batch_in_epoch = 0

        self._training_seconds = time.monotonic() - started - self._side_seconds
        save(self.out_dir, self.model, self.optimizer, self.state, name=LATEST)
        self._wall_seconds = time.monotonic() - started
        self._report()
        return self.state

    def _report(self) -> None:
        summary = self.summary
        self._emit(
            "finish",
            step=self.state.step,
            steps_this_run=summary.steps,
            tokens_seen=self.state.tokens_seen,
            tokens_this_run=summary.tokens,
            masked_this_run=summary.masked,
            training_seconds=round(summary.training_seconds, 3),
            wall_seconds=round(summary.wall_seconds, 3),
            training_tokens_per_second=summary.training_tokens_per_second,
        )
        log.info(
            "done step %d | %s steps this run | %s tokens (%s masked) | %s tok/s training,"
            " %s wall | best eval loss %7.4f ppl %s",
            self.state.step,
            f"{summary.steps:,}",
            f"{summary.tokens:,}",
            f"{summary.masked:,}",
            _rate(summary.training_tokens_per_second),
            _rate(summary.wall_clock_tokens_per_second),
            self.state.best_validation_loss,
            _perplexity(self.state.best_validation_loss),
        )

    def _optimiser_step(self, *, loss: float, accuracy: float, started: float) -> StepReport:
        """Clip, step, and report."""
        config = self.config.train
        learning_rate = config.learning_rate_at(self.state.step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_gradient)
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        self._run_steps += 1
        return StepReport(
            step=self.state.step,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            mask_probability=config.mask_probability(self.state.step),
            grad_norm=grad_norm,
            tokens_seen=self.state.tokens_seen,
            seconds=time.monotonic() - started,
        )

    def _should_log(self, step: int) -> bool:
        """First and last step always, so a short run is never silent."""
        config = self.config.train
        return step == 1 or step >= config.max_steps or step % config.log_every == 0

    def _throughput(self, tokens_seen: int) -> float:
        """Tokens per second over the window since the last log line.

        Windowed rather than cumulative because validation and checkpoint writes sit
        between logged steps; averaging them in reports the run's wall clock, not the
        training loop's speed, and the two differ by several times on a short run.
        """
        now = time.monotonic()
        rate = (tokens_seen - self._window_tokens) / max(now - self._window_started, 1e-9)
        self._window_started, self._window_tokens = now, tokens_seen
        return rate

    def _log_previews(self) -> None:
        if self.previewer is None:
            return
        for preview in self.previewer.preview(self.model, self.device):
            log.info("    gold %s", preview.gold)
            log.info("    pred %s  [%d/%d]", preview.predicted, preview.correct, preview.total)

    def _after_step(self, report: StepReport) -> None:
        config = self.config.train
        self._emit("step", **asdict(report))
        if self._should_log(report.step):
            log.info(
                "step %*d/%d | loss %7.4f ppl %8s acc %.4f | lr %.2e grad %5.2f mask %.3f "
                "| %s tok/s | epoch %d",
                len(str(config.max_steps)),
                report.step,
                config.max_steps,
                report.loss,
                _perplexity(report.loss),
                report.accuracy,
                report.learning_rate,
                report.grad_norm,
                report.mask_probability,
                f"{self._throughput(report.tokens_seen):,.0f}",
                self.state.epoch,
            )
        if report.step % config.validate_every == 0:
            with self._off_clock():
                self._evaluate(report.step)
        if report.step % config.save_every == 0:
            with self._off_clock():
                save(self.out_dir, self.model, self.optimizer, self.state, name=LATEST)
                refresh(self.out_dir, self.owner)
                self._emit("checkpoint", step=report.step, name=LATEST)

    def _evaluate(self, step: int) -> None:
        validation = self.validate()
        self.state.history.append({"step": step, "validation_loss": validation})
        self._emit("validation", step=step, loss=validation)
        improved = validation < self.state.best_validation_loss
        log.info(
            "eval %*d/%d | loss %7.4f ppl %8s%s",
            len(str(self.config.train.max_steps)),
            step,
            self.config.train.max_steps,
            validation,
            _perplexity(validation),
            "  <- best" if improved else "",
        )
        self._log_previews()
        if improved:
            self.state.best_validation_loss = validation
            save(self.out_dir, self.model, self.optimizer, self.state, name=BEST)
            self._emit("checkpoint", step=step, name=BEST)
