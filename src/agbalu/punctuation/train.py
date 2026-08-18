"""Fine-tuning the encoder for both heads, with the guards a run that can die needs.

Two learning rates, because the encoder arrives pretrained and the heads arrive random: one
rate for both either destroys what Masinissa knows or leaves the heads untrained.

The objective is plain cross-entropy. An earlier run weighted the classes by capped inverse
frequency at 20x, reasoning that `NONE` is 82.9% of labels and would otherwise swamp the
rest. It did not swamp anything — the weights did the opposite, and the model reached recall
0.919 on commas at precision 0.381, inserting 1,200 marks that were not there. This label
distribution is close to the standard benchmark's (IWSLT2011: 85.7% none, 7.53% comma, 6.3%
period, 0.47% question), and the published systems on it fine-tune with unweighted
cross-entropy. Ours is the normal case, so it gets the normal recipe.

Selection is on dev macro-F1, not loss. On the weighted run the two disagreed: the
loss-selected checkpoint scored 0.649 where the final one scored 0.670, and loss is not the
quantity anyone reports.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.optim import AdamW

from agbalu.model.checkpoint import BEST, LATEST, TrainingState
from agbalu.model.checkpoint import load as load_checkpoint
from agbalu.model.checkpoint import save as save_checkpoint
from agbalu.punctuation.dataset import Batch, EncodedCorpus, collate, iter_batches
from agbalu.punctuation.evaluate import macro_f1, per_class, upper_init_f1
from agbalu.punctuation.labels import CASE, IGNORE_INDEX, LOWER, NONE, PUNCTUATION
from agbalu.punctuation.model import Restorer

log: Final = logging.getLogger("agbalu.punctuation")

#: Warmup is a share of the schedule, never a fixed count: 500 fixed steps on a 200-step
#: smoke trains the whole run at a fraction of peak and measures nothing.
WARMUP_SHARE: Final = 0.06
WARMUP_CAP: Final = 2_000

#: Derived rather than fixed, because a fixed cadence over an unknown schedule gives either
#: four points or thousands. The first and last step always log regardless.
VALIDATIONS_PER_EPOCH: Final = 4
LOGS_PER_EPOCH: Final = 40

MAX_CONSECUTIVE_NON_FINITE: Final = 20

#: Weight decay applies to matrices. Biases, gains and norm parameters are excluded.
MATRIX_NDIM: Final = 2


class TrainingError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class TrainSettings:
    epochs: int = 3
    batch_size: int = 128
    encoder_lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    case_loss_weight: float = 1.0
    dropout: float = 0.1
    freeze_encoder: bool = False
    seed: int = 17

    def __post_init__(self) -> None:
        for name, value in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if value < 1:
                msg = f"{name} must be at least 1, got {value}"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Validation:
    loss: float
    punctuation_macro_f1: float
    case_noninitial_f1: float

    @property
    def score(self) -> float:
        """What the checkpoint is selected on. Loss is reported beside it and never used to
        choose: on the weighted run the two disagreed by 0.021 macro-F1."""
        return (self.punctuation_macro_f1 + self.case_noninitial_f1) / 2


@dataclass(slots=True)
class RunSummary:
    total_steps: int
    steps_this_run: int = 0
    labelled_this_run: int = 0
    seconds_this_run: float = 0.0
    best: Validation | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def labels_per_second(self) -> float | None:
        """`None`, never zero, for a run that completed no step. A clamped rate reads as a
        measurement: `max(rate, 1.0)` once reported a run length of 11,650,844 hours."""
        if not self.steps_this_run or self.seconds_this_run <= 0:
            return None
        return self.labelled_this_run / self.seconds_this_run


def schedule(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on each group's peak rate."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_optimizer(model: Restorer, settings: TrainSettings) -> AdamW:
    """Four groups: encoder and heads, each split on whether weight decay applies."""
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (head, flat): [] for head in (False, True) for flat in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith(("punctuation.", "case."))
        flat = parameter.ndim < MATRIX_NDIM or "norm" in name
        buckets[head, flat].append(parameter)

    groups = [
        {
            "params": parameters,
            "lr": settings.head_lr if head else settings.encoder_lr,
            "weight_decay": 0.0 if flat else settings.weight_decay,
        }
        for (head, flat), parameters in buckets.items()
        if parameters
    ]
    if not groups:
        msg = "no trainable parameters — freezing the encoder also froze the heads"
        raise TrainingError(msg)
    return AdamW(groups, betas=(0.9, 0.98), eps=1e-6)


def initial_positions(labelled: Tensor) -> Tensor:
    """True at each row's first labelled position: the sentence-initial word."""
    initial = torch.zeros_like(labelled)
    present = labelled.any(dim=1)
    if not bool(present.any()):
        return initial
    rows = torch.arange(labelled.size(0), device=labelled.device)[present]
    initial[rows, labelled.float().argmax(dim=1)[present]] = True
    return initial


class Trainer:
    def __init__(
        self,
        model: Restorer,
        train: EncodedCorpus,
        dev: EncodedCorpus,
        settings: TrainSettings,
        directory: Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_corpus = train
        self.dev_corpus = dev
        self.settings = settings
        self.directory = directory
        self.device = device

        self.steps_per_epoch = math.ceil(len(train) / settings.batch_size)
        self.total_steps = self.steps_per_epoch * settings.epochs
        self.warmup = min(WARMUP_CAP, max(1, int(self.total_steps * WARMUP_SHARE)))
        self.validate_every = max(1, self.steps_per_epoch // VALIDATIONS_PER_EPOCH)
        self.log_every = max(1, self.steps_per_epoch // LOGS_PER_EPOCH)

        self.optimizer = build_optimizer(model, settings)
        self.peaks = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.state = TrainingState()
        self.summary = RunSummary(total_steps=self.total_steps)
        self.rng = np.random.default_rng(settings.seed)
        self.best_score = 0.0

    def _emit(self, event: str, **fields: float | int | str | None) -> None:
        rendered = " ".join(
            f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in fields.items()
            if value is not None
        )
        log.info("%s %s", event, rendered)

    def _head_losses(self, batch: Batch, reduction: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        heads = self.model(batch.input_ids, batch.attention_mask)
        punctuation = functional.cross_entropy(
            heads.punctuation.reshape(-1, len(PUNCTUATION)),
            batch.punctuation.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction=reduction,
        )
        case = functional.cross_entropy(
            heads.case.reshape(-1, len(CASE)),
            batch.case.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction=reduction,
        )
        return punctuation, case, heads.punctuation, heads.case

    def _set_rate(self, step: int) -> float:
        multiplier = schedule(step, self.total_steps, self.warmup)
        for group, peak in zip(self.optimizer.param_groups, self.peaks, strict=True):
            group["lr"] = peak * multiplier
        return self.peaks[0] * multiplier

    @torch.inference_mode()
    def validate(self) -> Validation:
        """Token-weighted, so the number does not depend on how the split batched."""
        self.model.eval()
        punctuation_total = case_total = 0.0
        labelled_total = 0
        punctuation_pairs: list[tuple[int, int]] = []
        case_pairs: list[tuple[int, int]] = []

        for indices in iter_batches(self.dev_corpus, self.settings.batch_size, shuffle=False):
            batch = collate(self.dev_corpus, indices).to(self.device)
            punctuation, case, punctuation_logits, case_logits = self._head_losses(batch, "sum")

            labelled = batch.punctuation != IGNORE_INDEX
            gold_punctuation = batch.punctuation[labelled]
            punctuation_total += float(punctuation)
            case_total += float(case)
            labelled_total += int(gold_punctuation.numel())

            predicted_punctuation = punctuation_logits.argmax(-1)[labelled]
            punctuation_pairs.extend(
                zip(gold_punctuation.tolist(), predicted_punctuation.tolist(), strict=True)
            )

            noninitial = labelled & ~initial_positions(labelled)
            case_pairs.extend(
                zip(
                    batch.case[noninitial].tolist(),
                    case_logits.argmax(-1)[noninitial].tolist(),
                    strict=True,
                )
            )

        self.model.train()
        weighted = punctuation_total + self.settings.case_loss_weight * case_total
        loss = weighted / max(labelled_total, 1)
        marks = [
            entry
            for entry in per_class(punctuation_pairs, PUNCTUATION)
            if entry["label"] != PUNCTUATION[NONE]
        ]
        return Validation(
            loss=loss,
            punctuation_macro_f1=macro_f1(marks),
            case_noninitial_f1=upper_init_f1(per_class(case_pairs, CASE, skip=LOWER)),
        )

    def maybe_resume(self) -> bool:
        """Restore the step counters, and the best score with them.

        The score lives in `state.history` rather than in a field of its own: `TrainingState`
        records a validation *loss*, and writing a negated score into a field named for a loss
        is how a resumed run silently stops selecting on what it was selecting on.
        """
        if not (self.directory / LATEST).is_file():
            return False
        self.state = load_checkpoint(self.directory, self.model, self.optimizer, name=LATEST)
        self.summary.history = list(self.state.history)
        self.best_score = max((row.get("score", 0.0) for row in self.state.history), default=0.0)
        return True

    def _checkpoint(self, validation: Validation) -> None:
        save_checkpoint(self.directory, self.model, self.optimizer, self.state, name=LATEST)
        if validation.score > self.best_score:
            self.best_score = validation.score
            self.state.best_validation_loss = validation.loss
            self.summary.best = validation
            save_checkpoint(self.directory, self.model, self.optimizer, self.state, name=BEST)
            self._emit(
                "best",
                step=self.state.step,
                score=validation.score,
                loss=validation.loss,
                punctuation_f1=validation.punctuation_macro_f1,
                case_f1=validation.case_noninitial_f1,
            )

    def _optimizer_step(self) -> float:
        """Clip, read the norm, and gate the step on it: a non-finite check that runs after
        `step()` reports the damage it already did."""
        norm = float(
            nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.settings.max_grad_norm,
            )
        )
        if not math.isfinite(norm):
            self.state.consecutive_non_finite += 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.state.consecutive_non_finite >= MAX_CONSECUTIVE_NON_FINITE:
                msg = f"{self.state.consecutive_non_finite} consecutive non-finite gradients"
                raise TrainingError(msg)
            return norm
        self.state.consecutive_non_finite = 0
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def run(self) -> RunSummary:
        if self.state.step >= self.total_steps:
            self._emit("complete", step=self.state.step, total=self.total_steps)
            return self.summary

        self.model.train()
        self._emit(
            "start",
            step=self.state.step,
            total=self.total_steps,
            steps_per_epoch=self.steps_per_epoch,
            warmup=self.warmup,
            validate_every=self.validate_every,
            trainable=self.model.trainable_parameters(),
            device=str(self.device),
        )
        started = time.monotonic()

        for epoch in range(self.state.epoch, self.settings.epochs):
            self.state.epoch = epoch
            for indices in iter_batches(
                self.train_corpus, self.settings.batch_size, shuffle=True, generator=self.rng
            ):
                if self.state.step >= self.total_steps:
                    break
                batch = collate(self.train_corpus, indices).to(self.device)
                punctuation, case, _, _ = self._head_losses(batch, "mean")
                loss = punctuation + self.settings.case_loss_weight * case
                torch.autograd.backward(loss)
                norm = self._optimizer_step()

                rate = self._set_rate(self.state.step)
                self.state.step += 1
                self.state.masked_seen += batch.labelled
                self.summary.steps_this_run += 1
                self.summary.labelled_this_run += batch.labelled

                first_or_last = self.state.step in (1, self.total_steps)
                if first_or_last or self.state.step % self.log_every == 0:
                    self._emit(
                        "step",
                        step=self.state.step,
                        epoch=epoch,
                        loss=float(loss.detach()),
                        punctuation=float(punctuation.detach()),
                        case=float(case.detach()),
                        grad_norm=norm,
                        lr=rate,
                    )
                if first_or_last or self.state.step % self.validate_every == 0:
                    validation = self.validate()
                    self.summary.history.append(
                        {
                            "step": float(self.state.step),
                            "score": validation.score,
                            "loss": validation.loss,
                            "punctuation_macro_f1": validation.punctuation_macro_f1,
                            "case_noninitial_f1": validation.case_noninitial_f1,
                        }
                    )
                    self.state.history = self.summary.history
                    self._emit(
                        "validate",
                        step=self.state.step,
                        score=validation.score,
                        loss=validation.loss,
                        punctuation_f1=validation.punctuation_macro_f1,
                        case_f1=validation.case_noninitial_f1,
                    )
                    self._checkpoint(validation)

        self.summary.seconds_this_run = time.monotonic() - started
        self.state.history = self.summary.history
        self._emit(
            "finish",
            steps_this_run=self.summary.steps_this_run,
            seconds_this_run=self.summary.seconds_this_run,
            labels_per_second=self.summary.labels_per_second,
        )
        return self.summary
