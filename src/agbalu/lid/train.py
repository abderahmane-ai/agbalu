"""Training the language identifier, with the guards a run that can die needs.

Selection is on dev, and on the mean of macro-F1 and worst-group recall. The previous
trainer selected `best.pt` by macro-F1 on the held-out set — the published benchmark
split — so the reported number was chosen on the thing it reported.

The objective is cross-entropy under group-DRO weights plus the domain-aware
contrastive term; `losses.py` carries what each answers and why.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn
from torch.optim import AdamW

from agbalu.lid.config import DEFAULT_CHECKPOINT_DIR, ModelConfig, TrainConfig
from agbalu.lid.dataset import (
    Augmentation,
    Batch,
    Corpus,
    Record,
    group_batches,
    read_split,
    sequential_batches,
)
from agbalu.lid.evaluate import Report, build_report
from agbalu.lid.losses import (
    GroupWeights,
    SupervisedContrastiveLoss,
    cross_entropy_per_sample,
    ldam_margins,
)
from agbalu.lid.model import CharClassifier
from agbalu.model.checkpoint import BEST, LATEST, TrainingState
from agbalu.model.checkpoint import load as load_checkpoint
from agbalu.model.checkpoint import save as save_checkpoint

log: Final = logging.getLogger("agbalu.lid")

WARMUP_SHARE: Final = 0.06
WARMUP_CAP: Final = 1_000
VALIDATIONS_PER_EPOCH: Final = 4
LOGS_PER_EPOCH: Final = 20
MAX_CONSECUTIVE_NON_FINITE: Final = 20
MATRIX_NDIM: Final = 2


class TrainingError(Exception):
    """A run that cannot continue."""


@dataclass(slots=True)
class RunSummary:
    total_steps: int
    steps_this_run: int = 0
    rows_this_run: int = 0
    seconds_this_run: float = 0.0
    best: Report | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def rows_per_second(self) -> float | None:
        """`None`, never zero, for a run that completed no step."""
        if not self.steps_this_run or self.seconds_this_run <= 0:
            return None
        return self.rows_this_run / self.seconds_this_run


def schedule(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the peak rate."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_optimizer(model: nn.Module, settings: TrainConfig) -> AdamW:
    """Two groups, split on whether weight decay applies. Norms and gains are exempt."""
    decayed: list[nn.Parameter] = []
    flat: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (flat if parameter.ndim < MATRIX_NDIM or "norm" in name else decayed).append(parameter)
    groups = [
        {"params": decayed, "weight_decay": settings.weight_decay},
        {"params": flat, "weight_decay": 0.0},
    ]
    usable = [group for group in groups if group["params"]]
    if not usable:
        msg = "no trainable parameters"
        raise TrainingError(msg)
    return AdamW(usable, lr=settings.learning_rate, betas=(0.9, 0.98), eps=1e-8)


class Trainer:
    def __init__(
        self,
        model: CharClassifier,
        train: Corpus,
        dev: Corpus,
        settings: TrainConfig,
        directory: Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_corpus = train
        self.dev_corpus = dev
        self.settings = settings
        self.directory = directory
        self.device = device

        # The sampler draws with replacement, so an epoch is a defined number of steps
        # rather than a pass over a shuffled index; `epochs` would otherwise mean nothing
        # under group balancing.
        self.steps_per_epoch = max(1, len(train) // settings.batch_size)
        self.total_steps = self.steps_per_epoch * settings.epochs
        self.warmup = min(WARMUP_CAP, max(1, int(self.total_steps * WARMUP_SHARE)))
        self.validate_every = max(1, self.steps_per_epoch // VALIDATIONS_PER_EPOCH)
        self.log_every = max(1, self.steps_per_epoch // LOGS_PER_EPOCH)

        self.margins = ldam_margins(train.class_counts, model.config.margin_scale)
        self.deferred_at = int(self.total_steps * settings.deferred_share)
        self.group_weights = GroupWeights(len(train.groups), 0.0).to(device)
        self.contrastive = SupervisedContrastiveLoss(
            model.config.projection_dim,
            settings.memory_bank_size,
            settings.contrastive_temperature,
        ).to(device)

        self.optimizer = build_optimizer(model, settings)
        self.peak = settings.learning_rate
        self.state = TrainingState()
        self.summary = RunSummary(total_steps=self.total_steps)
        self.best_score = 0.0

    def apply_phase(self) -> bool:
        """Switch the margins and the group weights on once the deferred point is passed.

        Idempotent, and driven by `state.step` rather than by a flag, so a resumed run
        lands in the phase its step belongs to instead of restarting in phase one.
        """
        deferred = self.state.step >= self.deferred_at
        self.model.head.set_margins(self.margins if deferred else torch.zeros_like(self.margins))
        self.group_weights.step = self.settings.group_dro_step if deferred else 0.0
        return deferred

    def _emit(self, event: str, **fields: float | int | str | None) -> None:
        rendered = " ".join(
            f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in fields.items()
            if value is not None
        )
        log.info("%s %s", event, rendered)

    def _losses(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model(batch.input_ids, pad_mask=batch.pad_mask, labels=batch.labels)
        per_sample = cross_entropy_per_sample(
            output.logits, batch.labels, self.settings.label_smoothing
        )
        classification = self.group_weights(per_sample, batch.groups)
        contrastive = self.contrastive(output.projections, batch.labels, batch.groups)
        total = classification + self.settings.contrastive_weight * contrastive
        return total, classification, contrastive

    @torch.inference_mode()
    def evaluate(self, corpus: Corpus) -> Report:
        self.model.eval()
        pairs: list[tuple[int, int]] = []
        groups: list[str] = []
        for indices in sequential_batches(corpus, self.settings.eval_batch_size):
            batch = corpus.collate(indices).to(self.device)
            logits = self.model(batch.input_ids, pad_mask=batch.pad_mask).logits
            predictions = logits.argmax(dim=-1).cpu().tolist()
            pairs.extend(zip(batch.labels.cpu().tolist(), predictions, strict=True))
            groups.extend(corpus.records[index].group for index in indices)
        self.model.train()
        return build_report(pairs, corpus.classes, groups)

    def maybe_resume(self) -> bool:
        """Restore the counters and the best score, and the group weights with them."""
        if not (self.directory / LATEST).is_file():
            return False
        self.state = load_checkpoint(self.directory, self._checkpointed(), self.optimizer)
        self.summary.history = list(self.state.history)
        self.best_score = max((row.get("score", 0.0) for row in self.state.history), default=0.0)
        self.apply_phase()
        return True

    def _checkpointed(self) -> nn.Module:
        """The model, the group weights and the memory bank as one resumable module.

        The group-DRO weights are what the loss is taken against, and the bank is the
        negative pool: a run resumed without them spends its first epoch relearning
        state it had already paid for.
        """
        bundle = nn.Module()
        bundle.add_module("model", self.model)
        bundle.add_module("group_weights", self.group_weights)
        bundle.add_module("contrastive", self.contrastive)
        return bundle

    def _checkpoint(self, report: Report) -> None:
        save_checkpoint(
            self.directory, self._checkpointed(), self.optimizer, self.state, name=LATEST
        )
        if report.score > self.best_score:
            self.best_score = report.score
            self.state.best_validation_loss = 1.0 - report.score
            self.summary.best = report
            save_checkpoint(
                self.directory, self._checkpointed(), self.optimizer, self.state, name=BEST
            )
            worst, recall = report.worst_group
            self._emit(
                "best",
                step=self.state.step,
                score=report.score,
                macro_f1=report.macro_f1,
                worst_group=worst,
                worst_recall=recall,
            )

    def _optimizer_step(self) -> float:
        """Clip, read the norm, and gate the step on it: a non-finite check that runs
        after `step()` reports damage it has already done."""
        norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_grad_norm))
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

    def _set_rate(self, step: int) -> float:
        rate = self.peak * schedule(step, self.total_steps, self.warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def run(self) -> RunSummary:
        if self.state.step >= self.total_steps:
            self._emit("complete", step=self.state.step, total=self.total_steps)
            return self.summary

        self.model.train()
        deferred = self.apply_phase()
        self._emit(
            "start",
            step=self.state.step,
            total=self.total_steps,
            steps_per_epoch=self.steps_per_epoch,
            warmup=self.warmup,
            validate_every=self.validate_every,
            deferred_at=self.deferred_at,
            reweighted=int(deferred),
            groups=len(self.train_corpus.groups),
            batch=self.settings.batch_size,
            parameters=self.model.trainable_parameters(),
            device=str(self.device),
        )
        started = time.monotonic()
        remaining = self.total_steps - self.state.step
        sampler = group_batches(
            self.train_corpus,
            groups_per_batch=self.settings.groups_per_batch,
            samples_per_group=self.settings.samples_per_group,
            batches=remaining,
            seed=self.settings.seed + self.state.step,
        )

        for indices in sampler:
            batch = self.train_corpus.collate(indices).to(self.device)
            loss, classification, contrastive = self._losses(batch)
            torch.autograd.backward(loss)
            norm = self._optimizer_step()

            rate = self._set_rate(self.state.step)
            self.state.step += 1
            self.state.tokens_seen += int(batch.pad_mask.sum())
            self.summary.steps_this_run += 1
            self.summary.rows_this_run += len(batch)

            if self.state.step == self.deferred_at:
                self.apply_phase()
                self._emit("reweight", step=self.state.step, margin=float(self.margins.max()))

            first_or_last = self.state.step in (1, self.total_steps)
            if first_or_last or self.state.step % self.log_every == 0:
                self._emit(
                    "step",
                    step=self.state.step,
                    epoch=self.state.step // self.steps_per_epoch,
                    loss=float(loss.detach()),
                    ce=float(classification.detach()),
                    contrastive=float(contrastive.detach()),
                    grad_norm=norm,
                    lr=rate,
                )
            if first_or_last or self.state.step % self.validate_every == 0:
                report = self.evaluate(self.dev_corpus)
                worst, recall = report.worst_group
                self.summary.history.append(
                    {
                        "step": float(self.state.step),
                        "score": report.score,
                        "macro_f1": report.macro_f1,
                        "accuracy": report.accuracy,
                        "worst_group_recall": recall,
                    }
                )
                self.state.history = self.summary.history
                self._emit(
                    "validate",
                    step=self.state.step,
                    accuracy=report.accuracy,
                    macro_f1=report.macro_f1,
                    worst_group=worst,
                    worst_recall=recall,
                )
                self._checkpoint(report)

        self.summary.seconds_this_run = time.monotonic() - started
        self.state.history = self.summary.history
        self._emit(
            "finish",
            steps_this_run=self.summary.steps_this_run,
            seconds_this_run=self.summary.seconds_this_run,
            rows_per_second=self.summary.rows_per_second,
        )
        return self.summary


def build_corpora(
    data_dir: Path, model_config: ModelConfig, settings: TrainConfig
) -> tuple[Corpus, Corpus]:
    """Train and dev, sharing the group index the training split defines."""
    train_records = read_split(data_dir / "train.jsonl")
    dev_records = read_split(data_dir / "dev.jsonl")
    train = Corpus(
        train_records,
        model_config,
        augmentation=Augmentation(
            phrase_prob=settings.phrase_augment_prob,
            case_prob=settings.case_augment_prob,
            seed=settings.seed,
        ),
    )
    dev = Corpus(dev_records, model_config, groups=train.groups)
    return train, dev


def train_identifier(
    *,
    model_config: ModelConfig,
    settings: TrainConfig,
    data_dir: Path,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device: torch.device | None = None,
    resume: bool = True,
) -> RunSummary:
    """Run training end to end and leave `best.pt` and `latest.pt` behind."""
    resolved = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(settings.seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train, dev = build_corpora(data_dir, model_config, settings)
    log.info(
        "corpus train=%d dev=%d groups=%d classes=%d",
        len(train),
        len(dev),
        len(train.groups),
        len(train.classes),
    )
    model = CharClassifier(model_config).to(resolved)
    trainer = Trainer(model, train, dev, settings, checkpoint_dir, resolved)
    if resume:
        trainer.maybe_resume()
    summary = trainer.run()

    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model": asdict(model_config), "train": asdict(settings)}, indent=2),
        encoding="utf-8",
    )
    if summary.best is not None:
        (checkpoint_dir / "dev-report.json").write_text(
            json.dumps(summary.best.to_dict(), indent=2), encoding="utf-8"
        )
    return summary


def held_out_report(
    model: CharClassifier,
    records: Sequence[Record],
    groups: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> Report:
    """Score a trained model on the held-out set. Run once, after selection."""
    corpus = Corpus(records, model.config, groups=groups)
    model.eval()
    pairs: list[tuple[int, int]] = []
    names: list[str] = []
    with torch.inference_mode():
        for indices in sequential_batches(corpus, batch_size):
            batch = corpus.collate(indices).to(device)
            logits = model(batch.input_ids, pad_mask=batch.pad_mask).logits
            predictions = logits.argmax(dim=-1).cpu().tolist()
            pairs.extend(zip(batch.labels.cpu().tolist(), predictions, strict=True))
            names.extend(corpus.records[index].group for index in indices)
    return build_report(pairs, corpus.classes, names)
