"""Optimized training loop and atomic checkpointing for Feraoun-Pro.

Two resume paths, told apart by whether the checkpoint sits under this run's own `save_dir`:

* **Same run** — a preempted container restarting. Step counter, epoch and the best-CER
  trackers continue where the checkpoint left off, and the schedule is fast-forwarded to it.
* **Cross-run warm start** — continuing a different lineage, which is what adding Neo-Tifinagh
  to a Latin-trained checkpoint is. Weights and optimizer moments transplant; the step counter
  and cosine schedule start fresh against *this* call's `total_steps`. Carrying an absolute
  step count from a schedule sized for another corpus lands past `total_steps` and trips the
  already-done guard on a run that has trained nothing here.

Every evaluation buckets by script, because a pooled CER hides a Latin regression behind a
falling Tifinagh error the moment Tifinagh enters the mix — the two move in opposite
directions and their sum can improve while half the model gets worse. `best.pt` is gated on
both: a new best needs Tifinagh to improve *and* Latin to stay within
`LATIN_REGRESSION_TOLERANCE` of the baseline measured at resume.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from agbalu.model.checkpoint import BEST, LATEST, TrainingState, load, save
from agbalu.ocr.dataset_pro import (
    BatchOCRPro,
    BucketedBatchSampler,
    CollateOCRPro,
    SyntheticOCRDatasetPro,
)
from agbalu.ocr.model_pro import HybridCTCRecognizer
from agbalu.ocr.synthetic_pro import is_tifinagh_text
from agbalu.ocr.vocabulary_pro import decode_ctc

log = logging.getLogger(__name__)


class BatchLoader(Protocol):
    """What an evaluation needs from a loader, which is not what `DataLoader[T]` says.

    `T` is the *dataset element* type, and a `collate_fn` is exactly the thing that makes
    the iteration type differ from it — here `tuple[Image, str]` in, `BatchOCRPro` out.
    Stating the two methods that are used is more honest than casting the loader.
    """

    def __iter__(self) -> Iterator[BatchOCRPro]: ...

    def __len__(self) -> int: ...


FAST_EVAL_SAMPLES: Final[int] = 1000

LATIN_REGRESSION_TOLERANCE: Final[float] = 1.01
"""Relative Latin CER a new `best.pt` may cost against the resume baseline. Both evaluations
are deterministic over the same seeded renders, so the only difference between them is the
model — 1% leaves room for float drift and nothing else."""


class TrainingSetupError(RuntimeError):
    """A configuration that cannot produce the measurement it claims."""


@dataclass(slots=True)
class CheckpointGate:
    """Decides whether an evaluation earns `best.pt`.

    In a dual-script run the balanced error `0.5 * latin_cer + 0.5 * tifinagh_cer` must
    improve **and** neither script may pass its own ceiling — the resume baseline times the
    tolerance, or the best seen where no baseline was measured. A single-script run has no
    trade-off to police and is decided on the pooled error alone.
    """

    baseline_latin_cer: float | None = None
    baseline_tifinagh_cer: float | None = None
    best_latin_cer: float | None = None
    best_tifinagh_cer: float | None = None
    best_combined_cer: float = float("inf")
    best_pooled_cer: float = float("inf")
    tolerance: float = 1.15

    @property
    def latin_ceiling(self) -> float | None:
        if self.baseline_latin_cer is None:
            return None
        return self.baseline_latin_cer * self.tolerance

    @property
    def tifinagh_ceiling(self) -> float | None:
        reference = self.baseline_tifinagh_cer
        if reference is None:
            reference = self.best_tifinagh_cer
        return None if reference is None else reference * self.tolerance

    def latin_held(self, latin_cer: float | None) -> bool:
        ceiling = self.latin_ceiling
        return ceiling is None or latin_cer is None or latin_cer <= ceiling

    def tifinagh_held(self, tifinagh_cer: float | None) -> bool:
        ceiling = self.tifinagh_ceiling
        return ceiling is None or tifinagh_cer is None or tifinagh_cer <= ceiling

    def accepts(
        self, pooled_cer: float | None, latin_cer: float | None, tifinagh_cer: float | None
    ) -> bool:
        if latin_cer is not None and tifinagh_cer is not None:
            combined = 0.5 * latin_cer + 0.5 * tifinagh_cer
            improved = combined < self.best_combined_cer
            return improved and self.latin_held(latin_cer) and self.tifinagh_held(tifinagh_cer)

        # The per-script bests cannot decide here: a resume restores only the pooled one.
        return pooled_cer is not None and pooled_cer < self.best_pooled_cer

    def record(
        self, pooled_cer: float | None, latin_cer: float | None, tifinagh_cer: float | None
    ) -> None:
        if latin_cer is not None and tifinagh_cer is not None:
            combined = 0.5 * latin_cer + 0.5 * tifinagh_cer
            self.best_combined_cer = min(self.best_combined_cer, combined)
        if tifinagh_cer is not None:
            self.best_tifinagh_cer = (
                tifinagh_cer
                if self.best_tifinagh_cer is None
                else min(self.best_tifinagh_cer, tifinagh_cer)
            )
        if latin_cer is not None:
            self.best_latin_cer = (
                latin_cer if self.best_latin_cer is None else min(self.best_latin_cer, latin_cer)
            )
        if pooled_cer is not None:
            self.best_pooled_cer = min(self.best_pooled_cer, pooled_cer)


@dataclass(frozen=True, slots=True)
class TrainingConfigOCRPro:
    epochs: int = 6
    batch_size: int = 64
    lr: float = 2e-4
    min_lr_ratio: float = 0.05
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    log_every: int = 20
    eval_every: int = 500
    save_dir: Path = Path("artifacts/runs/feraoun-pro")


def build_cosine_schedule_with_warmup(
    optimizer: AdamW, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.05
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _rate(errors: int, total: int) -> float | None:
    """None where there is no support. A clamped 0.0 reads as a perfect score."""
    return errors / total if total else None


def _format_rate(value: float | None) -> str:
    return f"{value:.5f}" if value is not None else "n/a"


def evaluate_pro(
    model: HybridCTCRecognizer,
    val_loader: BatchLoader,
    device: torch.device,
    max_batches: int | None = None,
    log_samples: int = 3,
) -> dict[str, float | int | None]:
    """CER, WER and exact match, pooled and split by script, with prediction previews.

    `max_batches` truncates a length-sorted, unshuffled loader, so a fast evaluation is
    whichever lines sort shortest and its script balance is not the split's. It is a liveness
    signal; only the full pass carries a number worth acting on.
    """
    model.eval()
    total_loss = 0.0
    buckets: Final[tuple[str, ...]] = ("all", "latin", "tifinagh")
    counts: dict[str, dict[str, int]] = {
        name: {"chars": 0, "char_errors": 0, "words": 0, "word_errors": 0, "lines": 0, "exact": 0}
        for name in buckets
    }
    logged = {"latin": 0, "tifinagh": 0}

    import editdistance

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            pixel_values = batch.pixel_values.to(device)
            input_lengths = batch.input_lengths.to(device)
            ctc_targets = batch.ctc_targets.to(device)
            target_lengths = batch.target_lengths.to(device)

            output = model(
                pixel_values=pixel_values,
                input_lengths=input_lengths,
                ctc_targets=ctc_targets,
                target_lengths=target_lengths,
            )

            if output.loss is not None:
                total_loss += output.loss.item()

            preds = torch.argmax(output.ctc_logits, dim=-1).cpu().tolist()
            for pred_ids, gold_text in zip(preds, batch.texts, strict=False):
                hyp_text = decode_ctc(pred_ids)
                script = "tifinagh" if is_tifinagh_text(gold_text) else "latin"

                char_errors = editdistance.eval(hyp_text, gold_text)
                gold_words = gold_text.split()
                word_errors = editdistance.eval(hyp_text.split(), gold_words)
                exact = hyp_text == gold_text

                for name in ("all", script):
                    counts[name]["char_errors"] += char_errors
                    counts[name]["chars"] += len(gold_text)
                    counts[name]["word_errors"] += word_errors
                    counts[name]["words"] += len(gold_words)
                    counts[name]["lines"] += 1
                    counts[name]["exact"] += int(exact)

                if logged[script] < log_samples:
                    status = "MATCH" if exact else f"ED={char_errors}"
                    log.info("  [%s %s] gold: %r", script.upper(), status, gold_text)
                    log.info("                 pred: %r", hyp_text)
                    logged[script] += 1

    evaluated_batches = max_batches if max_batches is not None else len(val_loader)
    metrics: dict[str, float | int | None] = {"loss": total_loss / max(1, evaluated_batches)}
    for name in buckets:
        prefix = "" if name == "all" else f"{name}_"
        bucket = counts[name]
        metrics[f"{prefix}lines"] = bucket["lines"]
        metrics[f"{prefix}cer"] = _rate(bucket["char_errors"], bucket["chars"])
        metrics[f"{prefix}wer"] = _rate(bucket["word_errors"], bucket["words"])
        metrics[f"{prefix}exact_match"] = _rate(bucket["exact"], bucket["lines"])
    return metrics


def log_metrics(tag: str, metrics: dict[str, float | int | None]) -> None:
    """One line per evaluation, carrying each script's support beside its rate.

    The support is not decoration: a rate over a bucket of zero lines is what an
    all-Latin validation split reports for Tifinagh, and it reads as a result.
    """
    loss = metrics["loss"]
    log.info(
        "%-22s loss=%s cer=%s wer=%s exact=%s | latin cer=%s n=%s | tifinagh cer=%s n=%s",
        tag,
        f"{loss:.4f}" if isinstance(loss, float) else "n/a",
        _format_rate(metrics["cer"] if isinstance(metrics["cer"], float) else None),
        _format_rate(metrics["wer"] if isinstance(metrics["wer"], float) else None),
        _format_rate(metrics["exact_match"] if isinstance(metrics["exact_match"], float) else None),
        _format_rate(metrics["latin_cer"] if isinstance(metrics["latin_cer"], float) else None),
        metrics["latin_lines"],
        _format_rate(
            metrics["tifinagh_cer"] if isinstance(metrics["tifinagh_cer"], float) else None
        ),
        metrics["tifinagh_lines"],
    )


def _as_float(value: float | int | None) -> float | None:
    return value if isinstance(value, float) else None


def _results(
    gate: CheckpointGate,
    total_steps: int,
    steps_this_run: int,
    non_finite_steps: int,
    elapsed: float,
) -> dict[str, float | int | None]:
    """`*_this_run` beside the cumulative counter: a resumed run that divides the
    checkpoint's totals by this process's clock credits it with the previous run's work."""
    return {
        "total_steps": total_steps,
        "steps_this_run": steps_this_run,
        "best_cer": gate.best_pooled_cer if math.isfinite(gate.best_pooled_cer) else None,
        "best_latin_cer": gate.best_latin_cer,
        "best_tifinagh_cer": gate.best_tifinagh_cer,
        "baseline_latin_cer": gate.baseline_latin_cer,
        "baseline_tifinagh_cer": gate.baseline_tifinagh_cer,
        "non_finite_steps": non_finite_steps,
        "total_time": elapsed,
    }


def train_feraoun_pro(
    model: HybridCTCRecognizer,
    train_texts: list[str],
    val_texts: list[str],
    config: TrainingConfigOCRPro,
    device: torch.device,
    resume_from: Path | None = None,
) -> dict[str, float | int | None]:
    """Train Feraoun-Pro, gating `best.pt` on Tifinagh gain without Latin loss."""
    if not train_texts:
        message = "no training lines"
        raise TrainingSetupError(message)
    if not val_texts:
        message = "no validation lines; nothing would gate the checkpoint"
        raise TrainingSetupError(message)

    model.to(device)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SyntheticOCRDatasetPro(train_texts, augment=True)
    val_dataset = SyntheticOCRDatasetPro(val_texts, augment=False)

    collate_fn = CollateOCRPro()
    train_sampler = BucketedBatchSampler(train_texts, batch_size=config.batch_size, shuffle=True)
    val_sampler = BucketedBatchSampler(val_texts, batch_size=config.batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight", "BatchNorm2d.weight", "pos_encoding"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
    scheduler = build_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, config.min_lr_ratio
    )

    global_step = 0
    start_epoch = 0
    gate = CheckpointGate()
    non_finite_steps = 0

    if resume_from is not None:
        same_run = resume_from.parent == config.save_dir
        state = load(
            resume_from.parent, model, optimizer, name=resume_from.name, restore_rng=same_run
        )
        log.info(
            "RESUME       from=%s step=%d epoch=%d same_run=%s",
            resume_from,
            state.step,
            state.epoch,
            same_run,
        )
        if same_run:
            global_step = state.step
            start_epoch = global_step // steps_per_epoch
            gate.best_pooled_cer = state.best_validation_loss
            for _ in range(global_step):
                scheduler.step()

        # Measured here, not read off the checkpoint: the split, the mix and the renderer all
        # differ from the run that wrote it, so its stored number describes another
        # population. The gate compares two evaluations of this one.
        baseline = evaluate_pro(model, val_loader, device, max_batches=None)
        log_metrics("BASELINE", baseline)
        gate.baseline_latin_cer = _as_float(baseline["latin_cer"])
        gate.baseline_tifinagh_cer = _as_float(baseline["tifinagh_cer"])
        gate.best_latin_cer = gate.baseline_latin_cer
        gate.best_tifinagh_cer = gate.baseline_tifinagh_cer

    if global_step >= total_steps:
        # Two different situations reach here and they have different remedies, so the log
        # names which one this is. A checkpoint carrying more steps than this run's whole
        # schedule is not a finished run — it is another schedule's counter, which is what
        # continuing a different lineage under its own run name looks like.
        log.info(
            "NOOP         checkpoint is at step %d and this run's schedule is %d steps; "
            "nothing trained. To continue the same run, raise epochs. To warm-start a new "
            "lineage from it, give this run its own name so the schedule restarts and the "
            "source checkpoint is left intact.",
            global_step,
            total_steps,
        )
        return _results(gate, global_step, steps_this_run=0, non_finite_steps=0, elapsed=0.0)

    steps_at_start = global_step
    start_time = time.time()
    batches_per_fast_eval = max(1, FAST_EVAL_SAMPLES // config.batch_size)
    batches_to_skip = global_step % steps_per_epoch

    for epoch in range(start_epoch, config.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue
            step_start = time.time()
            global_step += 1
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_lengths = batch.input_lengths.to(device, non_blocking=True)
            ctc_targets = batch.ctc_targets.to(device, non_blocking=True)
            target_lengths = batch.target_lengths.to(device, non_blocking=True)
            aux_input_ids = batch.aux_input_ids.to(device, non_blocking=True)
            aux_labels = batch.aux_labels.to(device, non_blocking=True)

            output = model(
                pixel_values=pixel_values,
                input_lengths=input_lengths,
                ctc_targets=ctc_targets,
                target_lengths=target_lengths,
                aux_input_ids=aux_input_ids,
                aux_labels=aux_labels,
            )

            loss = output.loss
            if loss is None:
                continue

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            # The gradient, before the step: a check after `optimizer.step()` reports damage
            # already written into the weights. The norm has to reach the host to be
            # compared, so this is the one synchronisation worth paying for.
            if not math.isfinite(float(norm)):
                non_finite_steps += 1
                log.warning("NON_FINITE   step=%d gradient norm; step skipped", global_step)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_ms = (time.time() - step_start) * 1000.0

            if global_step % config.log_every == 0 or global_step in (
                steps_at_start + 1,
                total_steps,
            ):
                ctc_val = f"{output.ctc_loss.item():.4f}" if output.ctc_loss is not None else "n/a"
                aux_val = f"{output.aux_loss.item():.4f}" if output.aux_loss is not None else "n/a"
                log.info(
                    "STEP         %d/%d epoch=%d loss=%.4f ctc=%s aux=%s lr=%.2e ms_per_step=%.1f",
                    global_step,
                    total_steps,
                    epoch + 1,
                    loss.item(),
                    ctc_val,
                    aux_val,
                    scheduler.get_last_lr()[0],
                    step_ms,
                )

            if global_step % config.eval_every == 0:
                fast = evaluate_pro(model, val_loader, device, max_batches=batches_per_fast_eval)
                log_metrics(f"FAST step={global_step}", fast)
                # `latest.pt` every evaluation, so a preemption costs at most `eval_every`
                # steps. It is not gated: it is the resume point, not the release.
                save(
                    config.save_dir,
                    model,
                    optimizer,
                    TrainingState(
                        step=global_step, epoch=epoch, best_validation_loss=gate.best_pooled_cer
                    ),
                    name=LATEST,
                )
                model.train()

        # The full split gates `best.pt`. A length-sorted fast-eval prefix can hold almost
        # none of one script, and a checkpoint chosen on it is chosen on the other.
        epoch_metrics = evaluate_pro(model, val_loader, device, max_batches=None)
        log_metrics(f"EPOCH {epoch + 1}", epoch_metrics)

        pooled_cer = _as_float(epoch_metrics["cer"])
        latin_cer = _as_float(epoch_metrics["latin_cer"])
        tifinagh_cer = _as_float(epoch_metrics["tifinagh_cer"])

        if not gate.latin_held(latin_cer):
            log.warning(
                "LATIN_REGRESSION latin_cer=%s ceiling=%s baseline=%s; best.pt held back",
                _format_rate(latin_cer),
                _format_rate(gate.latin_ceiling),
                _format_rate(gate.baseline_latin_cer),
            )

        if gate.accepts(pooled_cer, latin_cer, tifinagh_cer):
            gate.record(pooled_cer, latin_cer, tifinagh_cer)
            save(
                config.save_dir,
                model,
                optimizer,
                TrainingState(
                    step=global_step, epoch=epoch, best_validation_loss=gate.best_pooled_cer
                ),
                name=BEST,
            )
            log.info(
                "NEW_BEST     step=%d epoch=%d latin_cer=%s tifinagh_cer=%s",
                global_step,
                epoch + 1,
                _format_rate(latin_cer),
                _format_rate(tifinagh_cer),
            )
        model.train()

    return _results(
        gate,
        total_steps=global_step,
        steps_this_run=global_step - steps_at_start,
        non_finite_steps=non_finite_steps,
        elapsed=time.time() - start_time,
    )
