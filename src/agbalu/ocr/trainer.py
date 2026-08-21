"""Training harness and atomic checkpointing for document OCR."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from agbalu.ocr.config import TrainConfig
from agbalu.ocr.evaluate import evaluate_ocr_model
from agbalu.ocr.models import VisionEncoderDecoder

log: Final = logging.getLogger("agbalu.ocr.trainer")

CHECKPOINT_KEYS: Final[tuple[str, ...]] = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "step",
    "epoch",
    "best_cer",
    "config",
)
"""What `save_checkpoint` writes and `Trainer.fit` requires back to resume. Checked on load:
a checkpoint missing any of them would resume as a fresh run reporting the checkpoint's own
counters as this run's measurement."""


class ResumeError(RuntimeError):
    """A checkpoint that cannot continue the run it claims to."""


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        warmup_delta = num_training_steps - num_warmup_steps
        progress = float(current_step - num_warmup_steps) / float(max(1, warmup_delta))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: VisionEncoderDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    best_cer: float,
    path: Path,
) -> None:
    """Write resumable state through a temporary file, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_cer": best_cer,
        "config": model.config,
    }

    torch.save(payload, temp_path)
    temp_path.replace(path)


class Trainer:
    """The training loop, its evaluations, and the checkpoints a preemption resumes from."""

    def __init__(
        self,
        model: VisionEncoderDecoder,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: TrainConfig,
        on_checkpoint: Callable[[Path], None] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        # Called after every checkpoint lands. On Modal this commits the volume: without
        # it a preempted run loses every intermediate save, because the container's
        # filesystem is discarded and only a commit persists.
        self.on_checkpoint = on_checkpoint

    def _save(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        step: int,
        epoch: int,
        best_cer: float,
        name: str,
    ) -> Path:
        path = self.config.output_dir / name
        save_checkpoint(self.model, optimizer, scheduler, step, epoch, best_cer, path)
        if self.on_checkpoint is not None:
            self.on_checkpoint(path)
        return path

    def _resume(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> tuple[int, int, float]:
        """Restore model, optimizer, scheduler and counters. Returns (step, epoch, best_cer).

        `strict=True` on the weights: a checkpoint whose vocabulary or depth differs from
        this config must fail here rather than load nothing and train from random init
        behind a healthy-looking curve.
        """
        source = self.config.resume_checkpoint
        if source is None or not Path(source).is_file():
            return 0, 0, float("inf")

        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
        if missing:
            message = f"{source} is missing {missing}; it cannot resume a run"
            raise ResumeError(message)

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint["step"])
        epoch = int(checkpoint["epoch"])
        best_cer = float(checkpoint["best_cer"])
        log.info(
            "Resumed %s at step %d, epoch %d, best CER %.4f",
            source,
            step,
            epoch,
            best_cer,
        )
        return step, epoch, best_cer

    def fit(self) -> dict[str, Any]:
        """Execute training loop with cosine decay schedule and CER evaluation."""
        device = torch.device(self.config.device)
        self.model.to(device)

        no_decay = ("bias", "norm")
        decay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ]
        no_decay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)
        ]

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
        )

        steps_per_epoch = len(self.train_loader)
        total_steps = self.config.max_steps or (steps_per_epoch * self.config.epochs)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        global_step, start_epoch, best_cer = self._resume(optimizer, scheduler)
        steps_at_start = global_step

        if global_step >= total_steps:
            log.info(
                "Already at %d of %d steps; nothing to train. Raise epochs to continue.",
                global_step,
                total_steps,
            )
            return {
                "total_steps": global_step,
                "steps_this_run": 0,
                "best_cer": best_cer if math.isfinite(best_cer) else None,
                "output_dir": str(self.config.output_dir),
            }

        log.info(
            "Starting OCR training on %s (Total Steps: %d, Epochs: %d, Batch Size: %d)",
            device,
            total_steps,
            self.config.epochs,
            self.config.batch_size,
        )

        start_time = time.time()
        running_loss = 0.0
        running_lines = 0
        logged_loss: float | None = None
        non_finite_steps = 0

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_log_path = self.config.output_dir / "metrics.jsonl"

        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            self.model.train()
            for batch in self.train_loader:
                global_step += 1
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad(set_to_none=True)
                output = self.model(pixel_values=pixel_values, labels=labels)

                loss = output.loss
                if loss is None:
                    continue

                loss.backward()
                norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )
                # The gradient, not the loss, and before the step: a check that runs after
                # `optimizer.step()` reports damage it has already done to the weights. The
                # norm has to reach the host to be compared, so this is the one sync worth
                # paying for.
                if not math.isfinite(float(norm)):
                    non_finite_steps += 1
                    log.warning("Step %d: non-finite gradient norm, step skipped", global_step)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                running_lines += pixel_values.size(0)

                if global_step % self.config.log_every_steps == 0:
                    elapsed = time.time() - start_time
                    throughput = running_lines / max(elapsed, 1e-6)
                    logged_loss = running_loss / self.config.log_every_steps
                    current_lr = scheduler.get_last_lr()[0]

                    log.info(
                        "Epoch %d | Step %d/%d (%.1f%%) | Loss: %.4f | LR: %.2e | %.1f lines/s",
                        epoch,
                        global_step,
                        total_steps,
                        100.0 * global_step / max(total_steps, 1),
                        logged_loss,
                        current_lr,
                        throughput,
                    )

                    running_loss = 0.0
                    running_lines = 0
                    start_time = time.time()

                if global_step % self.config.eval_every_steps == 0:
                    eval_metrics = evaluate_ocr_model(
                        self.model,
                        self.val_loader,
                        device=device,
                        max_batches=self.config.eval_batches,
                    )
                    cer = eval_metrics["cer"]
                    wer = eval_metrics["wer"]
                    em = eval_metrics["exact_match"]
                    val_loss = eval_metrics["val_loss"]
                    f1 = eval_metrics["diacritic_f1"]
                    p = eval_metrics["diacritic_precision"]
                    r = eval_metrics["diacritic_recall"]

                    log.info(
                        "[EVAL] Step %d: CER: %.4f | WER: %.4f | EM: %.2f%% | "
                        "Val Loss: %.4f | Diacritic F1: %.4f (P: %.4f, R: %.4f) over %d lines",
                        global_step,
                        cer,
                        wer,
                        em * 100.0,
                        val_loss,
                        f1,
                        p,
                        r,
                        eval_metrics["num_evaluated_lines"],
                    )

                    samples = eval_metrics.get("samples", [])
                    for s_idx, (ref, hyp) in enumerate(samples[:2], 1):
                        log.info("  [SAMPLE %d] Ref : %s", s_idx, ref)
                        log.info("  [SAMPLE %d] Pred: %s", s_idx, hyp)

                    with metrics_log_path.open("a", encoding="utf-8") as f:
                        log_entry = {
                            "timestamp": time.time(),
                            "step": global_step,
                            "epoch": epoch,
                            # The last logged window, which is up to `log_every_steps`
                            # behind this eval. `None` until the first window closes,
                            # because 0.0 would read as a converged run.
                            "train_loss": logged_loss,
                            "val_loss": val_loss,
                            "cer": cer,
                            "wer": wer,
                            "exact_match": em,
                            "diacritic_precision": p,
                            "diacritic_recall": r,
                            "diacritic_f1": f1,
                            "evaluated_lines": eval_metrics["num_evaluated_lines"],
                            "lr": scheduler.get_last_lr()[0],
                        }
                        f.write(json.dumps(log_entry) + "\n")

                    if cer < best_cer:
                        best_cer = cer
                        path = self._save(
                            optimizer, scheduler, global_step, epoch, best_cer, "best.pt"
                        )
                        log.info(
                            "New best: %s (step %d, CER %.4f, EM %.2f%%)",
                            path,
                            global_step,
                            best_cer,
                            em * 100.0,
                        )
                    self.model.train()

                if global_step % self.config.save_every_steps == 0:
                    self._save(optimizer, scheduler, global_step, epoch, best_cer, "latest.pt")

                if self.config.max_steps and global_step >= self.config.max_steps:
                    break

            if self.config.max_steps and global_step >= self.config.max_steps:
                break

        final = self._save(
            optimizer, scheduler, global_step, self.config.epochs, best_cer, "final.pt"
        )
        log.info("Final checkpoint %s at step %d, best CER %.4f", final, global_step, best_cer)

        return {
            "total_steps": global_step,
            "steps_this_run": global_step - steps_at_start,
            "non_finite_steps": non_finite_steps,
            "best_cer": best_cer if math.isfinite(best_cer) else None,
            "output_dir": str(self.config.output_dir),
        }
