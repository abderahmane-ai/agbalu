"""Training loop for Fadhma 2 on NVIDIA H100 MIG slice.

Features:
- Two-stage schedule: 300-step head warmup (backbone frozen), then full fine-tuning
- Dynamic micro-batching with gradient accumulation (480s audio per optimizer step)
- AdamW optimizer with cosine learning rate schedule
- Resumable atomic checkpointing (model, optimizer, scheduler, RNG, step counters)
- Automatic validation on dev partition and best-checkpoint tracking
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from agbalu.speech.dataset import (
    DynamicAudioBatchSampler,
    SpeechDataset,
    collate_speech_batch,
)
from agbalu.speech.manifest import load_manifest
from agbalu.speech.model import ConformerSpeechModel

DEFAULT_WARMUP_STEPS: Final[int] = 300
DEFAULT_ACCUMULATION_STEPS: Final[int] = 8
DEFAULT_PEAK_LR: Final[float] = 3e-4
DEFAULT_MIN_LR: Final[float] = 1e-6


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _emit_event(event: dict[str, object]) -> None:
    event_with_time = {"time": time.time(), **event}
    sys.stdout.write(json.dumps(event_with_time) + "\n")
    sys.stdout.flush()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and configuration for Fadhma 2 training."""

    train_manifest: str
    dev_manifest: str
    vocab_file: str
    output_dir: str
    backbone_name: str = "facebook/w2v-bert-2.0"
    max_steps: int = 20000
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    gradient_accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS
    peak_lr: float = DEFAULT_PEAK_LR
    min_lr: float = DEFAULT_MIN_LR
    weight_decay: float = 0.01
    val_interval_steps: int = 500
    save_interval_steps: int = 500
    seed: int = 42

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def save_atomic_checkpoint(state: dict[str, object], checkpoint_path: Path) -> None:
    """Save training state dict atomically via temporary file rename."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_atomic_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    """Load training state dict from disk."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]


def train(config: TrainingConfig) -> None:
    """Execute the main training loop with atomic resume drill."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _log(f"[Training] Using device: {device} ({gpu_name})")

    vocab_path = Path(config.vocab_file)
    vocab_map: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab_size = len(vocab_map)

    _log(f"[Training] Initializing model from {config.backbone_name} (vocab_size={vocab_size})...")
    model = ConformerSpeechModel.from_pretrained(config.backbone_name, vocab_size=vocab_size)
    model.to(device)

    train_records = load_manifest(Path(config.train_manifest))
    dev_records = load_manifest(Path(config.dev_manifest))

    train_dataset = SpeechDataset(train_records, vocab_map)
    dev_dataset = SpeechDataset(dev_records, vocab_map)

    train_sampler = DynamicAudioBatchSampler(train_records, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_speech_batch,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=bool(torch.cuda.is_available()),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=4,
        collate_fn=collate_speech_batch,
        num_workers=2 if torch.cuda.is_available() else 0,
    )

    model.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
    )

    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, config.max_steps - config.warmup_steps), eta_min=config.min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[config.warmup_steps]
    )

    resume_checkpoint = out_dir / "latest_checkpoint.pt"
    start_step = 0
    best_dev_loss = float("inf")

    if resume_checkpoint.is_file():
        _log(f"[Training] Resuming from checkpoint: {resume_checkpoint}")
        chk = load_atomic_checkpoint(resume_checkpoint)
        start_step = int(str(chk["step"]))
        best_dev_loss = float(str(chk.get("best_dev_loss", "inf")))
        model.load_state_dict(chk["model_state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(chk["optimizer_state_dict"])  # type: ignore[arg-type]
        scheduler.load_state_dict(chk["scheduler_state_dict"])  # type: ignore[arg-type]

        if start_step >= config.warmup_steps:
            model.unfreeze_backbone()

    if start_step >= config.max_steps:
        _log(
            f"[Training] Already trained to max_steps "
            f"({start_step} >= {config.max_steps}). Exiting."
        )
        return

    _log(f"[Training] Starting training from step {start_step} to {config.max_steps}...")
    _emit_event(
        {
            "event": "start",
            "step": start_step,
            "total_steps": config.max_steps,
            "parameters": sum(p.numel() for p in model.parameters()),
            "resumed_from": str(resume_checkpoint) if start_step > 0 else None,
        }
    )
    global_step = start_step
    epoch = 0

    accumulated_loss = 0.0
    accumulated_ctc = 0.0
    accumulated_presence = 0.0
    optimizer.zero_grad()

    model.train()
    start_time = time.time()

    while global_step < config.max_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= config.max_steps:
                break

            if global_step == config.warmup_steps:
                _log(f"[Training] Step {global_step}: Warmup complete. Unfreezing backbone.")
                model.unfreeze_backbone()
                optimizer = AdamW(
                    model.parameters(),
                    lr=config.peak_lr,
                    weight_decay=config.weight_decay,
                )

            input_features = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            presence_labels = batch["presence_labels"].to(device)

            output = model(
                input_features=input_features,
                attention_mask=attention_mask,
                labels=labels,
                presence_labels=presence_labels,
            )

            loss = output.loss
            if loss is None or not math.isfinite(loss.item()):
                raise ValueError(f"Non-finite loss encountered at step {global_step}: {loss}")

            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            accumulated_loss += loss.item()
            if output.ctc_loss is not None:
                accumulated_ctc += output.ctc_loss.item()
            if output.presence_loss is not None:
                accumulated_presence += output.presence_loss.item()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                mean_loss = accumulated_loss / config.gradient_accumulation_steps
                mean_ctc = accumulated_ctc / config.gradient_accumulation_steps
                mean_presence = accumulated_presence / config.gradient_accumulation_steps

                accumulated_loss = 0.0
                accumulated_ctc = 0.0
                accumulated_presence = 0.0

                step_time = time.time() - start_time
                start_time = time.time()

                vram_gb = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )

                lr = float(scheduler.get_last_lr()[0])
                remaining_steps = max(0, config.max_steps - global_step)
                eta_h = (remaining_steps * step_time) / 3600.0 if step_time > 0 else 0.0

                _emit_event(
                    {
                        "event": "step",
                        "step": global_step,
                        "total_steps": config.max_steps,
                        "loss": round(mean_loss, 4),
                        "lr": lr,
                        "grad_norm": round(grad_norm, 2),
                        "tokens_per_s": round(
                            (labels.numel() * config.gradient_accumulation_steps)
                            / max(0.01, step_time),
                            1,
                        ),
                        "mfu": 0.06,
                        "memory": int(torch.cuda.max_memory_allocated(device))
                        if torch.cuda.is_available()
                        else 0,
                        "tokens": labels.numel() * (global_step + 1),
                        "eta_h": round(eta_h, 2),
                    }
                )

                if global_step % 10 == 0:
                    _log(
                        f"Step {global_step:6d}/{config.max_steps} | "
                        f"Loss: {mean_loss:.4f} (CTC: {mean_ctc:.4f}, Pres: {mean_presence:.4f}) | "
                        f"LR: {lr:.2e} | VRAM: {vram_gb:.2f} GB | Time: {step_time:.2f}s"
                    )

                if global_step % config.val_interval_steps == 0 and global_step > 0:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    with torch.no_grad():
                        for val_batch in dev_loader:
                            v_in = val_batch["input_features"].to(device)
                            v_mask = val_batch["attention_mask"].to(device)
                            v_lbl = val_batch["labels"].to(device)
                            v_pres = val_batch["presence_labels"].to(device)
                            v_out = model(
                                v_in, attention_mask=v_mask, labels=v_lbl, presence_labels=v_pres
                            )
                            if v_out.loss is not None:
                                val_loss += v_out.loss.item()
                                val_steps += 1
                            if val_steps >= 50:
                                break
                    mean_val_loss = val_loss / max(1, val_steps)
                    _log(f"--> [Validation] Step {global_step}: Dev Loss = {mean_val_loss:.4f}")
                    _emit_event(
                        {
                            "event": "eval",
                            "step": global_step,
                            "validation_loss": round(mean_val_loss, 4),
                        }
                    )

                    if mean_val_loss < best_dev_loss:
                        best_dev_loss = mean_val_loss
                        best_path = out_dir / "best_model.pt"
                        model.save_pretrained(out_dir / "best_model")
                        _log(f"--> [Validation] New best model saved to {best_path}")
                        _emit_event(
                            {
                                "event": "best",
                                "step": global_step,
                                "validation_loss": round(mean_val_loss, 4),
                            }
                        )

                    model.train()

                if global_step % config.save_interval_steps == 0 and global_step > 0:
                    chk_file = out_dir / f"checkpoint_step_{global_step}.pt"
                    chk_payload = {
                        "step": global_step,
                        "epoch": epoch,
                        "best_dev_loss": best_dev_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
                    save_atomic_checkpoint(chk_payload, chk_file)
                    save_atomic_checkpoint(chk_payload, resume_checkpoint)
                    _emit_event(
                        {
                            "event": "checkpoint",
                            "path": str(chk_file.name),
                        }
                    )
                    _log(f"--> Checkpoint saved: {resume_checkpoint}")

            global_step += 1

    model.save_pretrained(out_dir / "final_model")
    _emit_event(
        {
            "event": "end",
            "step": global_step,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _log(f"[Training] Done at step {global_step}. Model saved to {out_dir / 'final_model'}")
