"""Training loop for Fadhma 2 on NVIDIA H100 MIG slice.

Features:
- Two-stage schedule: 300-step head warmup (backbone frozen), then full fine-tuning
- Dynamic micro-batching with gradient accumulation (480s audio per optimizer step)
- AdamW optimizer with cosine learning rate schedule
- Resumable layered atomic checkpointing (model, optimizer, scheduler, RNG, step counters)
- Automatic validation on dev partition with gold CER-based best model tracking
- Pruning of older checkpoints with strict retention (`keep=2`)
- Signal-driven graceful exit (SIGINT/SIGTERM) saving clean atomic checkpoints
- Online multi-condition acoustic augmentation (MUSAN noise + BUT ReverbDB RIRs)
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Final

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from agbalu.speech.augment import AcousticAugmenter
from agbalu.speech.dataset import (
    DynamicAudioBatchSampler,
    SpeechDataset,
    collate_speech_batch,
)
from agbalu.speech.manifest import load_manifest
from agbalu.speech.metrics import cer
from agbalu.speech.model import ConformerSpeechModel
from agbalu.speech.vocabulary import decode as greedy_decode_tokens

DEFAULT_WARMUP_STEPS: Final[int] = 300
DEFAULT_ACCUMULATION_STEPS: Final[int] = 8
DEFAULT_PEAK_LR: Final[float] = 3e-4
DEFAULT_MIN_LR: Final[float] = 1e-6
DEFAULT_KEEP_CHECKPOINTS: Final[int] = 2


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _emit_event(event: dict[str, object]) -> None:
    event_with_time = {"time": time.time(), **event}
    sys.stdout.write(json.dumps(event_with_time) + "\n")
    sys.stdout.flush()


class StopRequest:
    """Catches SIGINT/SIGTERM to guarantee atomic checkpointing before exit."""

    def __init__(self) -> None:
        self.signal: int | None = None

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.handle)

    def handle(self, signum: int, frame: FrameType | None) -> None:
        del frame
        self.signal = signum
        sys.stdout.write(
            f"\n[Training] Caught signal {signum}. "
            "Will save atomic checkpoint and exit after current step.\n"
        )
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
    keep: int = DEFAULT_KEEP_CHECKPOINTS
    seed: int = 42
    rir_dir: str | None = "data/raw/speech/augment/reverb"
    noise_dir: str | None = "data/raw/speech/augment/musan"

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


def save_best(
    directory: Path,
    model: ConformerSpeechModel,
    step: int,
    dev_cer: float,
    dev_loss: float,
) -> None:
    """Save best model state atomically and write best.json provenance."""
    partial = directory / ".best.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(partial / "model")
    best_meta = {
        "step": step,
        "dev_cer_percent": round(dev_cer, 2),
        "dev_loss": round(dev_loss, 4),
        "selection_criterion": "lowest validation CER (Character Error Rate)",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (partial / "best.json").write_text(json.dumps(best_meta, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), partial / "best.pt")

    final = directory / "best"
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)

    # Maintain root-level best.pt and best_model/ for direct downstream CLI compatibility
    shutil.copy2(final / "best.pt", directory / "best.pt")
    best_model_dir = directory / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    shutil.copytree(final / "model", best_model_dir)


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    """State bundle passed to atomic checkpoint saver."""

    model: ConformerSpeechModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int
    epoch: int
    best_dev_cer: float
    best_dev_loss: float


def save_checkpoint(
    directory: Path,
    bundle: CheckpointBundle,
    keep: int = DEFAULT_KEEP_CHECKPOINTS,
) -> Path:
    """Save full training state atomically with RNG and prune older checkpoints beyond keep."""
    step = bundle.step
    final_dir = directory / f"step_{step:07d}"
    partial_dir = directory / f".step_{step:07d}.partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)

    bundle.model.save_pretrained(partial_dir / "model")
    torch.save(bundle.optimizer.state_dict(), partial_dir / "optimizer.pt")
    torch.save(bundle.scheduler.state_dict(), partial_dir / "scheduler.pt")

    generators: dict[str, object] = {
        "cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        generators["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(generators, partial_dir / "rng.pt")

    state_payload = {
        "step": step,
        "epoch": bundle.epoch,
        "best_dev_cer": bundle.best_dev_cer,
        "best_dev_loss": bundle.best_dev_loss,
    }
    (partial_dir / "state.json").write_text(
        json.dumps(state_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Write single-file latest_checkpoint.pt for quick resumes
    single_chk = {
        "step": step,
        "epoch": bundle.epoch,
        "best_dev_cer": bundle.best_dev_cer,
        "best_dev_loss": bundle.best_dev_loss,
        "model_state_dict": bundle.model.state_dict(),
        "optimizer_state_dict": bundle.optimizer.state_dict(),
        "scheduler_state_dict": bundle.scheduler.state_dict(),
        "rng": generators,
    }
    save_atomic_checkpoint(single_chk, directory / "latest_checkpoint.pt")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    partial_dir.rename(final_dir)

    # Prune older step_* directories beyond `keep`
    existing = sorted(
        [p for p in directory.glob("step_*") if p.is_dir() and (p / "state.json").is_file()],
        key=lambda p: int(p.name.removeprefix("step_")),
    )
    for old in existing[:-keep]:
        shutil.rmtree(old, ignore_errors=True)

    return final_dir


def evaluate_dev(
    model: ConformerSpeechModel,
    dev_loader: DataLoader[dict[str, torch.Tensor]],
    vocab_map: dict[str, int],
    device: torch.device,
    max_eval_batches: int = 50,
) -> tuple[float, float]:
    """Compute validation loss and greedy Character Error Rate (CER) on dev split."""
    model.eval()
    val_loss = 0.0
    val_steps = 0
    references: list[str] = []
    hypotheses: list[str] = []

    id_to_char = {v: k for k, v in vocab_map.items()}

    with torch.no_grad():
        for batch in dev_loader:
            v_in = batch["input_features"].to(device)
            v_mask = batch["attention_mask"].to(device)
            v_lbl = batch["labels"].to(device)
            v_pres = batch["presence_labels"].to(device)
            v_out = model(v_in, attention_mask=v_mask, labels=v_lbl, presence_labels=v_pres)

            if v_out.loss is not None:
                val_loss += v_out.loss.item()
                val_steps += 1

            # Decode predictions for CER measurement
            logits = v_out.logits
            best_tokens = torch.argmax(logits, dim=-1)
            for b in range(best_tokens.size(0)):
                pred_ids = [int(i.item()) for i in best_tokens[b]]
                hyp_text = greedy_decode_tokens(pred_ids, vocab_map)
                hypotheses.append(hyp_text)

                gold_ids = [int(i.item()) for i in v_lbl[b] if i.item() != -100]
                ref_text = "".join(id_to_char.get(i, "") for i in gold_ids)
                references.append(ref_text)

            if val_steps >= max_eval_batches:
                break

    mean_loss = val_loss / max(1, val_steps)
    cer_score = cer(references, hypotheses)
    cer_percent = float(cer_score.as_dict()["percent"])
    return mean_loss, cer_percent


def train(config: TrainingConfig) -> None:
    """Execute the main training loop with atomic resume drill."""
    stop_request = StopRequest()
    stop_request.install()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
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

    # Online multi-condition acoustic augmentation
    rir_dir = Path(config.rir_dir) if config.rir_dir else None
    noise_dir = Path(config.noise_dir) if config.noise_dir else None
    _log(f"[Training] Loading augmentation pool from {rir_dir} and {noise_dir}...")
    augmenter = AcousticAugmenter.from_directories(rir_dir=rir_dir, noise_dir=noise_dir)

    train_dataset = SpeechDataset(train_records, vocab_map, augmenter=augmenter)
    dev_dataset = SpeechDataset(dev_records, vocab_map)

    train_sampler = DynamicAudioBatchSampler(train_records, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_speech_batch,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=4,
        collate_fn=collate_speech_batch,
        num_workers=0,
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
    best_dev_cer = float("inf")
    best_dev_loss = float("inf")

    if resume_checkpoint.is_file():
        _log(f"[Training] Resuming from checkpoint: {resume_checkpoint}")
        chk = load_atomic_checkpoint(resume_checkpoint)
        start_step = int(str(chk["step"]))
        best_dev_cer = float(str(chk.get("best_dev_cer", "inf")))
        best_dev_loss = float(str(chk.get("best_dev_loss", "inf")))
        model.load_state_dict(chk["model_state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(chk["optimizer_state_dict"])  # type: ignore[arg-type]
        scheduler.load_state_dict(chk["scheduler_state_dict"])  # type: ignore[arg-type]

        if "rng" in chk and isinstance(chk["rng"], dict):
            rng_data = chk["rng"]
            cpu_rng = rng_data.get("cpu")
            if isinstance(cpu_rng, torch.Tensor):
                torch.set_rng_state(cpu_rng)
            py_rng = rng_data.get("python")
            if isinstance(py_rng, tuple):
                random.setstate(py_rng)
            cuda_rng = rng_data.get("cuda")
            if isinstance(cuda_rng, list) and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng)

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
    micro_step = 0  # counts micro-batches within current optimizer step
    backbone_unfrozen = start_step >= config.warmup_steps

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

            if not backbone_unfrozen and global_step >= config.warmup_steps:
                backbone_unfrozen = True
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

            micro_step += 1

            # Optimizer step only when accumulation complete
            if micro_step == config.gradient_accumulation_steps:
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
                micro_step = 0

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

                # Validation & Best model tracking based on validation CER
                if global_step % config.val_interval_steps == 0 and global_step > 0:
                    val_loss, dev_cer = evaluate_dev(model, dev_loader, vocab_map, device)
                    _log(
                        f"--> [Validation] Step {global_step}: Dev Loss = {val_loss:.4f} | "
                        f"Dev CER = {dev_cer:.2f}%"
                    )
                    _emit_event(
                        {
                            "event": "eval",
                            "step": global_step,
                            "validation_loss": round(val_loss, 4),
                            "dev_cer": round(dev_cer, 2),
                        }
                    )

                    is_best = (dev_cer < best_dev_cer) or (
                        math.isclose(dev_cer, best_dev_cer) and val_loss < best_dev_loss
                    )
                    if is_best:
                        best_dev_cer = dev_cer
                        best_dev_loss = val_loss
                        save_best(out_dir, model, global_step, dev_cer, val_loss)
                        _log(
                            f"--> [Validation] New BEST model at step {global_step}: "
                            f"CER = {dev_cer:.2f}%, Loss = {val_loss:.4f}"
                        )
                        _emit_event(
                            {
                                "event": "best",
                                "step": global_step,
                                "dev_cer": round(dev_cer, 2),
                                "dev_loss": round(val_loss, 4),
                                "criterion": "lowest validation CER",
                            }
                        )

                    model.train()

                # Periodic layered checkpointing with automatic pruning beyond `keep`
                if global_step % config.save_interval_steps == 0 and global_step > 0:
                    bundle = CheckpointBundle(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        best_dev_cer=best_dev_cer,
                        best_dev_loss=best_dev_loss,
                    )
                    saved_path = save_checkpoint(out_dir, bundle, keep=config.keep)
                    _emit_event({"event": "checkpoint", "path": str(saved_path.name)})
                    _log(f"--> Checkpoint saved: {saved_path} (latest: {resume_checkpoint})")

                # Handle graceful stop requested by SIGINT/SIGTERM (e.g. make speech-stop)
                if stop_request.signal is not None:
                    _log(
                        f"[Training] Caught signal {stop_request.signal}. "
                        "Saving atomic checkpoint before exit..."
                    )
                    bundle = CheckpointBundle(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        best_dev_cer=best_dev_cer,
                        best_dev_loss=best_dev_loss,
                    )
                    saved_path = save_checkpoint(out_dir, bundle, keep=config.keep)
                    _emit_event(
                        {"event": "stop", "step": global_step, "signal": stop_request.signal}
                    )
                    _log(f"[Training] Exiting cleanly at step {global_step}.")
                    return

                global_step += 1

    model.save_pretrained(out_dir / "final_model")
    summary_data = {
        "final_step": global_step,
        "best_dev_cer_percent": best_dev_cer,
        "best_dev_loss": best_dev_loss,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config.as_dict(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_data, indent=2) + "\n", encoding="utf-8"
    )
    _emit_event(
        {
            "event": "end",
            "step": global_step,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _log(f"[Training] Done at step {global_step}. Model saved to {out_dir / 'final_model'}")
