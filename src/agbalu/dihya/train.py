"""High-performance GPU training engine for Dihya-22M.

Executes native mixed-precision (BF16) training with deferred LDAM margins,
Group-DRO online cell reweighting, and ConLID contrastive learning on HPC.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from agbalu.dihya import checkpoint
from agbalu.dihya.artifacts import run_lock, write_json
from agbalu.dihya.config import ModelConfig, TrainConfig
from agbalu.dihya.dataset import (
    Batch,
    CellSampler,
    TextDataset,
    collate_examples,
)
from agbalu.dihya.losses import (
    GroupWeights,
    LossOutput,
    SupervisedContrastiveLoss,
    ldam_margins,
)
from agbalu.dihya.model import Classifier
from agbalu.llm.provenance import config_hash


def emit_event(event_type: str, **kwargs: object) -> None:
    """Print structured JSON event line for tools.run_log parser."""
    payload = {"family": "dihya", "event": event_type, **kwargs}
    print(json.dumps(payload), flush=True)


def train_step(
    model: Classifier,
    batch: Batch,
    group_weights: GroupWeights,
    contrastive_loss_fn: SupervisedContrastiveLoss,
    config: TrainConfig,
    margins_active: bool,
    device: torch.device,
) -> LossOutput:
    """Execute single forward pass and compute multi-objective loss."""
    input_ids = batch.input_ids.to(device)
    pad_mask = batch.pad_mask.to(device)
    labels = batch.labels.to(device)
    script_labels = batch.script_labels.to(device)
    filter_labels = batch.filter_labels.to(device)
    domains = batch.domains.to(device)
    groups = batch.groups.to(device)

    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(
            input_ids,
            pad_mask=pad_mask,
            labels=labels if margins_active else None,
        )

        # 1. Primary classification loss with Group-DRO reweighting
        ce_per_sample = functional.cross_entropy(
            output.logits, labels, reduction="none", label_smoothing=config.label_smoothing
        )
        cls_loss = group_weights(ce_per_sample, groups) if margins_active else ce_per_sample.mean()

        # 2. ConLID supervised contrastive loss over projections
        con_loss = contrastive_loss_fn(output.projections, labels, domains)

        # 3. Auxiliary multi-task losses
        script_loss = functional.cross_entropy(
            output.script_logits, script_labels, label_smoothing=config.label_smoothing
        )
        filter_loss = functional.binary_cross_entropy_with_logits(
            output.filter_logits, filter_labels
        )

        total_loss = (
            cls_loss
            + config.contrastive_weight * con_loss
            + config.script_weight * script_loss
            + config.filter_weight * filter_loss
        )

    return LossOutput(
        total_loss=total_loss,
        classification_loss=cls_loss.detach(),
        contrastive_loss=con_loss.detach(),
        script_loss=script_loss.detach(),
        filter_loss=filter_loss.detach(),
    )


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_dataset: TextDataset,
    dev_dataset: TextDataset,
    run_dir: Path,
    device: torch.device,
    stop_after_steps: int | None = None,
) -> None:
    """Run full training schedule on HPC with validation and checkpoints."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with run_lock(run_dir):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            _train(
                model_config,
                train_config,
                train_dataset,
                dev_dataset,
                run_dir,
                device,
                stop_after_steps,
            )
        finally:
            torch.use_deterministic_algorithms(previous)


def _train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_dataset: TextDataset,
    dev_dataset: TextDataset,
    run_dir: Path,
    device: torch.device,
    stop_after_steps: int | None,
) -> None:
    """Execute the training schedule under an exclusive run lock."""
    if not train_dataset.examples or not dev_dataset.examples:
        msg = "Training and development datasets must both be nonempty"
        raise ValueError(msg)
    if stop_after_steps is not None and stop_after_steps < 1:
        msg = f"stop_after_steps must be positive, got {stop_after_steps}"
        raise ValueError(msg)
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = Classifier(model_config).to(device)

    # Build group mappings and compute initial class counts for LDAM
    class_counts = [0] * model_config.num_classes
    domain_map: dict[str, int] = {}
    group_map: dict[str, int] = {}

    for ex in train_dataset.examples:
        cid = model_config.classes.index(ex.label)
        class_counts[cid] += 1
        if ex.domain not in domain_map:
            domain_map[ex.domain] = len(domain_map)
        if ex.group_key not in group_map:
            group_map[ex.group_key] = len(group_map)

    sampler = CellSampler(
        train_dataset,
        groups_per_batch=train_config.groups_per_batch,
        samples_per_group=train_config.samples_per_group,
        seed=train_config.seed,
        min_group_support=train_config.min_group_support,
    )
    sampled_groups = {cell: index for index, cell in enumerate(sampler.cells)}
    group_map = {cell: sampled_groups[target] for cell, target in sampler.cell_aliases.items()}

    missing = [
        label for label, count in zip(model_config.classes, class_counts, strict=True) if not count
    ]
    if missing:
        msg = f"Training classes have zero support: {missing!r}"
        raise ValueError(msg)
    train_texts = {ex.text.casefold() for ex in train_dataset.examples}
    overlap = [ex.text for ex in dev_dataset.examples if ex.text.casefold() in train_texts]
    if overlap:
        msg = f"Train/dev overlap: {overlap[0]!r}"
        raise ValueError(msg)

    # Compute LDAM margins
    margins = ldam_margins(class_counts, model_config.margin_scale).to(device)
    model.head.set_margins(margins)

    group_weights = GroupWeights(
        num_groups=len(sampler.cells),
        step=train_config.group_dro_step,
    ).to(device)
    contrastive_fn = SupervisedContrastiveLoss(
        dim=model_config.projection_dim,
        capacity=train_config.memory_bank_size,
        temperature=train_config.contrastive_temperature,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    total_steps = len(sampler) * train_config.epochs
    warmup_steps = max(1, int(total_steps * train_config.warmup_share))
    floor = train_config.min_learning_rate / train_config.learning_rate

    def lr_factor(update: int) -> float:
        if update < warmup_steps:
            return (update + 1) / warmup_steps
        progress = min(1.0, (update - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_factor)

    deferred_cutoff_step = int(total_steps * train_config.deferred_share)

    emit_event(
        "start",
        train_rows=len(train_dataset),
        dev_rows=len(dev_dataset),
        steps=total_steps,
        classes=model_config.num_classes,
        groups=len(sampler.cells),
        params=model.trainable_parameters(),
        class_counts=dict(zip(model_config.classes, class_counts, strict=True)),
        group_support=dict(Counter(ex.group_key for ex in train_dataset.examples)),
        sampling_cell_aliases={
            cell: target for cell, target in sampler.cell_aliases.items() if cell != target
        },
        seed=train_config.seed,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )

    step = 0
    total_bytes = 0
    best_score: float | None = None
    start_time = time.time()
    identity = config_hash(
        {
            "model": asdict(model_config),
            "train": asdict(train_config),
            "groups": group_map,
            "train_data": [(ex.text, ex.label, ex.domain) for ex in train_dataset.examples],
            "dev_data": [(ex.text, ex.label, ex.domain) for ex in dev_dataset.examples],
        }
    )
    saved = checkpoint.load(run_dir, identity)
    if saved is not None:
        checkpoint.restore_module(model, saved, "model")
        checkpoint.restore_module(group_weights, saved, "group_weights")
        checkpoint.restore_module(contrastive_fn, saved, "contrastive")
        optimizer_state, scheduler_state = saved["optimizer"], saved["scheduler"]
        if not isinstance(optimizer_state, dict) or not isinstance(scheduler_state, dict):
            msg = "Invalid optimizer or scheduler state"
            raise ValueError(msg)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        saved_step, saved_best = saved["step"], saved["best_score"]
        if not isinstance(saved_step, int) or (
            saved_best is not None and not isinstance(saved_best, float)
        ):
            msg = "Invalid checkpoint step or best score"
            raise ValueError(msg)
        step, best_score = saved_step, saved_best
        torch_rng, python_rng, cuda_rng = saved["torch_rng"], saved["python_rng"], saved["cuda_rng"]
        if (
            not isinstance(torch_rng, torch.Tensor)
            or not isinstance(python_rng, tuple)
            or not isinstance(cuda_rng, list)
        ):
            msg = "Invalid checkpoint random state"
            raise ValueError(msg)
        torch.set_rng_state(torch_rng)
        random.setstate(python_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        emit_event("resume", step=step)
    if step >= total_steps or (stop_after_steps is not None and step >= stop_after_steps):
        emit_event("ended", ended="already_complete", step=step, steps_this_run=0)
        return
    initial_step = step
    eval_interval = max(1, len(sampler) // train_config.evaluations_per_epoch)
    requested_signal: list[int] = []

    def request_stop(signum: int, _frame: object) -> None:
        requested_signal.append(signum)

    old_signals = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    model.train()
    try:
        for epoch in range(step // len(sampler), train_config.epochs):
            sampler.set_epoch(epoch, train_config.seed)
            skip_batches = step % len(sampler) if epoch == step // len(sampler) else 0
            for batch_index, batch_indices in enumerate(sampler):
                if batch_index < skip_batches:
                    continue
                margins_active = step >= deferred_cutoff_step

                batch_examples = [train_dataset[idx] for idx in batch_indices]
                batch = collate_examples(batch_examples, model_config, domain_map, group_map)
                total_bytes += int(batch.pad_mask.sum())

                optimizer.zero_grad(set_to_none=True)
                losses = train_step(
                    model,
                    batch,
                    group_weights,
                    contrastive_fn,
                    train_config,
                    margins_active,
                    device,
                )

                torch.autograd.backward(losses.total_loss)
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                step += 1

                if step % 20 == 0 or step == total_steps:
                    elapsed = max(0.001, time.time() - start_time)
                    seq_per_sec = ((step - initial_step) * sampler.batch_size) / elapsed
                    emit_event(
                        "step",
                        step=step,
                        loss=round(float(losses.total_loss.item()), 4),
                        cls_loss=round(float(losses.classification_loss.item()), 4),
                        con_loss=round(float(losses.contrastive_loss.item()), 4),
                        speed=round(seq_per_sec, 1),
                        lr=scheduler.get_last_lr()[0],
                        tokens_this_run=total_bytes,
                        steps_this_run=step - initial_step,
                    )

                stopping = bool(requested_signal) or (
                    stop_after_steps is not None and step >= stop_after_steps
                )
                if step == 1 or step % eval_interval == 0 or step == total_steps or stopping:
                    from agbalu.dihya.evaluate import evaluate_dataset

                    checkpoint.save(
                        run_dir,
                        step,
                        best_score,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        group_weights=group_weights,
                        contrastive=contrastive_fn,
                        identity=identity,
                    )
                    report = evaluate_dataset(
                        model,
                        dev_dataset,
                        model_config,
                        device,
                        train_config.eval_batch_size,
                        min_group_support=train_config.min_group_support,
                    )
                    score = report["selection_score"]
                    worst_grp = str(report.get("worst_group", "unknown"))
                    worst_cls = str(report.get("worst_class", "unknown"))
                    worst_cls_rec = float(report.get("worst_class_recall", 0.0))
                    emit_event(
                        "eval",
                        step=step,
                        accuracy=report["accuracy"],
                        worst_group_recall=report["worst_group_recall"],
                        macro_f1=report["macro_f1"],
                        selection_score=score,
                        worst_group=worst_grp,
                        worst_class_recall=worst_cls_rec,
                        worst_class=worst_cls,
                    )
                    write_json(run_dir / f"eval-{step:07d}.stats.json", report)
                    if best_score is None or score > best_score:
                        best_score = score
                        run_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint.save_model(
                            run_dir / "best.pt",
                            model,
                            model_config,
                            train_config,
                            step,
                            report=report,
                        )
                        emit_event(
                            "best",
                            step=step,
                            score=best_score,
                            worst_group=worst_grp,
                            worst_class=worst_cls,
                        )
                    model.train()
                    checkpoint.save(
                        run_dir,
                        step,
                        best_score,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        group_weights=group_weights,
                        contrastive=contrastive_fn,
                        identity=identity,
                    )

                if stopping:
                    checkpoint.save_model(
                        run_dir / "final.pt", model, model_config, train_config, step, report=None
                    )
                    emit_event(
                        "ended",
                        ended="signal" if requested_signal else "stop_after_steps",
                        step=step,
                        steps_this_run=step - initial_step,
                        best_score=best_score,
                    )
                    if requested_signal:
                        raise SystemExit(128 + requested_signal[0])
                    return
    finally:
        for signum, previous in old_signals.items():
            signal.signal(signum, previous)
    checkpoint.save_model(
        run_dir / "final.pt", model, model_config, train_config, step, report=None
    )
    emit_event("ended", ended="completed", step=step, best_score=best_score)
