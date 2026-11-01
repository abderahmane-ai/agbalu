"""Commit checksummed training state and recover the latest complete snapshot."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from agbalu.dihya.artifacts import atomic_file, write_json
from agbalu.dihya.config import ModelConfig, TrainConfig
from agbalu.llm.provenance import code_stamp, file_digest


def save(
    directory: Path,
    step: int,
    best_score: float | None,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    group_weights: nn.Module,
    contrastive: nn.Module,
    identity: str,
) -> None:
    """Persist every mutable training component before updating the recovery pointer."""
    path = directory / "checkpoints" / f"step-{step:07d}.pt"
    payload = {
        "step": step,
        "best_score": best_score,
        "identity": identity,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "group_weights": group_weights.state_dict(),
        "contrastive": contrastive.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    with atomic_file(path) as stream:
        torch.save(payload, stream)
    write_json(
        path.with_suffix(".stats.json"),
        {
            "step": step,
            "sha256": file_digest(path),
            "code": code_stamp(),
            "config_hash": identity,
            "version_tag": None,
        },
    )
    write_json(directory / "latest.json", {"snapshot": str(path.relative_to(directory))})


def load(directory: Path, identity: str) -> dict[str, object] | None:
    """Reject incompatible or damaged snapshots before returning their state."""
    pointer = directory / "latest.json"
    if not pointer.is_file():
        return None
    name = json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        msg = f"{pointer}: invalid snapshot {name!r}"
        raise ValueError(msg)
    path = directory / name
    metadata = json.loads(path.with_suffix(".stats.json").read_text(encoding="utf-8"))
    if file_digest(path) != metadata["sha256"]:
        msg = f"{path}: checkpoint SHA256 mismatch"
        raise ValueError(msg)
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("identity") != identity:
        msg = f"{path}: incompatible training identity {identity}"
        raise ValueError(msg)
    return {str(key): value for key, value in payload.items()}


def restore_module(module: nn.Module, payload: Mapping[str, object], key: str) -> None:
    """Narrow tensor state at the serialization boundary."""
    state = payload[key]
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        msg = f"Checkpoint {key!r} is not a tensor state dictionary"
        raise ValueError(msg)
    module.load_state_dict(state, strict=True)


def save_model(
    path: Path,
    model: nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    *,
    report: object,
) -> None:
    """Keep inference weights separate from resumable optimizer state."""
    with atomic_file(path) as stream:
        torch.save(model.state_dict(), stream)
    write_json(
        path.with_suffix(".stats.json"),
        {
            "sha256": file_digest(path),
            "step": step,
            "report": report,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "code": code_stamp(),
            "version_tag": None,
        },
    )
