"""Commit verified restoration snapshots with exact optimizer and sampling recovery."""

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from agbalu.encoder.checkpoint import run_lock, sync_directory, verify_snapshot
from agbalu.encoder.corpus import read_json, write_json
from agbalu.llm.provenance import code_stamp, file_digest
from agbalu.restoration.model import Restorer

__all__ = ["State", "latest", "load", "promote", "prune", "run_lock", "save"]


class State(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = 0
    epoch: int = 0
    cursor: int = 0
    tokens: int = 0
    # Retain the serialized field name used by the first component-target pilots.
    words: int = 0
    best_score: float | None = None
    stale_evaluations: int = 0
    next_evaluation: int = 1
    ended: str | None = None


def latest(directory: Path) -> Path | None:
    pointer = directory / "latest.json"
    if not pointer.exists():
        return None
    payload = read_json(pointer)
    previous = payload.get("previous", [])
    if not isinstance(previous, list):
        msg = f"{pointer}: invalid previous snapshots"
        raise TypeError(msg)
    for name in (payload["snapshot"], *previous):
        if not isinstance(name, str) or Path(name).name != name:
            msg = f"{pointer}: invalid snapshot name {name!r}"
            raise ValueError(msg)
        path = directory / name
        try:
            verify_snapshot(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        return path
    msg = f"{directory}: no intact recovery snapshot"
    raise ValueError(msg)


def save(
    directory: Path, model: Restorer, optimizer: torch.optim.Optimizer, state: State, identity: str
) -> Path:
    required = sum(value.numel() * value.element_size() for value in model.parameters()) * 4
    if shutil.disk_usage(directory).free < required:
        msg = f"{directory}: checkpoint requires {required} bytes of free headroom"
        raise OSError(msg)
    name = f"step-{state.step:07d}-{time.time_ns()}"
    partial = directory / f".{name}.partial"
    path = directory / name
    partial.mkdir()
    rng = np.random.get_state(legacy=True)
    if not isinstance(rng, tuple):
        msg = "NumPy RNG state is not a legacy tuple"
        raise TypeError(msg)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.model_dump(),
        "identity": identity,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": [rng[0], rng[1].tolist(), *rng[2:]],
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        with (partial / "training.pt").open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        write_json(partial / "state.json", state.model_dump())
        write_json(
            partial / "snapshot.stats.json",
            {
                "code": code_stamp(),
                "config_hash": identity,
                "version_tag": None,
                "files": {
                    name: file_digest(partial / name) for name in ("training.pt", "state.json")
                },
            },
        )
        verify_snapshot(partial)
        sync_directory(partial)
        partial.rename(path)
        old = read_json(directory / "latest.json") if (directory / "latest.json").exists() else {}
        previous = [old["snapshot"]] if "snapshot" in old else []
        write_json(
            directory / "latest.json", {"snapshot": name, "previous": previous, "step": state.step}
        )
        sync_directory(directory)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return path


def load(path: Path, model: Restorer, optimizer: torch.optim.Optimizer, identity: str) -> State:
    verify_snapshot(path)
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    if payload["identity"] != identity:
        msg = f"{path}: configuration/data changed on resume"
        raise ValueError(msg)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    rng = payload["numpy_rng"]
    np.random.set_state((rng[0], np.asarray(rng[1], np.uint32), rng[2], rng[3], rng[4]))
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return State.model_validate(payload["state"])


def promote(directory: Path, snapshot: Path, metrics: dict[str, object]) -> None:
    write_json(directory / "best_deployment.json", {"snapshot": snapshot.name, "metrics": metrics})
    sync_directory(directory)


def prune(directory: Path) -> None:
    kept: set[str] = set()
    for name in ("latest", "best_deployment"):
        path = directory / f"{name}.json"
        if path.exists():
            payload = read_json(path)
            kept.add(str(payload["snapshot"]))
            previous = payload.get("previous", [])
            if isinstance(previous, list):
                kept.update(str(name) for name in previous)
    for path in directory.glob("step-*"):
        if path.is_dir() and path.name not in kept:
            shutil.rmtree(path)
