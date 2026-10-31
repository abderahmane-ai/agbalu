"""Commit immutable, checksummed training snapshots and recover their state."""

from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from agbalu.encoder.configuration import Recipe
from agbalu.encoder.corpus import write_json
from agbalu.encoder.model import Encoder
from agbalu.llm.provenance import code_stamp, config_hash, file_digest


class State(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = 0
    tokens: int = 0
    text_tokens: int = 0
    targets: int = 0
    cursor: int = 0
    context_tokens: int = 512
    next_fast_tokens: int = 0
    next_task_tokens: int = 0
    best_ce: float | None = None
    best_score: float | None = None
    best_corrupted: float | None = None
    reference_clean: float | None = None
    reference_corrupted: float | None = None
    stale_evaluations: int = 0
    fast_improved: bool = False
    budget_tail_tokens: int = 0


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"{directory}: a live process already owns this run"
            raise RuntimeError(message) from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "started_ns": time.time_ns()}))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_snapshot(path: Path) -> None:
    manifest = json.loads((path / "snapshot.stats.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = file_digest(path / name)
        if actual != expected:
            message = f"{path / name}: SHA256 {actual} != committed {expected}"
            raise ValueError(message)


def latest(directory: Path) -> Path | None:
    pointer = directory / "latest.json"
    if not pointer.is_file():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    names = [payload["snapshot"], *payload.get("previous", [])]
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            message = f"{pointer}: invalid snapshot name {name!r}"
            raise ValueError(message)
        path: Path = directory / name
        try:
            verify_snapshot(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        return path
    message = f"{directory}: all committed recovery snapshots failed integrity verification"
    raise ValueError(message)


def save(
    directory: Path, model: Encoder, optimizer: torch.optim.Optimizer, state: State, identity: str
) -> Path:
    required = sum(p.numel() * p.element_size() for p in model.parameters()) * 4
    if shutil.disk_usage(directory).free < required:
        message = f"{directory}: insufficient disk for {required} bytes of checkpoint headroom"
        raise OSError(message)
    name = f"step-{state.step:07d}-{time.time_ns()}"
    partial = directory / f".{name}.partial"
    final = directory / name
    partial.mkdir()
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        message = "NumPy did not return its legacy RNG tuple"
        raise TypeError(message)
    payload = {
        "weights": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.model_dump(),
        "identity": identity,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }
    with (partial / "training.pt").open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    write_json(partial / "state.json", state.model_dump())
    write_json(partial / "geometry.json", model.config.model_dump())
    files = {file.name: file_digest(file) for file in partial.iterdir() if file.is_file()}
    write_json(
        partial / "snapshot.stats.json",
        {
            "schema_version": 1,
            "code": code_stamp(),
            "config_hash": identity,
            "step": state.step,
            "tokens": state.tokens,
            "files": files,
        },
    )
    verify_snapshot(partial)
    sync_directory(partial)
    partial.rename(final)
    pointer = directory / "latest.json"
    old = json.loads(pointer.read_text()) if pointer.is_file() else {}
    previous = [old["snapshot"]] if old.get("snapshot") else []
    write_json(pointer, {"snapshot": name, "previous": previous, "step": state.step})
    sync_directory(directory)
    return final


def promote(directory: Path, snapshot: Path, role: str, metrics: dict[str, object]) -> None:
    if role not in {"best_mlm", "best_deployment", "transition"}:
        message = f"unknown checkpoint role {role!r}"
        raise ValueError(message)
    verify_snapshot(snapshot)
    write_json(directory / f"{role}.json", {"snapshot": snapshot.name, "metrics": metrics})
    sync_directory(directory)


def prune(directory: Path) -> None:
    protected: set[str] = set()
    for name in ("latest", "best_mlm", "best_deployment", "transition"):
        path = directory / f"{name}.json"
        if path.is_file():
            data = json.loads(path.read_text())
            protected.add(data["snapshot"])
            protected.update(data.get("previous", []))
    for path in directory.glob("step-*"):
        if path.is_dir() and path.name not in protected:
            shutil.rmtree(path)


def restore(
    snapshot: Path, model: Encoder, optimizer: torch.optim.Optimizer, identity: str
) -> State:
    verify_snapshot(snapshot)
    payload = torch.load(snapshot / "training.pt", map_location="cpu", weights_only=True)
    if payload["identity"] != identity:
        message = f"{snapshot}: configuration/data identity changed"
        raise ValueError(message)
    model.load_state_dict(payload["weights"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])
    numpy_rng = payload["numpy_rng"]
    np.random.set_state(
        (
            numpy_rng[0],
            np.asarray(numpy_rng[1], dtype=np.uint32),
            numpy_rng[2],
            numpy_rng[3],
            numpy_rng[4],
        )
    )
    return State.model_validate(payload["state"])


def run_identity(recipe: Recipe, data: dict[str, object]) -> str:
    return config_hash({"recipe": recipe.model_dump(), "data": data})
