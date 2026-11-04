"""Commit complete adapter recovery state at optimizer boundaries."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from agbalu.codec_tts.adaptation import adapter_state, restore_adapter
from agbalu.codec_tts.artifacts import (
    atomic_binary,
    digest_file,
    provenance,
    sync_directory,
    write_json,
)
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.configuration import Specification


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    step: int = Field(default=0, ge=0)
    cursor: int = Field(default=0, ge=0)
    targets: int = Field(default=0, ge=0)
    examples: int = Field(default=0, ge=0)
    best_ce: float | None = None
    best_speech: float | None = None
    stale_evaluations: int = Field(default=0, ge=0)


class Pointer(Specification):
    snapshot: str
    previous: tuple[str, ...] = ()
    score: float | None = None
    evidence_sha256: str | None = None


class Snapshot(Specification):
    identity: str
    step: int = Field(ge=0)
    training_sha256: str
    provenance: dict[str, object]


class NumpyState(Specification):
    keys: tuple[int, ...] = Field(min_length=624, max_length=624)
    position: int = Field(ge=0, le=624)
    has_gauss: int = Field(ge=0, le=1)
    gauss: float


@contextmanager
def preserved_rng() -> Iterator[None]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    python = random.getstate()
    numpy = np.random.get_state(legacy=False)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python)
        np.random.set_state(numpy)


def snapshot_path(root: Path, name: str) -> Path:
    if not name.startswith("step-") or not name.removeprefix("step-").isdigit():
        message = f"invalid checkpoint name={name!r} under {root}"
        raise ValueError(message)
    return root / name


def verify_snapshot(path: Path, identity: str) -> Snapshot:
    manifest = Snapshot.model_validate_json((path / "snapshot.stats.json").read_bytes())
    observed = digest_file(path / "training.pt")
    if manifest.identity != identity or observed != manifest.training_sha256:
        message = f"checkpoint identity/checksum mismatch at {path}: observed sha256={observed}"
        raise ValueError(message)
    return manifest


def latest_snapshot(root: Path, identity: str) -> Path | None:
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = Pointer.model_validate_json(pointer_path.read_bytes())
    failures: list[str] = []
    for name in (pointer.snapshot, *pointer.previous):
        path = snapshot_path(root, name)
        try:
            verify_snapshot(path, identity)
        except (OSError, ValueError) as error:
            failures.append(str(error))
        else:
            return path
    message = f"no valid recovery snapshot under {root}: {failures}"
    raise ValueError(message)


def tensor_mapping(value: object) -> dict[str, Tensor]:
    if not isinstance(value, dict):
        message = f"expected tensor mapping, got {type(value).__name__}"
        raise TypeError(message)
    result: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            message = f"invalid tensor entry name={name!r}, type={type(tensor).__name__}"
            raise TypeError(message)
        result[name] = tensor
    return result


def save_snapshot(
    root: Path, model: Generator, optimizer: torch.optim.Optimizer, state: RunState, identity: str
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step-{state.step:08d}"
    if destination.is_dir():
        verify_snapshot(destination, identity)
    else:
        numpy_state = np.random.get_state(legacy=False)
        inner = numpy_state["state"]
        numpy_snapshot = NumpyState(
            keys=tuple(inner["key"].tolist()),
            position=inner["pos"],
            has_gauss=numpy_state["has_gauss"],
            gauss=numpy_state["gauss"],
        )
        payload = {
            "adapter": adapter_state(model),
            "optimizer": optimizer.state_dict(),
            "state": state.model_dump(),
            "identity": identity,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "numpy_rng": numpy_snapshot.model_dump(),
        }
        with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=root) as temporary:
            stage = Path(temporary) / "snapshot"
            with atomic_binary(stage / "training.pt") as stream:
                torch.save(payload, stream)
            manifest = Snapshot(
                identity=identity,
                step=state.step,
                training_sha256=digest_file(stage / "training.pt"),
                provenance=provenance(identity),
            )
            write_json(stage / "snapshot.stats.json", manifest.model_dump())
            stage.rename(destination)
            sync_directory(root)
    latest = root / "latest.json"
    previous: tuple[str, ...] = ()
    if latest.is_file():
        old = Pointer.model_validate_json(latest.read_bytes())
        previous = tuple(
            name for name in (old.snapshot, *old.previous) if name != destination.name
        )[:1]
    write_json(latest, Pointer(snapshot=destination.name, previous=previous).model_dump())
    return destination


def restore_snapshot(
    path: Path, model: Generator, optimizer: torch.optim.Optimizer, identity: str
) -> RunState:
    verify_snapshot(path, identity)
    payload: object = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("identity") != identity:
        message = f"invalid training payload at {path}"
        raise ValueError(message)
    restore_adapter(model, tensor_mapping(payload.get("adapter")))
    optimizer_state, torch_rng = payload.get("optimizer"), payload.get("torch_rng")
    python_rng, numpy_rng, cuda_rng = (
        payload.get("python_rng"),
        payload.get("numpy_rng"),
        payload.get("cuda_rng"),
    )
    if not isinstance(optimizer_state, dict) or not isinstance(torch_rng, Tensor):
        message = f"missing optimizer/torch RNG state at {path}"
        raise TypeError(message)
    if (
        not isinstance(python_rng, tuple)
        or not isinstance(numpy_rng, dict)
        or not isinstance(cuda_rng, list)
    ):
        message = f"missing Python/NumPy/CUDA RNG state at {path}"
        raise TypeError(message)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(torch_rng)
    random.setstate(python_rng)
    numpy_snapshot = NumpyState.model_validate(numpy_rng)
    np.random.set_state(
        {
            "bit_generator": "MT19937",
            "state": {
                "key": np.asarray(numpy_snapshot.keys, dtype=np.uint32),
                "pos": numpy_snapshot.position,
            },
            "has_gauss": numpy_snapshot.has_gauss,
            "gauss": numpy_snapshot.gauss,
        }
    )
    if torch.cuda.is_available():
        if not all(isinstance(value, Tensor) for value in cuda_rng):
            message = f"invalid CUDA RNG tensors at {path}"
            raise ValueError(message)
        torch.cuda.set_rng_state_all(cuda_rng)
    return RunState.model_validate(payload.get("state"))


def prune_snapshots(root: Path) -> None:
    retained: set[str] = set()
    for role in ("latest", "best_ce", "best_speech"):
        path = root / f"{role}.json"
        if path.is_file():
            pointer = Pointer.model_validate_json(path.read_bytes())
            retained.update((pointer.snapshot, *pointer.previous))
    shortlist = root / "shortlist.json"
    if shortlist.is_file():
        payload: object = json.loads(shortlist.read_bytes())
        if not isinstance(payload, dict) or not all(isinstance(item, str) for item in payload):
            message = f"invalid shortlist at {shortlist}"
            raise ValueError(message)
        retained.update(payload)
    for path in root.glob("step-*"):
        snapshot_path(root, path.name)
        if path.name not in retained and (path / "snapshot.stats.json").is_file():
            shutil.rmtree(path)


def retain_candidate(root: Path, snapshot: Path, score: float) -> None:
    path = root / "shortlist.json"
    candidates: dict[str, float] = {}
    if path.is_file():
        payload = json.loads(path.read_bytes())
        for name, value in payload.items():
            snapshot_path(root, name)
            candidates[name] = float(value)
    candidates[snapshot.name] = score
    selected = sorted(candidates, key=lambda name: (candidates[name], name))[:3]
    write_json(path, {name: candidates[name] for name in selected})
