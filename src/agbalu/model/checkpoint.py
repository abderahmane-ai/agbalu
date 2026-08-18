"""Atomic, checksummed, resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.optim import Optimizer

from agbalu.model.config import ModelError

log: Final = logging.getLogger("agbalu.model")

LATEST: Final = "latest.pt"
BEST: Final = "best.pt"
MANIFEST: Final = "checkpoints.jsonl"

ResumeFrom = Literal["latest", "best"]
CHECKPOINTS: Final[dict[ResumeFrom, str]] = {"latest": LATEST, "best": BEST}
"""The operator-facing names, mapped to the files they select."""

DISK_HEADROOM: Final = 3.0
"""Free space required as a multiple of the payload. Two checkpoints plus slack, because
the atomic write holds the old and new file at once."""

_READ_BLOCK: Final = 1 << 20


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0
    tokens_seen: int = 0
    masked_seen: int = 0
    """Positions actually scored. `tokens_seen` is every token the model read, which is
    what throughput means; the two differ by the mask rate, so reporting one as the other
    understated tokens/s by ~4x."""
    best_validation_loss: float = float("inf")
    consecutive_non_finite: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingState:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def _numpy_keys() -> NDArray[np.uint32]:
    """The MT19937 key array. `get_state` is a union type; only the legacy tuple
    form carries the keys, and that is the form NumPy still returns by default."""
    state = np.random.get_state(legacy=True)
    if not isinstance(state, tuple):  # pragma: no cover - NumPy would have to change
        msg = "numpy RNG state is not the legacy tuple form"
        raise ModelError(msg)
    return state[1]


def _rng_state() -> dict[str, Any]:
    return {
        "python": list(random.getstate()[1]),
        "numpy": _numpy_keys().tolist(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate((3, tuple(int(v) for v in payload["python"]), None))
    numpy_state = np.asarray(payload["numpy"], dtype=np.uint32)
    np.random.set_state(("MT19937", numpy_state, 624, 0, 0.0))
    torch.set_rng_state(torch.as_tensor(payload["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and payload["cuda"]:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(s, dtype=torch.uint8) for s in payload["cuda"]]
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _free_bytes(directory: Path) -> int:
    return shutil.disk_usage(directory).free


def save(
    directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    state: TrainingState,
    *,
    name: str = LATEST,
) -> Path:
    """Atomically write a fully resumable checkpoint, or raise before touching the old one."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    staging = target.with_suffix(".partial")

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.to_dict(),
        "rng": _rng_state(),
    }

    previous = target.stat().st_size if target.is_file() else 0
    if previous and _free_bytes(directory) < previous * DISK_HEADROOM:
        msg = (
            f"refusing to checkpoint: {_free_bytes(directory) / 1e9:.2f} GB free, "
            f"need {previous * DISK_HEADROOM / 1e9:.2f} GB"
        )
        raise ModelError(msg)

    with staging.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())

    checksum = sha256(staging)
    staging.replace(target)
    (directory / f"{name}.sha256").write_text(checksum + "\n", encoding="utf-8")

    with (directory / MANIFEST).open("a", encoding="utf-8") as manifest:
        manifest.write(
            json.dumps(
                {
                    "name": name,
                    "step": state.step,
                    "epoch": state.epoch,
                    "tokens_seen": state.tokens_seen,
                    "best_validation_loss": state.best_validation_loss,
                    "sha256": checksum,
                    "bytes": target.stat().st_size,
                }
            )
            + "\n"
        )
    log.debug("checkpoint %s step=%d sha256=%s", name, state.step, checksum[:12])
    return target


def load(
    directory: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    name: str = LATEST,
    restore_rng: bool = True,
) -> TrainingState:
    """Verify, then restore. A checkpoint that fails its checksum is never loaded."""
    target = directory / name
    if not target.is_file():
        msg = f"no checkpoint at {target}"
        raise ModelError(msg)

    expected_path = directory / f"{name}.sha256"
    if expected_path.is_file():
        expected = expected_path.read_text(encoding="utf-8").strip()
        actual = sha256(target)
        if actual != expected:
            msg = f"{target} is corrupt: sha256 {actual[:12]} != recorded {expected[:12]}"
            raise ModelError(msg)

    payload = torch.load(target, map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "state", "rng"):
        if key not in payload:
            msg = f"{target} is missing `{key}`; it is not a training checkpoint"
            raise ModelError(msg)

    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        _restore_rng(payload["rng"])
    state = TrainingState.from_dict(payload["state"])
    log.info("resumed %s at step=%d epoch=%d", name, state.step, state.epoch)
    return state


def resume_point(directory: Path, *, prefer: ResumeFrom = "latest") -> Path | None:
    """The checkpoint to resume from, verified, or None to start fresh.

    Falls back to the other one when the preferred is missing or fails its checksum: a
    corrupt rolling checkpoint costs progress, not the whole job. Callers must load the
    name this returns — loading a fixed name instead defeats the fallback.
    """
    other: ResumeFrom = "best" if prefer == "latest" else "latest"
    for name in (CHECKPOINTS[prefer], CHECKPOINTS[other]):
        target = directory / name
        expected_path = directory / f"{name}.sha256"
        if not target.is_file():
            continue
        if expected_path.is_file() and sha256(target) != expected_path.read_text().strip():
            log.warning("%s failed its checksum; falling back", target)
            continue
        if name != CHECKPOINTS[prefer]:
            log.warning(
                "%s was asked for but is unusable; resuming from %s", CHECKPOINTS[prefer], name
            )
        return target
    return None
