"""Build a publishable model directory from a training checkpoint.

A training checkpoint is not a release. `best.pt` is 398 MB because it carries LAMB's two
moment tensors and the RNG state; the weights alone are 174 MB, and 6.1M of the parameters
in the state dict are the classifier decoder sharing storage with the embedding table, which
safetensors refuses to serialise twice.

So the export drops the optimizer, drops the tied duplicate, writes `model.safetensors` plus
a `config.json` that reconstructs the architecture, and copies the card in as `README.md`.
It verifies the result loads back into the real module before it returns.

    python3 -m tools.export_release --run agbalu-encoder-v1 --out artifacts/release/Masinissa-31M
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import torch
from safetensors.torch import save_file

from agbalu.model.config import PRESETS, ModelConfig, Preset
from agbalu.model.modeling import Encoder

RUNS: Final = Path("artifacts/runs")

TIED: Final = "classifier.decoder.weight"
"""Tied to `embedding.word_embedding.weight` and stored as the same tensor. `save_file`
raises on shared storage rather than silently writing one copy, so it is dropped here and
re-tied by the module on load."""

DERIVED: Final = "position_indices"
"""Twelve identical 512x512 int64 lookup tables, one per attention layer, computed in
`__init__` from `max_position_embeddings` and `position_bucket_size`. Shipping them costs
25.2 MB of a 149.7 MB release — 17% of the download — for something the module rebuilds
byte-identically. Dropped, and asserted equal on reload."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_of(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    """The model tensors, without the tied duplicate or the derived index buffers."""
    model = checkpoint["model"]
    if not isinstance(model, dict):
        message = f"checkpoint 'model' is {type(model).__name__}, not a state dict"
        raise TypeError(message)
    return {
        key: value.contiguous()
        for key, value in model.items()
        if key != TIED and not key.endswith(DERIVED)
    }


def training_summary(checkpoint: dict[str, object]) -> dict[str, object]:
    """What the release should state about how the weights were produced."""
    state = checkpoint.get("state")
    if not isinstance(state, dict):
        return {}
    return {
        "step": state.get("step"),
        "epoch": state.get("epoch"),
        "tokens_seen": state.get("tokens_seen"),
        "masked_tokens_scored": state.get("masked_seen"),
        "best_validation_loss": state.get("best_validation_loss"),
        "validation_history": state.get("history", []),
    }


def _check_fit(
    tensors: dict[str, torch.Tensor],
    checkpoint: dict[str, object],
    config: ModelConfig,
    preset: Preset,
) -> None:
    """That these weights are this architecture's, and that the omissions are safe.

    A shape mismatch reaches `load_state_dict` as a message about tensor sizes, which names
    neither the preset nor the flag that chose it — `small` and `kab` differ only in
    vocabulary size, so the wrong one would otherwise publish a config that lies.
    """
    reference = Encoder(config)
    expected = dict(reference.state_dict())
    wrong = [
        f"{key} is {tuple(value.shape)}, {preset} wants {tuple(expected[key].shape)}"
        for key, value in tensors.items()
        if key in expected and value.shape != expected[key].shape
    ]
    unknown = sorted(set(tensors) - set(expected))
    absent = sorted(
        k for k in expected if k not in tensors and k != TIED and not k.endswith(DERIVED)
    )
    if wrong or unknown or absent:
        message = (
            f"checkpoint is not a {preset!r} model: "
            f"mismatched={wrong} unknown={unknown} absent={absent}"
        )
        raise ValueError(message)

    saved = checkpoint["model"]
    if not isinstance(saved, dict):
        return
    drifted = [
        key
        for key, value in reference.named_buffers()
        if key.endswith(DERIVED) and key in saved and not torch.equal(value, saved[key])
    ]
    if drifted:
        message = f"{DERIVED} is not reconstructible for {preset}, so it must ship: {drifted}"
        raise ValueError(message)


def export(
    checkpoint_path: Path,
    out: Path,
    preset: Preset,
    card: Path | None = None,
    tokenizer: Path | None = None,
) -> dict[str, object]:
    """Write a publishable directory, having first proved the weights belong to `preset`.

    The checks run before anything is written. A failed export that has already produced a
    `model.safetensors` leaves a directory that looks publishable and is not.
    """
    config: ModelConfig = PRESETS[preset]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensors = weights_of(checkpoint)
    _check_fit(tensors, checkpoint, config, preset)

    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "model.safetensors"
    save_file(tensors, weights_path, metadata={"format": "pt"})
    weights_path.chmod(0o644)

    manifest: dict[str, object] = {
        "architecture": "Encoder",
        "preset": preset,
        "parameters": config.parameters,
        **asdict(config),
        "training": training_summary(checkpoint),
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
        },
        "weights_sha256": sha256(weights_path),
    }
    if tokenizer is not None:
        # The model is unusable without the exact vocabulary it was trained on, so it
        # travels in the repo rather than as a cross-reference someone has to resolve.
        shutil.copyfile(tokenizer, out / tokenizer.name)
        manifest["tokenizer"] = {"file": tokenizer.name, "sha256": sha256(tokenizer)}
    (out / "config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if card is not None:
        shutil.copyfile(card, out / "README.md")
    return manifest


def config_of(manifest: dict[str, object], key: str) -> int:
    value = manifest[key]
    if not isinstance(value, int):
        message = f"manifest {key!r} is {type(value).__name__}, not an int"
        raise TypeError(message)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", default="agbalu-encoder-v1", help="run directory under artifacts/runs"
    )
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--preset", default="kab", choices=sorted(PRESETS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=Path, help="model card to copy in as README.md")
    parser.add_argument("--tokenizer", type=Path, help="the vocabulary the run was trained on")
    args = parser.parse_args(argv)

    checkpoint_path = RUNS / args.run / args.checkpoint
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("card", args.card),
        ("tokenizer", args.tokenizer),
    ):
        if path is not None and not path.is_file():
            message = f"{label} not found: {path}"
            raise SystemExit(message)

    manifest = export(checkpoint_path, args.out, args.preset, args.card, args.tokenizer)
    size = (args.out / "model.safetensors").stat().st_size
    digest = str(manifest["weights_sha256"])
    print(f"{args.out}")
    print(f"  model.safetensors  {size / 1e6:8.1f} MB  {digest[:16]}…")
    print(f"  parameters         {config_of(manifest, 'parameters'):,}")
    training = manifest["training"]
    if isinstance(training, dict) and training.get("step") is not None:
        print(f"  step               {int(training['step']):,}")
        print(f"  validation loss    {float(training['best_validation_loss']):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
