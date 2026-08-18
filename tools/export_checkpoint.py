"""Stage a PyTorch checkpoint as a publishable model repository.

A training checkpoint is not a release. It carries optimizer moments, scheduler state and
whatever else a resume needs, none of which a downloader can use and all of which they
pay to fetch — the encoder's `best.pt` is 398 MB against 124.5 MB published.

What is written is the weights in safetensors, a `config.json` naming the architecture,
whatever extra files the model needs to be usable, and the card. What is *not* written is
anything the source did not actually contain: the export reads the file and reports its
contents, so a card claiming a checkpoint holds a validation curve can be checked against
the export's own manifest.

Every check runs before the target directory is touched. A refused export must not leave a
directory that looks published — which is also why a corrupt source fails here rather than
after 800 MB has been copied.

    python3 -m tools.export_checkpoint --source artifacts/checkpoints/Juba-27M/juba_final.pt \\
        --out artifacts/release/Juba-27M --card docs/cards/juba-27m.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from torch import Tensor

WEIGHT_KEYS: Final = ("model_state_dict", "model", "state_dict")
"""Where a checkpoint keeps its weights, in the order they are looked for. Three spellings
because three trainers wrote these files; a fourth belongs here, not in a caller."""

DROP_PREFIXES: Final = ("optimizer", "scheduler", "rng")
"""Resume state. Present in a checkpoint on purpose and absent from a release on purpose."""


class ExportError(Exception):
    """A checkpoint that cannot be published as described."""


@dataclass(frozen=True, slots=True)
class Staged:
    """One file the export wrote."""

    name: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "bytes": self.bytes, "sha256": self.sha256}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(source: Path) -> dict[str, object]:
    """The checkpoint, with a readable message when the file on disk is not one.

    `torch.load` on a truncated file raises a miniz error naming neither the path nor the
    cause. That is not hypothetical: `artifacts/asr/best.pt` is 833 MB of a partial volume
    download, and it fails exactly this way.
    """
    import torch

    if not source.is_file():
        message = f"no checkpoint at {source}"
        raise ExportError(message)
    try:
        loaded = torch.load(source, map_location="cpu", weights_only=False)
    # Broad on purpose: a truncated file surfaces as a miniz RuntimeError, a mangled one as
    # an UnpicklingError, and a wrong-format one as a zipfile error. All three mean the
    # same thing to a caller, and none of them names the path.
    except Exception as error:
        message = (
            f"{source} ({source.stat().st_size:,} bytes) is not a readable checkpoint: "
            f"{error}. A partial download reads as a corrupt zip; refetch it."
        )
        raise ExportError(message) from error
    if not isinstance(loaded, dict):
        message = f"{source} holds {type(loaded).__name__}, not a checkpoint dict"
        raise ExportError(message)
    return loaded


def weights(checkpoint: dict[str, object]) -> dict[str, Tensor]:
    """The state dict, checked to be one rather than assumed.

    `torch.load` returns `Any`, so without this a checkpoint whose `model` key holds a
    string reaches `save_file` and fails there instead of here.
    """
    import torch

    for key in WEIGHT_KEYS:
        found = checkpoint.get(key)
        if isinstance(found, dict) and found:
            wrong = [name for name, value in found.items() if not isinstance(value, torch.Tensor)]
            if wrong:
                message = f"{key} holds non-tensors under {wrong[:3]}"
                raise ExportError(message)
            return dict(found)
    message = f"no weights under any of {WEIGHT_KEYS}; the file holds {sorted(checkpoint)}"
    raise ExportError(message)


def untie(tensors: dict[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    """The tensors with every aliased duplicate removed, and the names that were removed.

    A tied head and embedding are one tensor under two names. `save_file` refuses shared
    storage rather than writing one copy silently, so the duplicate is dropped here and
    the module re-ties it on load. Detected by storage identity rather than by naming the
    key, because this tool serves more than one architecture and a hard-coded name is a
    silent failure on the next one.
    """
    kept: dict[str, Tensor] = {}
    seen: set[tuple[int, int]] = set()
    dropped: list[str] = []
    for name, tensor in tensors.items():
        identity = (tensor.untyped_storage().data_ptr(), int(tensor.storage_offset()))
        if identity in seen:
            dropped.append(name)
            continue
        seen.add(identity)
        kept[name] = tensor.contiguous()
    return kept, dropped


def drop_derived(
    tensors: dict[str, Tensor], suffixes: Sequence[str]
) -> tuple[dict[str, Tensor], list[str]]:
    """The tensors with every derived lookup table removed, and the names that were removed.

    A table the module rebuilds from its own config is not weights. `position_indices` is
    twelve identical 512x512 int64 buckets — 25.2 MB against a 125.6 MB model — and shipping
    it both inflates the download and makes `from_pretrained` report unexpected keys to every
    downloader. The published modules keep these on plain attributes precisely so they are
    absent here.
    """
    if not suffixes:
        return tensors, []
    kept = {k: v for k, v in tensors.items() if not k.endswith(tuple(suffixes))}
    return kept, sorted(set(tensors) - set(kept))


def config_of(checkpoint: dict[str, object]) -> dict[str, object]:
    found = checkpoint.get("config")
    if not isinstance(found, dict):
        message = "the checkpoint carries no config, so the export cannot say what shape it is"
        raise ExportError(message)
    return found


def contents(checkpoint: dict[str, object]) -> list[str]:
    """Top-level keys, so what the checkpoint *does* hold is recorded rather than assumed."""
    return sorted(checkpoint)


def stage(
    source: Path,
    out: Path,
    card: Path,
    extras: Sequence[Path] = (),
    config_path: Path | None = None,
    derived: Sequence[str] = (),
) -> list[Staged]:
    """Write the release, after every check has passed.

    `config_path` supplies the architecture for a checkpoint that carries none. A resumable
    training checkpoint holds the optimizer, the scheduler and the RNG and never writes its
    config down, so without this the weights publish with nothing that can construct them.
    """
    from safetensors.torch import save_file

    if not card.is_file():
        message = f"card not found at {card}"
        raise ExportError(message)
    missing = [str(extra) for extra in extras if not extra.is_file()]
    if missing:
        message = f"extra files not found: {missing}"
        raise ExportError(message)

    if config_path is not None and not config_path.is_file():
        message = f"no config at {config_path}"
        raise ExportError(message)

    checkpoint = load(source)
    kept, tied = untie(weights(checkpoint))
    kept, derived_dropped = drop_derived(kept, derived)
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path is not None
        else config_of(checkpoint)
    )
    dropped = sorted(k for k in checkpoint if k.startswith(DROP_PREFIXES))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    save_file(kept, out / "model.safetensors")
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(card, out / "README.md")
    for extra in extras:
        shutil.copyfile(extra, out / extra.name)

    staged = [
        Staged(name=path.name, bytes=path.stat().st_size, sha256=digest(path))
        for path in sorted(out.iterdir())
    ]
    (out / "export.stats.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_contents": contents(checkpoint),
                "dropped_state": dropped,
                "dropped_tied": tied,
                "dropped_derived": derived_dropped,
                "tensors": len(kept),
                "parameters": sum(tensor.numel() for tensor in kept.values()),
                "files": [entry.as_dict() for entry in staged],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return staged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument(
        "--extra", type=Path, action="append", default=[], help="repeatable; copied verbatim"
    )
    parser.add_argument(
        "--config", type=Path, help="architecture, for a checkpoint that carries none"
    )
    parser.add_argument(
        "--derived",
        action="append",
        default=[],
        help="repeatable key suffix the module rebuilds on load; dropped from the weights",
    )
    args = parser.parse_args(argv)

    try:
        staged = stage(args.source, args.out, args.card, args.extra, args.config, args.derived)
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for entry in staged:
        print(f"   {entry.name:<24} {entry.bytes:>13,} B  {entry.sha256[:16]}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
