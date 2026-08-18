"""Stage a fine-tuned MT checkpoint as a publishable Hub repository.

A Modal download is not a release. The files arrive nested under `final/`, carry the
trainer's resumption state, and have no card. `hf upload` publishes a directory verbatim,
so the nesting would put the weights at `final/model.safetensors`, where `from_pretrained`
does not look for them.

The vocabulary is trimmed, so `config.json`'s `vocab_size` and the length of `keep.json`
are one number stored twice. They are checked against each other, and every other check
runs, before anything is written: a refused staging must not leave a directory that looks
publishable.

Files are hard-linked where the filesystem allows. The weights are 4.65 GB and a copy
would be a second 4.65 GB.

    python3 -m tools.stage_mt_release --source artifacts/checkpoints/Amrouche-1.3B \
        --out artifacts/release/Amrouche-1.3B --card docs/cards/amrouche-1.3b.md
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Final

from agbalu.mt.vocab import read_keep

TRAINING_STATE: Final = frozenset(
    {
        "training_args.bin",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }
)
"""Resumption state, not release material. `training_args.bin` also records local paths."""

REQUIRED: Final = ("model.safetensors", "config.json", "keep.json")
"""`keep.json` is required because the weights are unusable without it: the tokenizer
speaks NLLB's full vocabulary and the model speaks the trimmed one."""


class StagingError(Exception):
    """The source directory cannot be published as it stands."""


def resolve_source(source: Path) -> Path:
    """The directory holding the weights, following a single `final/` nesting."""
    for candidate in (source, source / "final"):
        if (candidate / "model.safetensors").is_file():
            return candidate
    message = f"no model.safetensors in {source} or {source / 'final'}"
    raise StagingError(message)


def vocabulary_size(config: Path) -> int:
    payload = json.loads(config.read_text(encoding="utf-8"))
    size = payload.get("vocab_size")
    if not isinstance(size, int):
        message = f"{config} has no integer vocab_size"
        raise StagingError(message)
    return size


def check(source: Path, out: Path, card: Path) -> tuple[int, list[Path]]:
    """The kept-token count and the files that will ship, or an exception."""
    if not card.is_file():
        message = f"card not found: {card}"
        raise StagingError(message)
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        message = f"{source} is missing {', '.join(missing)}"
        raise StagingError(message)

    resolved_out, resolved_source = out.resolve(), source.resolve()
    if resolved_out == resolved_source or resolved_source.is_relative_to(resolved_out):
        # `hf upload` would publish both the flat copy and the nested original.
        message = f"out {out} must not contain the source {source}"
        raise StagingError(message)

    kept = read_keep(source / "keep.json")
    declared = vocabulary_size(source / "config.json")
    if declared != len(kept):
        message = f"config.json vocab_size {declared} against {len(kept)} kept ids"
        raise StagingError(message)

    shipped = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.name not in TRAINING_STATE and path.name != "README.md"
    )
    return len(kept), shipped


def place(origin: Path, target: Path) -> None:
    """Hard-link, or copy where the filesystem refuses to link across devices."""
    target.unlink(missing_ok=True)
    try:
        os.link(origin, target)
    except OSError:
        shutil.copyfile(origin, target)


def stage(source: Path, out: Path, card: Path) -> tuple[int, list[Path]]:
    """Write the publishable directory. Returns the kept-token count and the files in it."""
    source = resolve_source(source)
    kept, shipped = check(source, out, card)

    out.mkdir(parents=True, exist_ok=True)
    for path in shipped:
        place(path, out / path.name)
    shutil.copyfile(card, out / "README.md")
    return kept, sorted(out.iterdir())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        kept, staged = stage(args.source, args.out, args.card)
    except StagingError as error:
        raise SystemExit(str(error)) from error

    print(f"{args.out}")
    for path in staged:
        print(f"  {path.name:26} {path.stat().st_size / 1e6:9.2f} MB")
    print(f"  {len(staged)} files, vocabulary {kept:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
