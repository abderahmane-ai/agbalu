"""The full model rebuilt from a trained text tower and the checkpoint it was trained from.

SFT trains and saves the text tower alone (`Qwen3_5ForCausalLM`); vision, MTP, the tokenizer
and the chat template stay in the full checkpoint it started from (the merge). This writes
that full checkpoint again with every text-tower tensor taken from the trained tower, cast to
the template's dtype, and every other tensor and file copied byte for byte. Both key sets
must be equal, so a tensor is never dropped or left untrained in silence.
"""

from __future__ import annotations

import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Final

from agbalu.llm.merge import (
    COPIED_FILES,
    INDEX_FILE,
    SHOWN,
    TEXT_PREFIX,
    Checkpoint,
    MergeError,
    digests,
    listed,
)
from agbalu.llm.progress import Tracker
from agbalu.llm.provenance import code_stamp, file_digest

if TYPE_CHECKING:
    import torch
    from safetensors import safe_open

REPORT_FILE: Final = "assembly.json"


def check_tower(template: Checkpoint, tower: Checkpoint) -> set[str]:
    """The text-tower names, after proving the tower holds exactly those tensors."""
    expected = template.names(TEXT_PREFIX) | ({"lm_head.weight"} & set(template.shards))
    held = tower.names()
    if held != expected:
        message = (
            f"{tower.root} does not match {template.root}'s text tower: missing "
            f"[{listed(expected - held)}], extra [{listed(held - expected)}]"
        )
        raise MergeError(message)
    return expected


def replaced(old: torch.Tensor, new: torch.Tensor, name: str) -> torch.Tensor:
    """The trained value in the template's dtype; a shape change or non-finite value refuses."""
    import torch

    if new.shape != old.shape:
        message = f"{name}: trained shape {tuple(new.shape)} against {tuple(old.shape)}"
        raise MergeError(message)
    value = new.to(old.dtype)
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        message = f"{name}: non-finite values in the trained tower"
        raise MergeError(message)
    return value


def assemble(template_root: Path, tower_root: Path, out: Path) -> dict[str, object]:
    """Write the full model to `out` through a hidden partial directory; return the report."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if out.exists():
        message = f"{out} exists; an assembly never overwrites"
        raise MergeError(message)
    template, tower = Checkpoint.read(template_root), Checkpoint.read(tower_root)
    names = check_tower(template, tower)

    partial = out.with_name(f".{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    taken = copied = 0
    recast: list[str] = []
    tracker = Tracker("assemble tensors", len(template.shards))
    with ExitStack() as stack:
        handles: dict[tuple[str, str], safe_open] = {}

        def opened(checkpoint: Checkpoint, shard: str) -> safe_open:
            key = (str(checkpoint.root), shard)
            if key not in handles:
                handles[key] = stack.enter_context(
                    safe_open(checkpoint.root / shard, framework="pt")
                )
            return handles[key]

        for shard in template.files():
            tensors: dict[str, torch.Tensor] = {}
            for name in sorted(n for n, file in template.shards.items() if file == shard):
                value = opened(template, shard).get_tensor(name)
                if name in names:
                    new = opened(tower, tower.shards[name]).get_tensor(name)
                    if new.dtype != value.dtype:
                        recast.append(name)
                    value = replaced(value, new, name)
                    taken += 1
                else:
                    copied += 1
                tensors[name] = value
                tracker.advance()
            save_file(tensors, partial / shard, metadata={"format": "pt"})

    if taken != len(names) or taken + copied != len(template.shards):
        message = (
            f"took {taken} of {len(names)} tower tensors and copied {copied}; "
            f"the template holds {len(template.shards)}"
        )
        raise MergeError(message)
    if (template.root / INDEX_FILE).is_file():
        shutil.copyfile(template.root / INDEX_FILE, partial / INDEX_FILE)
    present = [name for name in COPIED_FILES if (template.root / name).is_file()]
    for name in present:
        shutil.copyfile(template.root / name, partial / name)

    report: dict[str, object] = {
        "inputs": {
            "template": {"root": str(template_root), "digests": digests(template)},
            "tower": {"root": str(tower_root), "digests": digests(tower)},
        },
        "tensors": {"from_tower": taken, "copied": copied, "total": len(template.shards)},
        "recast_to_template_dtype": {"count": len(recast), "first": sorted(recast)[:SHOWN]},
        "files_copied": present,
        "outputs": {shard: file_digest(partial / shard) for shard in template.files()},
        "code": code_stamp(),
    }
    (partial / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.rename(out)
    return report
