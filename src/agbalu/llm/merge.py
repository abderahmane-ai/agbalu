"""The CPT delta added to the post-trained model, one coefficient at a time (task 11.21).

`θ(α) = I + α·(C − B)` for every text-tower tensor, computed in fp32 and stored in I's dtype.
Vision, MTP and everything else CPT never trained are copied from I byte for byte. B, C and I
carry the same 426 text-tower names under `model.language_model.` (read from the pod's
checkpoints on 2026-10-16), so nothing is renamed: the three key sets are proved equal and a
difference is refused by name. The output keeps I's shard layout and files, so it loads as the
full vision-language model and as the text tower alone.

At α = 0, and whenever C equals B, the output is I exactly: adding a zero to an fp32 copy of a
bf16 value and rounding back returns the same bits. `tests/unit/test_llm_merge.py` holds both.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from agbalu.llm.progress import Tracker
from agbalu.llm.provenance import code_stamp, file_digest

if TYPE_CHECKING:
    import torch
    from safetensors import safe_open

TEXT_PREFIX: Final = "model.language_model."
INDEX_FILE: Final = "model.safetensors.index.json"
SINGLE_FILE: Final = "model.safetensors"
REPORT_FILE: Final = "merge.json"
COPIED_FILES: Final = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
"""I's own files beside the weights. Each one I has is copied; none is required but the
config, so a cache that never downloaded the image processor still merges."""

TOKENIZER_FIELDS: Final = ("model", "added_tokens")
"""What must match between B's and I's `tokenizer.json`: the vocabulary and merges, and every
added token with its id. The chat template may differ and comes from I."""

ALPHA_RANGE: Final = (0.0, 1.0)
"""Task arithmetic collapsed near α 1.5 and was robust in [0.5, 1.0) (arXiv 2603.28263)."""

ALPHAS: Final = (0.0, 0.25, 0.5, 0.75, 1.0)
"""The sweep `docs/llm_post_training_design.md` §3 fixes; 0 is the unchanged control."""

SHOWN: Final = 10
"""Names listed in an error before the rest are counted."""


class MergeError(Exception):
    """The three checkpoints cannot be merged, or a merged tensor is not finite."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A safetensors checkpoint directory: which shard holds each tensor, and its config."""

    root: Path
    shards: dict[str, str]
    config: dict[str, object]

    @classmethod
    def read(cls, root: Path) -> Checkpoint:
        config_path = root / "config.json"
        if not config_path.is_file():
            message = f"{root}: no config.json"
            raise MergeError(message)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = root / INDEX_FILE
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                message = f"{index}: no weight_map"
                raise MergeError(message)
            shards = {str(name): str(shard) for name, shard in weight_map.items()}
        elif (root / SINGLE_FILE).is_file():
            from safetensors import safe_open

            with safe_open(root / SINGLE_FILE, framework="pt") as handle:
                shards = dict.fromkeys(handle.keys(), SINGLE_FILE)
        else:
            message = f"{root}: neither {INDEX_FILE} nor {SINGLE_FILE}"
            raise MergeError(message)
        missing = sorted({shard for shard in shards.values() if not (root / shard).is_file()})
        if missing:
            message = f"{root}: shards named in the index are absent: {missing}"
            raise MergeError(message)
        return cls(root, shards, config)

    def names(self, prefix: str = "") -> set[str]:
        return {name for name in self.shards if name.startswith(prefix)}

    def files(self) -> list[str]:
        return sorted(set(self.shards.values()))


@dataclass(frozen=True, slots=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str


def listed(names: Sequence[str] | set[str]) -> str:
    ordered = sorted(names)
    shown = ", ".join(ordered[:SHOWN])
    return shown if len(ordered) <= SHOWN else f"{shown} and {len(ordered) - SHOWN} more"


def is_tied(config: Mapping[str, object]) -> bool:
    return config.get("tie_word_embeddings") is True


def tensor_infos(checkpoint: Checkpoint, names: set[str]) -> dict[str, TensorInfo]:
    """Shape and dtype of `names` from the safetensors headers, without reading the data."""
    from safetensors import safe_open

    infos: dict[str, TensorInfo] = {}
    for shard in checkpoint.files():
        wanted = [name for name, file in checkpoint.shards.items() if file == shard]
        wanted = [name for name in wanted if name in names]
        if not wanted:
            continue
        with safe_open(checkpoint.root / shard, framework="pt") as handle:
            for name in wanted:
                view = handle.get_slice(name)
                infos[name] = TensorInfo(tuple(int(n) for n in view.get_shape()), view.get_dtype())
    return infos


def text_tower(base: Checkpoint, cpt: Checkpoint, instruct: Checkpoint) -> list[str]:
    """The tensors to merge, after proving the three checkpoints agree on them exactly."""
    for label, checkpoint in (("base", base), ("cpt", cpt), ("instruct", instruct)):
        if not is_tied(checkpoint.config):
            message = f"{label} {checkpoint.root}: tie_word_embeddings is not true"
            raise MergeError(message)
        if "lm_head.weight" in checkpoint.shards:
            message = f"{label} {checkpoint.root}: an lm_head.weight beside tied embeddings"
            raise MergeError(message)
    outside = cpt.names() - cpt.names(TEXT_PREFIX)
    if outside:
        message = f"cpt {cpt.root} holds tensors outside the text tower: {listed(outside)}"
        raise MergeError(message)
    names = base.names(TEXT_PREFIX)
    if not names:
        message = f"base {base.root}: no tensor under {TEXT_PREFIX!r}"
        raise MergeError(message)
    for label, other in (("cpt", cpt), ("instruct", instruct)):
        theirs = other.names(TEXT_PREFIX)
        if theirs != names:
            message = (
                f"{label} {other.root} text tower differs from base: "
                f"only in base [{listed(names - theirs)}], only in {label} "
                f"[{listed(theirs - names)}]"
            )
            raise MergeError(message)
    reference = tensor_infos(base, names)
    for label, other in (("cpt", cpt), ("instruct", instruct)):
        infos = tensor_infos(other, names)
        differ = [name for name in sorted(names) if infos[name] != reference[name]]
        if differ:
            first = differ[0]
            message = (
                f"{label}: {len(differ)} tensors differ in shape or dtype from base, first "
                f"{first} {infos[first]} against {reference[first]}"
            )
            raise MergeError(message)
    return sorted(names)


def tokenizer_fields(root: Path) -> dict[str, object]:
    path = root / "tokenizer.json"
    if not path.is_file():
        message = f"{root}: no tokenizer.json to compare"
        raise MergeError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {field: payload.get(field) for field in TOKENIZER_FIELDS}


def check_tokenizers(base: Path, instruct: Path) -> None:
    """B and I must tokenise identically, or the delta lands on different token rows."""
    ours, theirs = tokenizer_fields(base), tokenizer_fields(instruct)
    differ = [field for field in TOKENIZER_FIELDS if ours[field] != theirs[field]]
    if differ:
        message = f"tokenizers of {base} and {instruct} differ in {differ}"
        raise MergeError(message)


def merged(
    instruct: torch.Tensor, base: torch.Tensor, cpt: torch.Tensor, alpha: float, name: str
) -> torch.Tensor:
    """`I + α(C − B)` in fp32, returned in I's dtype; a non-finite value is refused."""
    import torch

    value = instruct.float() + alpha * (cpt.float() - base.float())
    if not bool(torch.isfinite(value).all()):
        message = f"{name}: the merged tensor is not finite at α {alpha}"
        raise MergeError(message)
    return value.to(instruct.dtype)


@dataclass(slots=True)
class ChangeTally:
    """How far the merge moved each tensor from I, as ||θ − I|| / ||I||."""

    count: int = 0
    total: float = 0.0
    largest: float = 0.0
    largest_name: str = ""

    def add(self, name: str, before: torch.Tensor, after: torch.Tensor) -> None:
        reference = float(before.float().norm())
        change = float((after.float() - before.float()).norm())
        relative = change / reference if reference > 0 else change
        self.count += 1
        self.total += relative
        if relative > self.largest:
            self.largest, self.largest_name = relative, name

    def as_dict(self) -> dict[str, object]:
        return {
            "tensors": self.count,
            "mean_relative_change": self.total / self.count if self.count else None,
            "largest_relative_change": self.largest,
            "largest_tensor": self.largest_name,
        }


def check_alpha(alpha: float) -> None:
    low, high = ALPHA_RANGE
    if not math.isfinite(alpha) or not low <= alpha <= high:
        message = f"α {alpha} is outside [{low}, {high}]"
        raise MergeError(message)


def digests(checkpoint: Checkpoint) -> dict[str, str]:
    return {shard: file_digest(checkpoint.root / shard) for shard in checkpoint.files()}


@dataclass(frozen=True, slots=True)
class Inputs:
    """B, C and I on disk, and where each came from (repository and revision, or run)."""

    base: Path
    cpt: Path
    instruct: Path
    sources: Mapping[str, str] = field(default_factory=dict)


def merge(inputs: Inputs, alpha: float, out: Path) -> dict[str, object]:
    """Write `θ(α)` to `out` through a hidden partial directory; return the report."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    check_alpha(alpha)
    if out.exists():
        message = f"{out} exists; a merge never overwrites"
        raise MergeError(message)
    base, cpt, instruct = (Checkpoint.read(r) for r in (inputs.base, inputs.cpt, inputs.instruct))
    names = set(text_tower(base, cpt, instruct))
    check_tokenizers(inputs.base, inputs.instruct)

    partial = out.with_name(f".{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    tally = ChangeTally()
    copied = 0
    tracker = Tracker("merge tensors", len(instruct.shards))
    with ExitStack() as stack:
        handles: dict[tuple[str, str], safe_open] = {}

        def opened(checkpoint: Checkpoint, label: str, shard: str) -> safe_open:
            key = (label, shard)
            if key not in handles:
                handles[key] = stack.enter_context(
                    safe_open(checkpoint.root / shard, framework="pt")
                )
            return handles[key]

        for shard in instruct.files():
            tensors: dict[str, torch.Tensor] = {}
            for name in sorted(n for n, file in instruct.shards.items() if file == shard):
                value = opened(instruct, "instruct", shard).get_tensor(name)
                if name in names:
                    old = opened(base, "base", base.shards[name]).get_tensor(name)
                    new = opened(cpt, "cpt", cpt.shards[name]).get_tensor(name)
                    result = merged(value, old, new, alpha, name)
                    tally.add(name, value, result)
                    value = result
                else:
                    copied += 1
                tensors[name] = value
                tracker.advance()
            save_file(tensors, partial / shard, metadata={"format": "pt"})

    if tally.count != len(names) or tally.count + copied != len(instruct.shards):
        message = (
            f"merged {tally.count} of {len(names)} text tensors and copied {copied}; "
            f"I holds {len(instruct.shards)}"
        )
        raise MergeError(message)
    if (instruct.root / INDEX_FILE).is_file():
        shutil.copyfile(instruct.root / INDEX_FILE, partial / INDEX_FILE)
    present = [name for name in COPIED_FILES if (instruct.root / name).is_file()]
    for name in present:
        shutil.copyfile(instruct.root / name, partial / name)

    report: dict[str, object] = {
        "alpha": alpha,
        "formula": "I + alpha * (C - B), fp32, stored in I's dtype",
        "inputs": {
            "base": {"root": str(inputs.base), "digests": digests(base)},
            "cpt": {"root": str(inputs.cpt), "digests": digests(cpt)},
            "instruct": {"root": str(inputs.instruct), "digests": digests(instruct)},
        },
        "sources": dict(inputs.sources),
        "tensors": {"merged": tally.count, "copied": copied, "total": len(instruct.shards)},
        "change": tally.as_dict(),
        "files_copied": present,
        "outputs": {shard: file_digest(partial / shard) for shard in instruct.files()},
        "code": code_stamp(),
    }
    (partial / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.rename(out)
    return report


def same_tensors(first: Path, second: Path) -> list[str]:
    """Tensor names whose bytes differ between two checkpoints; empty when identical."""
    import torch
    from safetensors import safe_open

    left, right = Checkpoint.read(first), Checkpoint.read(second)
    if set(left.shards) != set(right.shards):
        return sorted(set(left.shards) ^ set(right.shards))
    differ: list[str] = []
    for name in sorted(left.shards):
        with (
            safe_open(left.root / left.shards[name], framework="pt") as a,
            safe_open(right.root / right.shards[name], framework="pt") as b,
        ):
            x, y = a.get_tensor(name), b.get_tensor(name)
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                differ.append(name)
    return differ


TIED_HEAD: Final = "lm_head.weight"
"""Present in a loaded model's state dict as a view of the embedding, absent from the file."""

PROBE_TENSOR: Final = f"{TEXT_PREFIX}embed_tokens.weight"
"""Compared by value after loading, which proves the merged numbers, not fresh
initialisation, reached the model."""


def key_gaps(loaded: set[str], stored: set[str]) -> tuple[set[str], set[str]]:
    """Checkpoint tensors the model did not take, and model tensors the checkpoint lacks."""
    return stored - loaded, loaded - stored - {TIED_HEAD}


def verify(out: Path) -> dict[str, object]:
    """Load the merge as the full model and as the text tower; both must take every tensor."""
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    checkpoint = Checkpoint.read(out)
    with safe_open(out / checkpoint.shards[PROBE_TENSOR], framework="pt") as handle:
        probe: torch.Tensor = handle.get_tensor(PROBE_TENSOR)
    expected = {
        "full": checkpoint.names(),
        "text": checkpoint.names(TEXT_PREFIX),
    }
    loaded: dict[str, object] = {}
    for label, factory in (("full", AutoModelForImageTextToText), ("text", AutoModelForCausalLM)):
        model = factory.from_pretrained(out, dtype=torch.bfloat16)
        state = model.state_dict()
        untaken, unfilled = key_gaps(set(state), expected[label])
        if untaken or unfilled:
            message = (
                f"{out} as {label}: checkpoint tensors not loaded [{listed(untaken)}], "
                f"model tensors not in the checkpoint [{listed(unfilled)}]"
            )
            raise MergeError(message)
        if not torch.equal(state[PROBE_TENSOR].cpu(), probe):
            message = f"{out} as {label}: {PROBE_TENSOR} differs from the file after loading"
            raise MergeError(message)
        loaded[label] = {
            "class": type(model).__name__,
            "tensors": len(state),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del model, state
    return loaded
