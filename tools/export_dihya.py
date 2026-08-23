"""Export Dihya-5M from its training checkpoint into a publishable directory.

`best.pt` is 63.4 MB of resumable state — optimizer moments, RNG, the group-DRO weights
and the contrastive memory bank. The model a downloader loads is 20.4 MB of it. This
writes that, in safetensors, under the key names `agbalu.hub.dihya` declares, with the
calibration folded into `config.json` so a downloader who reads only the weights still
gets the prior shift and the conformal quantile the reported numbers were measured with.

    python3 -m tools.export_dihya --run artifacts/runs/dihya --out artifacts/release/Dihya-5M
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from torch import Tensor

MAX_LOGIT_DELTA: Final = 1e-5
"""The exported model is a re-implementation, not the trained object. Agreement to this
is what distinguishes a correct port from a plausible one — a wrong `eps`, a dropped
residual or a transposed head all still produce well-formed logits."""

PROBES: Final[tuple[str, ...]] = (
    "Azul fell-awen, amek tettilim ass-a?",
    "Mayemmi ur teccid ticemrarin nnecm a gma?",
    "Bonjour tout le monde, comment allez-vous?",
)
"""Kabyle, Tarifit and French. A probe of one language cannot fail on a head that
collapsed to a single class."""


class ExportError(Exception):
    """A run directory that cannot produce a loadable release."""


@dataclass(frozen=True, slots=True)
class Staged:
    """One written file."""

    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """What the export wrote, and what it dropped to get there."""

    source: str
    source_bytes: int
    dropped_training_only: list[str]
    tensors: int
    parameters: int
    max_logit_delta_against_trained: float
    files: list[Staged] = field(default_factory=list)


def hub_config(run: Path) -> dict[str, object]:
    """The training config and the calibration, as `DihyaConfig`'s fields.

    The calibration is not a sidecar the loader may or may not read: without the prior
    shift the argmax minimises the training prior's error rather than balanced error, so
    a repository shipping the weights alone ships a model that scores differently from
    the one the card reports.
    """
    from agbalu.hub.dihya.configuration_dihya import DihyaConfig

    stored = json.loads((run / "config.json").read_text(encoding="utf-8"))["model"]
    calibration_path = run / "calibration.json"
    if not calibration_path.is_file():
        message = f"no calibration.json in {run}; the prediction sets have no quantile"
        raise ExportError(message)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    if list(calibration["classes"]) != list(stored["classes"]):
        message = "calibration and model disagree on the class order"
        raise ExportError(message)

    config = DihyaConfig(
        vocab_size=int(stored["vocab_size"]),
        hidden_size=int(stored["embed_dim"]),
        intermediate_size=int(stored["intermediate_dim"]),
        num_attention_heads=int(stored["num_heads"]),
        num_hidden_layers=int(stored["num_layers"]),
        conv_kernels=[int(k) for k in stored["conv_kernels"]],
        conv_dim=int(stored["conv_dim"]),
        max_position_embeddings=int(stored["max_length"]),
        dropout_prob=float(stored["dropout"]),
        logit_scale=float(stored["scale"]),
        classes=[str(name) for name in stored["classes"]],
        prior_shift=[float(x) for x in calibration["prior_shift"]],
        q_hat=float(calibration["q_hat"]),
        target_coverage=float(calibration["target_coverage"]),
    )
    return config.to_dict()


def release_weights(run: Path) -> tuple[dict[str, Tensor], list[str]]:
    """The trunk and head under the published module's key names, and what was dropped.

    Two things go. The contrastive projection shapes the trunk during training and no
    forward pass reads it at inference. `head.margins` is additive on the target logit
    only, so it is identity without labels — both would report unexpected keys to every
    downloader.
    """
    import torch

    from agbalu.lid.infer import exported_weights

    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if "model" not in payload:
        message = f"{run / 'best.pt'} holds {sorted(payload)}, with no model state"
        raise ExportError(message)
    trunk = exported_weights(payload["model"])
    kept = {
        name: tensor.contiguous()
        for name, tensor in trunk.items()
        if not name.startswith("projection.") and name != "head.margins"
    }
    if not kept:
        message = "the checkpoint carried no trunk weights"
        raise ExportError(message)
    return kept, sorted(set(trunk) - set(kept))


def check_agreement(run: Path, out: Path) -> float:
    """Largest absolute logit difference between the trained model and the export.

    Raises:
        ExportError: the two disagree by more than `MAX_LOGIT_DELTA`.
    """
    import torch

    from agbalu.hub.dihya.modeling_dihya import DihyaForSequenceClassification
    from agbalu.lid.config import PAD_ID
    from agbalu.lid.dataset import encode
    from agbalu.lid.infer import Identifier

    trained = Identifier.load(run).model
    released = DihyaForSequenceClassification.from_pretrained(out).eval()

    rows = [encode(text, trained.config.max_length) for text in PROBES]
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    pad_mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        pad_mask[index, : len(row)] = True

    with torch.inference_mode():
        expected = trained(input_ids, pad_mask=pad_mask).logits
        actual = released(input_ids, attention_mask=pad_mask.long()).logits
    delta = float((expected - actual).abs().max())
    if delta > MAX_LOGIT_DELTA:
        message = (
            f"the exported model disagrees with the trained one by {delta:.3e} "
            f"(limit {MAX_LOGIT_DELTA:.0e}); the port is wrong, not the tolerance"
        )
        raise ExportError(message)
    return delta


def stage(run: Path, out: Path, card: Path) -> Manifest:
    """Write the release directory, after every check has passed."""
    from safetensors.torch import save_file

    for name in ("best.pt", "config.json", "calibration.json", "held-out-report.json"):
        if not (run / name).is_file():
            message = f"{run / name} is missing — pull the run with `make modal-lid-pull`"
            raise ExportError(message)
    if not card.is_file():
        message = f"card not found at {card}"
        raise ExportError(message)

    config = hub_config(run)
    kept, dropped = release_weights(run)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    save_file(kept, out / "model.safetensors")
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("calibration.json", "held-out-report.json"):
        shutil.copyfile(run / name, out / name)
    shutil.copyfile(card, out / "README.md")

    delta = check_agreement(run, out)

    manifest = Manifest(
        source=str(run / "best.pt"),
        source_bytes=(run / "best.pt").stat().st_size,
        dropped_training_only=dropped,
        tensors=len(kept),
        parameters=sum(int(tensor.numel()) for tensor in kept.values()),
        max_logit_delta_against_trained=delta,
        files=[
            Staged(
                name=entry.name,
                bytes=entry.stat().st_size,
                sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
            )
            for entry in sorted(out.iterdir())
        ],
    )
    (out / "export.stats.json").write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Dihya-5M for publication.")
    parser.add_argument("--run", type=Path, default=Path("artifacts/runs/dihya"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/release/Dihya-5M"))
    parser.add_argument("--card", type=Path, default=Path("docs/cards/dihya-5m.md"))
    args = parser.parse_args(argv)

    try:
        manifest = stage(args.run, args.out, args.card)
    except ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 1

    print(
        f"{args.out}: {manifest.parameters:,} parameters in {manifest.tensors} tensors, "
        f"agreeing with the trained model to {manifest.max_logit_delta_against_trained:.2e}"
    )
    for entry in manifest.files:
        print(f"  {entry.name:<24} {entry.bytes:>12,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
