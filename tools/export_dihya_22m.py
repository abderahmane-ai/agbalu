"""Export Dihya-22M from its training checkpoint into a publishable directory.

`best.pt` is ~88.8 MB of training state — optimizer moments, the group-DRO weights,
LDAM margin buffers, auxiliary heads, and the contrastive memory bank.
The published model is 21,999,616 parameters in safetensors (~88.0 MB), under the key
names `agbalu.hub.dihya` declares, with split-conformal calibration folded into `config.json`.

Usage:
    python3 -m tools.export_dihya_22m \
        --run artifacts/runs/dihya-22m-v2 \
        --out artifacts/release/Dihya-22M \
        --card docs/cards/dihya-22m.md
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
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from torch import Tensor

MAX_LOGIT_DELTA: Final = 1e-5

PROBES: Final[tuple[tuple[str, str], ...]] = (
    ("Azul fell-awen, amek tettilim ass-a?", "kab_Latn"),
    ("Mayemmi ur teccid ticemrarin nnecm a gma?", "rif_Latn"),
    ("Bonjour tout le monde, comment allez-vous?", "fra_Latn"),
    ("ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ.", "kab_Tfng"),
    ("Hello everyone, how are you doing today?", "eng_Latn"),
    ("صباح الخير كيداير لاباس عليك", "ary_Arab"),
    ("واش راك خويا لاباس؟", "arq_Arab"),
)

DROPPED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "head.margins",
        "script_head.weight",
        "filter_head.weight",
        "projection.0.weight",
        "projection.2.weight",
    }
)


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
    """Build the published Hugging Face config dictionary."""
    from agbalu.dihya.config import ModelConfig
    from agbalu.hub.dihya.configuration_dihya import DihyaConfig

    m_cfg = ModelConfig()

    cal_path = run / "calibration.stats.json"
    if not cal_path.is_file():
        message = f"no calibration.stats.json in {run}"
        raise ExportError(message)
    calibration = json.loads(cal_path.read_text(encoding="utf-8"))

    config = DihyaConfig(
        vocab_size=m_cfg.vocab_size,
        hidden_size=m_cfg.embed_dim,
        intermediate_size=m_cfg.intermediate_dim,
        num_attention_heads=m_cfg.num_heads,
        num_hidden_layers=m_cfg.num_layers,
        conv_kernels=[int(k) for k in m_cfg.conv_kernels],
        conv_dim=m_cfg.conv_dim,
        max_position_embeddings=m_cfg.max_length,
        dropout_prob=m_cfg.dropout,
        logit_scale=m_cfg.scale,
        classes=list(m_cfg.classes),
        prior_shift=[0.0] * len(m_cfg.classes),
        q_hat=float(calibration["q_hat"]),
        target_coverage=0.99,
    )
    result = config.to_dict()
    result["architectures"] = ["DihyaForSequenceClassification"]
    result["auto_map"] = {
        "AutoConfig": "configuration_dihya.DihyaConfig",
        "AutoModel": "modeling_dihya.DihyaForSequenceClassification",
        "AutoModelForSequenceClassification": "modeling_dihya.DihyaForSequenceClassification",
    }
    return result


def release_weights(run: Path) -> tuple[dict[str, Tensor], list[str]]:
    """Filter out auxiliary training heads and LDAM margins."""
    import torch

    ckpt_path = run / "best.pt"
    if not ckpt_path.is_file():
        message = f"checkpoint not found: {ckpt_path}"
        raise ExportError(message)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    kept = {
        name: tensor.contiguous() for name, tensor in payload.items() if name not in DROPPED_KEYS
    }
    dropped = sorted(set(payload) - set(kept))
    return kept, dropped


def check_agreement(run: Path, out: Path) -> float:
    """Verify logit parity between trained Classifier and exported Hub model."""
    import torch

    from agbalu.dihya.config import ModelConfig
    from agbalu.dihya.model import Classifier
    from agbalu.hub.dihya.modeling_dihya import DihyaForSequenceClassification

    m_cfg = ModelConfig()
    trained = Classifier(m_cfg)
    trained.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    trained.eval()

    released = DihyaForSequenceClassification.from_pretrained(out)
    cast("torch.nn.Module", released).eval()

    offset = 2
    probe_texts = [text for text, _ in PROBES]
    rows = [[b + offset for b in text.encode("utf-8")] for text in probe_texts]
    width = max(len(row) for row in rows)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long)
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
            f"exported model disagrees with trained one by {delta:.3e} "
            f"(limit {MAX_LOGIT_DELTA:.0e})"
        )
        raise ExportError(message)

    predictions = released.identify(probe_texts)
    for (text, expected_lang), got in zip(PROBES, predictions, strict=True):
        if got["language"] != expected_lang:
            message = f"probe failed: {text!r} -> {got['language']}, expected {expected_lang}"
            raise ExportError(message)

    return delta


def stage(run: Path, out: Path, card: Path) -> Manifest:
    """Write the complete, self-contained Hugging Face release directory."""
    from safetensors.torch import save_file

    from tools.stage_hub import build_byte_tokenizer

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

    hub_src = Path("src/agbalu/hub/dihya")
    shutil.copyfile(hub_src / "configuration_dihya.py", out / "configuration_dihya.py")
    shutil.copyfile(hub_src / "modeling_dihya.py", out / "modeling_dihya.py")

    tokenizer = build_byte_tokenizer()
    tokenizer.save_pretrained(out)

    if (run / "calibration.stats.json").is_file():
        shutil.copyfile(run / "calibration.stats.json", out / "calibration.json")
    if (run / "test-report.stats.json").is_file():
        shutil.copyfile(run / "test-report.stats.json", out / "test-report.json")

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
    parser = argparse.ArgumentParser(description="Export Dihya-22M for publication.")
    parser.add_argument("--run", type=Path, default=Path("artifacts/runs/dihya-22m-v2"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/release/Dihya-22M"))
    parser.add_argument("--card", type=Path, default=Path("docs/cards/dihya-22m.md"))
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
        print(f"  {entry.name:<28} {entry.bytes:>12,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
