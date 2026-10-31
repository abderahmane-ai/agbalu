"""Score a frozen encoder snapshot on the UD_Kabyle-ADPT test split, beside its baselines.

The linear probe is fitted on the full treebank train split, exactly as
`python -m agbalu.bench.cli pos` fits the published encoder's, and every system is scored
by the same `agbalu.bench.pos` harness under the same four conditions. The published
Masinissa-31M is re-measured here rather than quoted, so the comparison cannot differ in
procedure, and its numbers must reproduce the ones its card reports.

    python -m tools.eval_encoder_test --snapshot artifacts/runs/encoder-pretrain/step-0027231-... \\
        --out artifacts/encoder/measurements/pos-test.stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from agbalu.bench import pos
from agbalu.bench.probe import Encoder as ProbeEncoder
from agbalu.bench.probe import ProbeConfig, ProbeTagger
from agbalu.bench.taggers import MostFrequentTagger
from agbalu.encoder import checkpoint
from agbalu.encoder.configuration import Geometry, Locations
from agbalu.encoder.corpus import tokenizer as load_tokenizer
from agbalu.encoder.model import Encoder
from agbalu.llm.provenance import file_digest
from agbalu.model.config import PRESETS
from agbalu.model.modeling import Encoder as PublishedEncoder
from agbalu.normalise.rules import load_rules
from agbalu.treebank import read_split

PUBLISHED_SHA256 = "dca2a960c06a4bb14a5dd1d33557b8b4537230807a32d0a4debbee7fda44fbad"
PUNCTUATION = frozenset({"PUNCT"})


def freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def new_encoder(snapshot: Path, device: torch.device) -> Encoder:
    checkpoint.verify_snapshot(snapshot)
    geometry = Geometry.model_validate(json.loads((snapshot / "geometry.json").read_text()))
    model = Encoder(geometry.model_copy(update={"backend": "reference"})).to(device)
    payload = torch.load(snapshot / "training.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["weights"], strict=True)
    freeze(model)
    return model


def published_encoder(device: torch.device) -> PublishedEncoder:
    path = Path(hf_hub_download("agbalu/Masinissa-31M", "model.safetensors"))
    if file_digest(path) != PUBLISHED_SHA256:
        message = f"{path}: published weights differ from the digest on the card"
        raise ValueError(message)
    model = PublishedEncoder(PRESETS["kab"]).to(device)
    model.load_state_dict(load_file(path, device=str(device)), strict=False)
    freeze(model)
    return model


def report(tagger: pos.Tagger, test: list[object], rules: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for setting in pos.SETTINGS:
        for condition in pos.CONDITIONS:
            items = pos.items_for(test, setting, condition, rules)  # type: ignore[arg-type]
            run = pos.run(tagger, items, setting, condition)
            everything = pos.score(run)
            lexical = pos.score(run, PUNCTUATION)
            result[f"{setting}/{condition}"] = {
                "accuracy": everything.accuracy,
                "nonpunct_accuracy": lexical.accuracy,
                "macro_f1": everything.macro_f1,
                "macro_f1_nonpunct": lexical.macro_f1,
                "scored": everything.scored,
                "unscorable": run.unscorable,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    locations = Locations()
    train = read_split(locations.treebank, "train")
    test = read_split(locations.treebank, "test")
    vocabulary = load_tokenizer(locations.tokenizer)
    rules = load_rules()
    systems: dict[str, object] = {}

    candidates: tuple[tuple[str, ProbeEncoder], ...] = (
        ("candidate", new_encoder(args.snapshot, device)),
        ("masinissa-31m", published_encoder(device)),
    )
    for name, encoder in candidates:
        tagger = ProbeTagger.fit(
            encoder, vocabulary, train, device=device, step=0, config=ProbeConfig(epochs=20)
        )
        systems[name] = report(tagger, test, rules)  # type: ignore[arg-type]
    systems["most-frequent-tag"] = report(MostFrequentTagger.fit(train), test, rules)  # type: ignore[arg-type]

    payload = {
        "split": "test",
        "sentences": len(test),
        "snapshot": args.snapshot.name,
        "snapshot_sha256": file_digest(args.snapshot / "training.pt"),
        "probe": {"epochs": 20, "seed": 0, "fitted_on": "train", "train_sentences": len(train)},
        "systems": systems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
