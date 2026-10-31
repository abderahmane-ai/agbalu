"""Evaluate fixed masked targets and frozen downstream probes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from agbalu.bench import pos
from agbalu.bench.probe import Encoder as ProbeEncoder
from agbalu.bench.probe import ProbeConfig, ProbeTagger
from agbalu.encoder.attention import make_layout
from agbalu.encoder.batches import Batches
from agbalu.encoder.configuration import Locations
from agbalu.encoder.corpus import tokenizer
from agbalu.encoder.masking import mask_groups
from agbalu.encoder.model import Encoder, Layer
from agbalu.encoder.previews import MASK_SEED, PREVIEW_TOKENS, SENTENCES, prepare_sentences
from agbalu.llm.provenance import file_digest
from agbalu.model.config import PRESETS
from agbalu.model.modeling import Encoder as PublishedEncoder
from agbalu.normalise.rules import load_rules
from agbalu.treebank import Sentence, read_split


def development_split(root: Path) -> tuple[list[Sentence], list[Sentence]]:
    train: list[Sentence] = []
    dev: list[Sentence] = []
    for sentence in read_split(root, "train"):
        # Identical text always lands in the same partition, independently of row order.
        digest = hashlib.sha256(sentence.text.encode("utf-8")).digest()
        (dev if int.from_bytes(digest[:4], "big") % 5 == 0 else train).append(sentence)
    if not train or not dev:
        message = f"{root}: need nonempty train-derived probe train and dev partitions"
        raise ValueError(message)
    return train, dev


@torch.inference_mode()
def fixed_mlm(model: Encoder, batches: Batches, device: torch.device) -> dict[str, object]:
    was_training = model.training
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for index in range(len(batches)):
        batch = batches[index]
        ids = batch.ids.to(device)
        groups = batch.groups.to(device)
        masked = mask_groups(ids, groups, 0.2, 1000 + index, model.config.vocab_size)
        layout = make_layout(
            batch.documents.to(device),
            batch.positions.to(device),
            batch.valid.to(device),
            flex=model.config.backend == "flex",
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, correct = model.forward(masked.ids, layout, masked.labels)
        totals += torch.stack((loss.double(), correct.double(), masked.targets.double()))
    loss_sum, correct_sum, targets = totals.tolist()
    model.train(was_training)
    if not targets:
        message = "fixed validation produced zero targets"
        raise ValueError(message)
    return {
        "ce": loss_sum / targets,
        "accuracy": correct_sum / targets,
        "targets": int(targets),
        "mask_seed": 1000,
        "mask_probability": 0.2,
    }


@torch.inference_mode()
def preview(
    model: Encoder, batches: Batches | None, locations: Locations, device: torch.device
) -> tuple[dict[str, str], dict[str, object]]:
    del batches
    processor = tokenizer(locations.tokenizer)
    examples = prepare_sentences(processor, model.config.vocab_size)
    ids = torch.zeros(PREVIEW_TOKENS, dtype=torch.long)
    documents = torch.full_like(ids, -1)
    positions = torch.zeros_like(ids)
    selected = torch.zeros(PREVIEW_TOKENS, dtype=torch.bool)
    cursor = 0
    for index, example in enumerate(examples):
        end = cursor + example.original.numel()
        ids[cursor:end] = example.masked
        documents[cursor:end] = index
        positions[cursor:end] = torch.arange(end - cursor)
        selected[cursor:end] = example.selected
        cursor = end
    valid = documents >= 0
    layout = make_layout(
        documents.to(device),
        positions.to(device),
        valid.to(device),
        flex=model.config.backend == "flex",
    )
    captured: dict[str, list[torch.Tensor]] = {"activation_rms": [], "routing_entropy": []}

    def observe(name: str, value: torch.Tensor) -> None:
        captured[name].append(value)

    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.contextualise_packed(ids.to(device), layout, observer=observe)
            logits = model.logits(hidden[selected.to(device)])
    finally:
        model.train(was_training)
    restored = ids.clone()
    restored[selected] = logits.argmax(-1).cpu()
    texts: dict[str, str] = {}
    cursor = 0
    for index, example in enumerate(examples, 1):
        end = cursor + example.original.numel()
        texts[f"sentence {index} original"] = example.text
        texts[f"sentence {index} masked"] = processor.decode(example.masked.tolist())
        texts[f"sentence {index} restored"] = processor.decode(restored[cursor:end].tolist())
        cursor = end
    diagnostic: dict[str, object] = {
        name: torch.stack(values).tolist() for name, values in captured.items()
    }
    diagnostic["relative_embedding_rms"] = float(model.relative.float().square().mean().sqrt())
    diagnostic["preview_set_sha256"] = hashlib.sha256("\n".join(SENTENCES).encode()).hexdigest()
    diagnostic["preview_mask_seed"] = MASK_SEED
    diagnostic["qk_scale_rms"] = [
        float(
            torch.cat((layer.attention.q_norm.weight, layer.attention.k_norm.weight))
            .float()
            .square()
            .mean()
            .sqrt()
        )
        for layer in model.layers
        if isinstance(layer, Layer)
    ]
    return texts, diagnostic


def probe_scores(
    encoder: ProbeEncoder, locations: Locations, device: torch.device, *, seed: int = 0
) -> dict[str, object]:
    train, dev = development_split(locations.treebank)
    tagger = ProbeTagger.fit(
        encoder,
        tokenizer(locations.tokenizer),
        train,
        device=device,
        step=0,
        config=ProbeConfig(seed=seed),
        cache_features=True,
    )
    rules = load_rules()
    results: dict[str, object] = {
        "train_sentences": len(train),
        "dev_sentences": len(dev),
        "head_seed": seed,
        "probe_epochs": 20,
    }
    for setting in pos.SETTINGS:
        for condition in pos.CONDITIONS:
            items = pos.items_for(dev, setting, condition, rules)
            run = pos.run(tagger, items, setting, condition)
            all_tags = pos.score(run)
            lexical = pos.score(run, frozenset({"PUNCT"}))
            results[f"{setting}/{condition}"] = {
                "accuracy": all_tags.accuracy,
                "nonpunct_accuracy": lexical.accuracy,
                "macro_f1": all_tags.macro_f1,
                "scored": all_tags.scored,
                "unscorable": run.unscorable,
                "by_label": [
                    {"label": item.label, "support": item.support, "f1": item.f1}
                    for item in all_tags.label_scores()
                ],
            }
    return results


def downstream(model: Encoder, locations: Locations, device: torch.device) -> dict[str, object]:
    reference = Encoder(model.config.model_copy(update={"backend": "reference"})).to(device)
    reference.load_state_dict(model.state_dict(), strict=True)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    # Dense reference evaluation avoids compiling a separate GPU graph for every sentence length.
    return probe_scores(reference, locations, device)


def primary_scores(report: dict[str, object]) -> tuple[float, float]:
    values: list[float] = []
    for name in ("surface/canonical", "surface/corrupted"):
        part = report[name]
        if not isinstance(part, dict) or not isinstance(part.get("nonpunct_accuracy"), int | float):
            message = f"missing finite primary metric in {name}: {part!r}"
            raise TypeError(message)
        values.append(float(part["nonpunct_accuracy"]))
    return values[0], values[1]


def baseline(locations: Locations, device: torch.device) -> dict[str, object]:
    path = Path(hf_hub_download("agbalu/Masinissa-31M", "model.safetensors"))
    expected = "dca2a960c06a4bb14a5dd1d33557b8b4537230807a32d0a4debbee7fda44fbad"
    actual = file_digest(path)
    if actual != expected:
        message = f"{path}: published baseline SHA256 {actual} != {expected}"
        raise ValueError(message)
    model = PublishedEncoder(PRESETS["kab"]).to(device)
    loaded = model.load_state_dict(load_file(path, device=str(device)), strict=False)
    allowed = {"classifier.decoder.weight"} | {
        f"transformer.attention_layers.{index}.position_indices" for index in range(12)
    }
    if set(loaded.missing_keys) - allowed or loaded.unexpected_keys:
        message = f"published baseline key mismatch: {loaded}"
        raise ValueError(message)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "model": "agbalu/Masinissa-31M",
        "weights_sha256": actual,
        **probe_scores(model, locations, device),
    }
