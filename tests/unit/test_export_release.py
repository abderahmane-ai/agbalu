"""Turning a training checkpoint into a publishable directory.

A wrong export is a wrong public artifact, so every property the release depends on is
asserted here: that the weights reload into the real module, that the two omissions are safe
omissions, and that the manifest's checksums describe the bytes actually written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tools.export_release import DERIVED, TIED, export, sha256, weights_of

from agbalu.model.config import PRESETS, Preset
from agbalu.model.modeling import Encoder

PRESET: Preset = "small"
"""The 8,192-vocabulary reference config: same architecture, a third of the export time."""

Manifest = dict[str, object]


def summary(manifest: Manifest) -> Manifest:
    training = manifest["training"]
    assert isinstance(training, dict)
    return training


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real checkpoint of the real module, in the shape the trainer writes."""
    torch.manual_seed(0)
    model = Encoder(PRESETS[PRESET])
    path = tmp_path_factory.mktemp("run") / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {"state": {i: {"m": torch.zeros(4)} for i in range(3)}},
            "rng": torch.get_rng_state(),
            "state": {"step": 20, "epoch": 1, "tokens_seen": 1234, "best_validation_loss": 5.5},
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def exported(checkpoint: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    out = tmp_path_factory.mktemp("release") / "Model"
    manifest = export(checkpoint, out, PRESET)
    return out, manifest


def test_the_weights_reload_into_the_real_module(exported: tuple[Path, Manifest]) -> None:
    out, _ = exported
    model = Encoder(PRESETS[PRESET])
    missing, unexpected = model.load_state_dict(load_file(out / "model.safetensors"), strict=False)
    assert not unexpected
    assert all(key == TIED or key.endswith(DERIVED) for key in missing)


def test_the_reloaded_model_produces_the_checkpoint_s_outputs(
    checkpoint: Path, exported: tuple[Path, Manifest]
) -> None:
    """The point of the export is that the published weights *are* the trained weights."""
    out, _ = exported
    original = Encoder(PRESETS[PRESET])
    original.load_state_dict(torch.load(checkpoint, weights_only=False)["model"])
    published = Encoder(PRESETS[PRESET])
    published.load_state_dict(load_file(out / "model.safetensors"), strict=False)
    for a, b in zip(original.parameters(), published.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_tied_decoder_is_not_written_twice(exported: tuple[Path, Manifest]) -> None:
    """It shares storage with the embedding table; safetensors refuses shared tensors."""
    out, _ = exported
    assert TIED not in load_file(out / "model.safetensors")


def test_the_derived_index_buffers_are_not_shipped(exported: tuple[Path, Manifest]) -> None:
    out, _ = exported
    assert not [k for k in load_file(out / "model.safetensors") if k.endswith(DERIVED)]


def test_the_module_rebuilds_the_dropped_buffers_identically(checkpoint: Path) -> None:
    """The assertion the export rests on: dropping them changes nothing on load."""
    saved = torch.load(checkpoint, weights_only=False)["model"]
    fresh = dict(Encoder(PRESETS[PRESET]).named_buffers())
    derived = [k for k in saved if k.endswith(DERIVED)]
    assert derived
    assert all(torch.equal(fresh[k], saved[k]) for k in derived)


def test_the_optimizer_state_is_dropped(checkpoint: Path, exported: tuple[Path, Manifest]) -> None:
    out, _ = exported
    published = (out / "model.safetensors").stat().st_size
    assert published < checkpoint.stat().st_size


def test_the_manifest_checksum_describes_the_bytes_written(exported: tuple[Path, Manifest]) -> None:
    out, manifest = exported
    assert manifest["weights_sha256"] == sha256(out / "model.safetensors")


def test_the_manifest_reconstructs_the_architecture(exported: tuple[Path, Manifest]) -> None:
    _, manifest = exported
    config = PRESETS[PRESET]
    assert manifest["parameters"] == config.parameters
    assert manifest["vocab_size"] == config.vocab_size
    assert manifest["num_hidden_layers"] == config.num_hidden_layers


def test_the_training_summary_survives_the_export(exported: tuple[Path, Manifest]) -> None:
    _, manifest = exported
    training = summary(manifest)
    assert training["step"] == 20
    assert training["best_validation_loss"] == 5.5


def test_config_json_is_valid_utf8_json(exported: tuple[Path, Manifest]) -> None:
    out, manifest = exported
    assert json.loads((out / "config.json").read_text(encoding="utf-8")) == manifest


def test_a_card_is_copied_in_as_readme(checkpoint: Path, tmp_path: Path) -> None:
    card = tmp_path / "card.md"
    card.write_text("# Azul\n\nɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ\n", encoding="utf-8")
    out = tmp_path / "with-card"
    export(checkpoint, out, PRESET, card)
    assert (out / "README.md").read_text(encoding="utf-8").endswith("ţ\n")


def test_a_tokenizer_travels_with_the_weights(checkpoint: Path, tmp_path: Path) -> None:
    """A model is unusable without the vocabulary it was trained on."""
    vocab = tmp_path / "vocab.model"
    vocab.write_bytes(b"\x00sentencepiece-ish")
    out = tmp_path / "with-vocab"
    manifest = export(checkpoint, out, PRESET, None, vocab)
    assert (out / "vocab.model").read_bytes() == vocab.read_bytes()
    assert manifest["tokenizer"] == {"file": "vocab.model", "sha256": sha256(vocab)}


def test_a_checkpoint_whose_model_is_not_a_state_dict_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model": ["not", "a", "dict"], "state": {}}, path)
    with pytest.raises(TypeError, match="not a state dict"):
        weights_of(torch.load(path, weights_only=False))


def test_weights_for_the_wrong_preset_are_refused(checkpoint: Path, tmp_path: Path) -> None:
    """`small` and `kab` differ only in vocabulary size, which is exactly the mistake that
    would otherwise publish a model whose config lies about it."""
    out = tmp_path / "mismatched"
    with pytest.raises(ValueError, match="not a 'kab' model"):
        export(checkpoint, out, "kab")
    assert not out.exists(), "a refused export must not leave a directory that looks published"
