"""Reload staged merged speech weights independently from the source graph."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from agbalu.codec_tts.adaptation import adapter_state, inject_adapter
from agbalu.codec_tts.backbone import Generator, load_generator
from agbalu.codec_tts.configuration import Geometry, Recipe
from agbalu.codec_tts.release import export_model

pytestmark = pytest.mark.integration


def test_export_preserves_source_and_reloads_merged_weights(tmp_path: Path) -> None:
    geometry = Geometry(
        hidden_size=16,
        intermediate_size=24,
        layers=1,
        heads=2,
        kv_heads=1,
        head_dim=8,
        text_vocabulary=32,
    )
    model = Generator(geometry)
    source = tmp_path / "base"
    source.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        native_name = name.replace("body.embed_tokens.", "tied.embedding.text_embedding.")
        native_name = native_name.replace(
            "audio_embedding.", "tied.embedding.modality_embeddings.0.embedding."
        )
        tensors[native_name] = value.detach().clone()
    tensors["tied.head.text_head.weight"] = tensors["tied.embedding.text_embedding.weight"].clone()
    tensors["tied.head.modality_heads.0.weight"] = tensors[
        "tied.embedding.modality_embeddings.0.embedding.weight"
    ].clone()
    save_file(tensors, source / "model.safetensors")
    inject_adapter(model)
    original = adapter_state(model)
    recipe = Recipe(base_directory=source)
    destination = tmp_path / "export"
    report = export_model(model, recipe, destination)
    assert report["dropped_native_keys"] == []
    assert (destination / "src/agbalu/codec_tts/cli.py").is_file()
    assert (destination / "STAMP.json").is_file()
    reloaded, loaded = load_generator(
        destination / "model.safetensors", geometry, torch.device("cpu")
    )
    assert loaded["ties_verified"] is True
    assert torch.equal(model.audio_embedding.weight, reloaded.audio_embedding.weight)
    assert export_model(model, recipe, destination) == report
    for name, tensor in adapter_state(model).items():
        assert torch.equal(tensor, original[name])
