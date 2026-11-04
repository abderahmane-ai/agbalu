"""Load native weight names without leaving derived buffers on meta."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from agbalu.codec_tts.backbone import Generator, load_generator
from agbalu.codec_tts.configuration import Geometry

pytestmark = pytest.mark.integration


def test_native_checkpoint_round_trip(tmp_path: Path) -> None:
    geometry = Geometry(
        hidden_size=16,
        intermediate_size=24,
        layers=1,
        heads=2,
        kv_heads=1,
        head_dim=8,
        text_vocabulary=32,
    )
    original = Generator(geometry)
    native: dict[str, torch.Tensor] = {}
    for name, value in original.state_dict().items():
        native_name = name
        if name.startswith("body.embed_tokens."):
            native_name = name.replace("body.embed_tokens.", "tied.embedding.text_embedding.")
        elif name.startswith("audio_embedding."):
            native_name = name.replace(
                "audio_embedding.", "tied.embedding.modality_embeddings.0.embedding."
            )
        native[native_name] = value.detach().clone()
    native["tied.head.text_head.weight"] = native["tied.embedding.text_embedding.weight"].clone()
    native["tied.head.modality_heads.0.weight"] = native[
        "tied.embedding.modality_embeddings.0.embedding.weight"
    ].clone()
    path = tmp_path / "model.safetensors"
    save_file(native, path)
    loaded, report = load_generator(path, geometry, torch.device("cpu"))
    assert report["ties_verified"] is True
    assert all(not tensor.is_meta for tensor in loaded.buffers())
    for name, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    assert loaded.audio_embedding.weight.requires_grad is False
