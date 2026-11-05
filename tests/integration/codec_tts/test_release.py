"""Reload staged merged speech weights independently from the source graph."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    AutoModelForTextToWaveform,
    AutoTokenizer,
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
    PreTrainedTokenizerFast,
)

from agbalu.codec_tts.adaptation import adapter_state, inject_adapter
from agbalu.codec_tts.artifacts import write_json
from agbalu.codec_tts.backbone import Generator, load_generator
from agbalu.codec_tts.configuration import Geometry, Recipe
from agbalu.codec_tts.release import export_model, stage_model, verify_stage

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


def test_staged_adapter_is_exact_portable_and_integrity_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.HF_MODULES_CACHE", str(tmp_path / "modules")
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = ModuleType("transformers_modules")
    monkeypatch.setattr(
        namespace, "__path__", [str(tmp_path / "modules/transformers_modules")], raising=False
    )
    monkeypatch.setitem(sys.modules, "transformers_modules", namespace)
    geometry = Geometry(
        hidden_size=8,
        intermediate_size=12,
        layers=1,
        heads=1,
        kv_heads=1,
        head_dim=8,
        text_vocabulary=16,
    )
    model = Generator(geometry)
    source = tmp_path / "base"
    source.mkdir()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "Azul": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)
    tensors: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        native = name.replace("body.embed_tokens.", "tied.embedding.text_embedding.")
        native = native.replace(
            "audio_embedding.", "tied.embedding.modality_embeddings.0.embedding."
        )
        tensors[native] = value.detach().clone()
    tensors["tied.head.text_head.weight"] = tensors["tied.embedding.text_embedding.weight"].clone()
    tensors["tied.head.modality_heads.0.weight"] = tensors[
        "tied.embedding.modality_embeddings.0.embedding.weight"
    ].clone()
    codec_config = HiggsAudioV2TokenizerConfig(
        codebook_dim=8,
        semantic_model_config={
            "model_type": "hubert",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "conv_dim": [8] * 7,
            "num_conv_pos_embedding_groups": 2,
            "num_conv_pos_embeddings": 16,
        },
        acoustic_model_config={
            "model_type": "dac",
            "encoder_hidden_size": 4,
            "decoder_hidden_size": 8,
            "hidden_size": 8,
            "codebook_dim": 2,
            "n_codebooks": 9,
            "downsampling_ratios": [2, 2],
            "upsampling_ratios": [2, 2],
        },
    )
    codec = HiggsAudioV2TokenizerModel(codec_config)
    tensors.update(
        {
            "tied.embedding.modality_embeddings.0.model." + name: value
            for name, value in codec.state_dict().items()
        }
    )
    save_file(tensors, source / "model.safetensors")
    inject_adapter(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("factor_b"):
                parameter.fill_(0.01)
    settings = Path("resources/codec_tts/inference.json")
    codec_path = tmp_path / "codec.json"
    write_json(codec_path, codec_config.to_dict())
    recipe = Recipe(base_directory=source, codec_configuration=codec_path)
    destination = tmp_path / "stage"
    with pytest.raises(FileNotFoundError, match="LICENSE"):
        stage_model(model, recipe, destination, settings, "a" * 64)
    assert not destination.exists()
    (source / "LICENSE").write_text("Upstream license fixture.\n", encoding="utf-8")
    report = stage_model(model, recipe, destination, settings, "a" * 64)
    assert (destination / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    assert report["format"] == "transformers-auto-class"
    assert (destination / "NOTICE").is_file()
    assert verify_stage(destination)["verified"] is True
    assert stage_model(model, recipe, destination, settings, "a" * 64) == report
    restored, loading = AutoModelForTextToWaveform.from_pretrained(
        destination, trust_remote_code=True, dtype=torch.float32, output_loading_info=True
    )
    assert set(loading["missing_keys"]) == set()
    assert set(loading["unexpected_keys"]) == set()
    assert (
        AutoConfig.from_pretrained(destination, trust_remote_code=True).model_type
        == "agbalu_speech"
    )
    assert restored.generator.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.generator.state_dict()[name])
    for name, value in codec.state_dict().items():
        assert torch.equal(value, restored.audio_tokenizer.state_dict()[name])
    mixed = AutoModelForTextToWaveform.from_pretrained(
        destination, trust_remote_code=True, dtype=torch.bfloat16
    )
    assert mixed.generator.audio_embedding.weight.dtype == torch.bfloat16
    assert all(value.dtype == torch.float32 for value in mixed.audio_tokenizer.parameters())
    for name, value in mixed.generator.named_parameters():
        if name.endswith(("factor_a", "factor_b")):
            assert value.dtype == torch.float32
    with pytest.raises(FileNotFoundError, match=r"missing-reference\.wav"):
        restored.generate("Azul", tmp_path / "missing-reference.wav", "Hello")
    roundtrip = tmp_path / "roundtrip"
    restored.save_pretrained(roundtrip)
    reloaded = AutoModelForTextToWaveform.from_pretrained(
        roundtrip, trust_remote_code=True, dtype=torch.float32
    )
    assert AutoTokenizer.from_pretrained(roundtrip).encode("Azul") == [1]
    for name, value in restored.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name])
    environment = dict(
        os.environ,
        PYTHONPATH="",
        PYDANTIC_DISABLE_PLUGINS="1",
        HF_MODULES_CACHE=str(tmp_path / "modules"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from transformers import AutoModelForTextToWaveform; "
            f"m=AutoModelForTextToWaveform.from_pretrained({str(destination)!r}, "
            "trust_remote_code=True); "
            "import sys; assert 'agbalu' not in sys.modules; "
            "assert callable(m.generate); print(m.config.model_type)",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "agbalu_speech" in result.stdout
    (destination / "modeling_speech.py").write_text("changed")
    with pytest.raises(ValueError, match=r"modeling_speech\.py"):
        verify_stage(destination)
