"""Free-running decoding, and loading a checkpoint or a published directory."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from agbalu.tifinagh.config import ModelConfig
from agbalu.tifinagh.infer import CheckpointError, Transliterator
from agbalu.tifinagh.model import CharTransformer
from agbalu.tifinagh.tokenizer import CharTokenizer

SMALL = ModelConfig(
    # The real vocabulary, because the tests decode real Tifinagh: a smaller embedding
    # indexes out of range on the first character outside it.
    vocab_size=128,
    hidden_size=16,
    intermediate_size=48,
    num_attention_heads=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
)


def engine() -> Transliterator:
    torch.manual_seed(0)
    return Transliterator(CharTransformer(SMALL), CharTokenizer(), torch.device("cpu"))


def test_a_batch_returns_one_string_per_input_in_the_input_order() -> None:
    """Rows that finish early are padded rather than removed, so position is preserved."""
    outputs = engine().greedy_batch(["ⴰ", "ⴰⵣⵓⵍ ⴰⵎⵇⵔⴰⵏ", "ⴱ"], max_length=8)
    assert len(outputs) == 3
    assert all(isinstance(text, str) for text in outputs)


def test_an_empty_batch_returns_an_empty_list_rather_than_raising() -> None:
    assert engine().greedy_batch([]) == []


def test_decoding_stops_at_the_length_bound() -> None:
    """An untrained model rarely emits EOS. The bound is what keeps that finite."""
    assert len(engine().greedy_batch(["ⴰⵣⵓⵍ"], max_length=4)[0]) <= 4


def test_the_empty_string_decodes_without_raising() -> None:
    assert isinstance(engine().greedy_batch([""], max_length=4)[0], str)


def test_beam_search_and_greedy_both_return_a_string() -> None:
    api = engine()
    assert isinstance(api.transliterate("ⴰⵣⵓⵍ", num_beams=1, max_length=6), str)
    assert isinstance(api.transliterate("ⴰⵣⵓⵍ", num_beams=3, max_length=6), str)


def test_one_beam_is_greedy() -> None:
    api = engine()
    assert (
        api.transliterate("ⴰⵣⵓⵍ", num_beams=1, max_length=6)
        == api.greedy_batch(["ⴰⵣⵓⵍ"], max_length=6)[0]
    )


def test_loading_a_path_that_does_not_exist_names_the_path() -> None:
    with pytest.raises(CheckpointError, match="no checkpoint"):
        Transliterator.load(Path("artifacts/checkpoints/nowhere.pt"))


def test_a_checkpoint_without_weights_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "juba.pt"
    torch.save({"config": {}}, path)
    with pytest.raises(CheckpointError, match="model_state_dict"):
        Transliterator.load(path)


def test_a_checkpoint_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """The shapes come from the checkpoint's own config, not from `ModelConfig()`."""
    model = CharTransformer(SMALL)
    path = tmp_path / "juba.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(SMALL)}, path)
    loaded = Transliterator.load(path)
    assert loaded.model.config == SMALL


def test_a_tokenizer_wider_than_the_embedding_is_refused_by_name() -> None:
    """Otherwise it surfaces as an IndexError from inside `nn.Embedding`."""
    narrow = ModelConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=48,
        num_attention_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    with pytest.raises(CheckpointError, match="embedding rows"):
        Transliterator(CharTransformer(narrow), CharTokenizer(), torch.device("cpu"))


def test_a_directory_missing_its_weights_says_which_file_is_missing(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointError, match=r"model\.safetensors"):
        Transliterator.load(tmp_path)


def test_a_published_directory_loads_and_reties_the_head(tmp_path: Path) -> None:
    """The export drops `lm_head.weight` because it is the embedding under a second name.
    A loader that did not re-tie it would publish an unloadable release."""
    model = CharTransformer(SMALL)
    tensors = {
        name: tensor.contiguous()
        for name, tensor in model.state_dict().items()
        if name != "lm_head.weight"
    }
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(asdict(SMALL)), encoding="utf-8")

    loaded = Transliterator.load(tmp_path)
    assert loaded.model.lm_head.weight is loaded.model.embed.weight
    assert torch.equal(loaded.model.embed.weight, model.embed.weight)


def test_a_published_directory_loads_past_the_transformers_fields(tmp_path: Path) -> None:
    """`tools.stage_hub` adds `auto_map`, `architectures` and `model_type` to the same
    file, and a frozen dataclass rejects every one of them by name."""
    model = CharTransformer(SMALL)
    save_file(
        {
            name: tensor.contiguous()
            for name, tensor in model.state_dict().items()
            if name != "lm_head.weight"
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **asdict(SMALL),
                "model_type": "juba",
                "architectures": ["JubaForSeq2SeqLM"],
                "auto_map": {"AutoConfig": "configuration_juba.JubaConfig"},
                "transformers_version": "5.12.1",
            }
        ),
        encoding="utf-8",
    )
    assert torch.equal(Transliterator.load(tmp_path).model.embed.weight, model.embed.weight)


def test_a_published_directory_missing_a_real_tensor_is_refused(tmp_path: Path) -> None:
    """`strict=False` covers exactly one documented omission and nothing else."""
    model = CharTransformer(SMALL)
    tensors = {
        name: tensor.contiguous()
        for name, tensor in model.state_dict().items()
        if name not in {"lm_head.weight", "enc_norm.weight"}
    }
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(asdict(SMALL)), encoding="utf-8")

    with pytest.raises(CheckpointError, match=r"enc_norm\.weight"):
        Transliterator.load(tmp_path)
