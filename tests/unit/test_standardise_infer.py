"""Unit tests for Standardiser inference engine."""

from __future__ import annotations

import torch

from agbalu.standardise.config import ModelConfig
from agbalu.standardise.infer import Standardiser
from agbalu.standardise.model import CharTransformer
from agbalu.standardise.tokenizer import Tokenizer


def test_standardiser_mock_inference() -> None:
    config = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=192,
        num_attention_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    tokenizer = Tokenizer.build()
    model = CharTransformer(config)
    standardiser = Standardiser(model, tokenizer, device=torch.device("cpu"))

    input_text = "achimi ur d-thekhedmedh ara?"
    output_text = standardiser.standardise(input_text, max_length=64)
    assert isinstance(output_text, str)


def test_standardiser_batch() -> None:
    config = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=192,
        num_attention_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    tokenizer = Tokenizer.build()
    model = CharTransformer(config)
    standardiser = Standardiser(model, tokenizer, device=torch.device("cpu"))

    inputs = ["achimi?", "khedmegh g taddarth.", "tamazight l3ali."]
    outputs = standardiser.standardise_batch(inputs, batch_size=2)
    assert len(outputs) == 3
    assert all(isinstance(o, str) for o in outputs)
