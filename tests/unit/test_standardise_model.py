"""`CharTransformer`'s shapes, and the teacher-forced path its loss is computed on."""

from __future__ import annotations

import torch

from agbalu.standardise.config import ModelConfig
from agbalu.standardise.model import CharTransformer


def test_char_transformer_parameter_count() -> None:
    config = ModelConfig()
    model = CharTransformer(config)
    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == config.parameters
    assert actual_params == 47_797_760


def test_char_transformer_forward_pass() -> None:
    config = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=192,
        num_attention_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    model = CharTransformer(config)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(4, 120, (batch_size, seq_len))
    target_ids = torch.randint(4, 120, (batch_size, seq_len))

    loss_out = model(input_ids, target_ids, pad_id=0)
    assert loss_out.loss.item() > 0.0
    assert loss_out.tokens.item() > 0
    assert loss_out.correct.item() >= 0
