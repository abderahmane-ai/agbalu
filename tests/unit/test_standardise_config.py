"""Unit tests for Boulifa-48M model and training configurations."""

from __future__ import annotations

import pytest

from agbalu.standardise.config import ConfigError, ModelConfig, TrainConfig


def test_model_config_defaults() -> None:
    config = ModelConfig()
    assert config.vocab_size == 128
    assert config.hidden_size == 512
    assert config.intermediate_size == 1536
    assert config.num_attention_heads == 8
    assert config.num_encoder_layers == 6
    assert config.num_decoder_layers == 6
    assert config.head_size == 64
    assert config.parameters == 47_797_760


def test_model_config_validation() -> None:
    with pytest.raises(ConfigError, match="not divisible"):
        ModelConfig(hidden_size=513, num_attention_heads=8)

    with pytest.raises(ConfigError, match="must be at least"):
        ModelConfig(vocab_size=5)


def test_train_config_defaults() -> None:
    config = TrainConfig()
    assert config.run_name == "boulifa-48m-v1"
    assert config.micro_batch_size == 64
    assert config.learning_rate == 5e-4
