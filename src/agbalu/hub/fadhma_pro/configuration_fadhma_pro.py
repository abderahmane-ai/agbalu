"""Configuration for standalone Fadhma-Pro Hugging Face model."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class FadhmaProConfig(PretrainedConfig):
    """Configuration class for Fadhma-Pro ASR model."""

    model_type = "fadhma_pro"

    def __init__(
        self,
        vocab_size: int = 40,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
        sample_rate: int = 16_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.sample_rate = sample_rate
