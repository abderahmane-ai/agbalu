"""Configuration for the Masinissa-73M encoder."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig


class MasinissaConfig(PreTrainedConfig):
    """Configuration for Masinissa-73M: a modern deep bidirectional encoder with
    Block Attention Residuals, SwiGLU, and QK RMSNorm."""

    model_type = "masinissa"

    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 512,
        intermediate_size: int = 1408,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 20,
        block_layers: int = 4,
        position_buckets: int = 32,
        max_position_embeddings: int = 1024,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.block_layers = block_layers
        self.position_buckets = position_buckets
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("bos_token_id", bos_token_id)
        kwargs.setdefault("eos_token_id", eos_token_id)
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


__all__ = ["MasinissaConfig"]
