"""Configuration for the Feraoun-36M document OCR model."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig


class FeraounConfig(PreTrainedConfig):
    """Field names are the published `config.json`'s, which is what a release must load."""

    model_type = "feraoun"

    def __init__(
        self,
        vocab_size: int = 171,
        hidden_size: int = 384,
        intermediate_size: int = 1536,
        num_decoder_layers: int = 6,
        num_decoder_heads: int = 6,
        max_length: int = 256,
        dropout: float = 0.1,
        encoder_layers: int = 12,
        encoder_image_size: int = 384,
        encoder_patch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        # The width of the positional table, so it is an architecture field and not a
        # generation one. `PreTrainedConfig` no longer carries `max_length` itself.
        self.max_length = max_length
        self.dropout = dropout
        self.encoder_layers = encoder_layers
        self.encoder_image_size = encoder_image_size
        self.encoder_patch_size = encoder_patch_size
        # There is no key-value cache: every decoding step re-reads the whole prefix, as the
        # published evaluation did. Set here rather than passed to the superclass, which
        # drops unrecognised keyword arguments.
        self.use_cache = False

        # `setdefault`, because a saved config carries these and passing both spellings to
        # the superclass is a duplicate-keyword TypeError.
        kwargs.setdefault("is_encoder_decoder", True)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("decoder_start_token_id", 1)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_decoder_heads


__all__ = ["FeraounConfig"]
