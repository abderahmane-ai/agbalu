"""Configuration for the Juba-27M character encoder-decoder."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig


class JubaConfig(PreTrainedConfig):
    """Field names are the published `config.json`'s, which is what a release must load."""

    model_type = "juba"

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 384,
        intermediate_size: int = 1152,
        num_attention_heads: int = 6,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-6,
        dropout_prob: float = 0.1,
        label_smoothing: float = 0.05,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob
        self.label_smoothing = label_smoothing
        self.unk_token_id = unk_token_id
        # There is no key-value cache: every decoding step re-reads the whole prefix, as
        # the published evaluation did. Set here rather than passed to the superclass,
        # which drops unrecognised keyword arguments — without it `generate` allocates a
        # `DynamicCache` from `num_hidden_layers`, which this model does not carry.
        self.use_cache = False

        # `setdefault`, because a saved config carries these and passing both spellings to
        # the superclass is a duplicate-keyword TypeError.
        kwargs.setdefault("is_encoder_decoder", True)
        kwargs.setdefault("decoder_start_token_id", bos_token_id)
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("bos_token_id", bos_token_id)
        kwargs.setdefault("eos_token_id", eos_token_id)
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


__all__ = ["JubaConfig"]
