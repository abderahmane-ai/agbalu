"""Configuration for the Matoub-82M Kabyle text-to-speech model."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig


class MatoubConfig(PreTrainedConfig):
    """Field names are the published `config.json`'s, which is what a release must load."""

    model_type = "matoub"

    def __init__(
        self,
        vocab_size: int = 178,
        hidden_size: int = 512,
        style_dim: int = 128,
        num_layers: int = 3,
        max_duration: int = 50,
        dropout: float = 0.2,
        text_encoder_kernel_size: int = 5,
        num_mels: int = 80,
        sampling_rate: int = 24_000,
        plbert_hidden_size: int = 768,
        plbert_num_attention_heads: int = 12,
        plbert_intermediate_size: int = 2048,
        plbert_num_hidden_layers: int = 12,
        plbert_max_position_embeddings: int = 512,
        upsample_rates: tuple[int, ...] = (10, 6),
        upsample_kernel_sizes: tuple[int, ...] = (20, 12),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        gen_istft_n_fft: int = 20,
        gen_istft_hop_size: int = 5,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.max_duration = max_duration
        self.dropout = dropout
        self.text_encoder_kernel_size = text_encoder_kernel_size
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.plbert_hidden_size = plbert_hidden_size
        self.plbert_num_attention_heads = plbert_num_attention_heads
        self.plbert_intermediate_size = plbert_intermediate_size
        self.plbert_num_hidden_layers = plbert_num_hidden_layers
        self.plbert_max_position_embeddings = plbert_max_position_embeddings
        self.upsample_rates = tuple(upsample_rates)
        self.upsample_kernel_sizes = tuple(upsample_kernel_sizes)
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = tuple(resblock_kernel_sizes)
        self.resblock_dilation_sizes = tuple(tuple(d) for d in resblock_dilation_sizes)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)

    @property
    def max_token_length(self) -> int:
        """Longest token sequence PL-BERT can position, less the two special slots."""
        return self.plbert_max_position_embeddings - 2


__all__ = ["MatoubConfig"]
