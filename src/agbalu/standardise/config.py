"""Architecture and training configuration for the orthography standardiser.

Published as `agbalu/Boulifa-48M`. The release name is not a code identifier, so
the classes here are named for what they configure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

MIN_VOCAB_SIZE: Final = 10
"""Four specials plus a usable alphabet. Below this the model cannot spell anything."""


class ConfigError(Exception):
    """A configuration that cannot build a model."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shapes of the character-level encoder-decoder for orthography standardisation.

    The defaults configure the 47.8M release (`Boulifa-48M`).
    """

    vocab_size: int = 128
    hidden_size: int = 512
    intermediate_size: int = 1_536
    num_attention_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_position_embeddings: int = 512
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    label_smoothing: float = 0.05
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            message = (
                f"hidden_size {self.hidden_size} is not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
            raise ConfigError(message)
        if self.vocab_size < MIN_VOCAB_SIZE:
            message = f"vocab_size must be at least {MIN_VOCAB_SIZE}, got {self.vocab_size}"
            raise ConfigError(message)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def parameters(self) -> int:
        """Parameter count from the shapes alone, without building the model.

        Asserted against a built model in the suite, because a count derived by hand is
        a claim about the module tree and drifts the moment a projection is added.
        """
        embedding = self.vocab_size * self.hidden_size
        conv_stem = (
            self.hidden_size * 3
            + self.hidden_size * 5
            + 2 * self.hidden_size * self.hidden_size
            + 3 * self.hidden_size
        )
        encoder_layer = (
            4 * self.hidden_size * self.hidden_size
            + 3 * self.hidden_size * self.intermediate_size
            + 2 * self.hidden_size
        )
        decoder_layer = (
            8 * self.hidden_size * self.hidden_size
            + 3 * self.hidden_size * self.intermediate_size
            + 3 * self.hidden_size
        )
        final_norms = 2 * self.hidden_size
        return (
            embedding
            + conv_stem
            + self.num_encoder_layers * encoder_layer
            + self.num_decoder_layers * decoder_layer
            + final_norms
        )


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The recipe the release was trained under."""

    run_name: str = "boulifa-48m-v1"
    micro_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    max_steps: int = 25_000
    eval_every: int = 500
    save_every: int = 1_000
    max_grad_norm: float = 1.0
    seed: int = 42
