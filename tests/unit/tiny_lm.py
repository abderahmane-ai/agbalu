"""A random Qwen3.5 of about 40k parameters and a character tokenizer, for CPU tests.

The architecture is the real one, Gated DeltaNet and full attention layers both, so the
trainer's checkpoint, resume and scoring paths run as they do on the HPC. CPU attention
ignores the packed boundaries, so a test that compares packed with unpacked scores keeps
one document per row.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from agbalu.llm.forward import LanguageModel

VOCABULARY: Final = 64
EOS: Final = 0


def config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=VOCABULARY,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        eos_token_id=EOS,
        pad_token_id=EOS,
        tie_word_embeddings=False,
    )


def save(directory: Path, seed: int = 0) -> Path:
    """A freshly initialised model written where `from_pretrained` can read it."""
    torch.manual_seed(seed)
    Qwen3_5ForCausalLM(config()).save_pretrained(directory)
    return directory


def load(path: str | Path, dtype: torch.dtype) -> LanguageModel:
    model: LanguageModel = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    return model


class CharTokenizer:
    """One id per character, never the separator."""

    eos = EOS
    backend = None

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [[ord(c) % (VOCABULARY - 1) + 1 for c in text] for text in texts]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(96 + i % 26) for i in ids if i != EOS)
