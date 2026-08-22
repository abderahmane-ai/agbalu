"""Fadhma-Pro: 40-token CTC Vocabulary and OmniASR-v2 Supervised Head Slicing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

from agbalu.normalise.rules import ALPHABET, HYPHEN

WORD_DELIMITER: Final = "|"
PAD_TOKEN: Final = "[PAD]"  # noqa: S105
UNK_TOKEN: Final = "[UNK]"  # noqa: S105

SPEAKABLE: Final = frozenset({ch for ch in ALPHABET if ch.islower()} | {HYPHEN, " "})

# Sliced 40 CTC classes in canonical order
CANONICAL_LETTERS: Final = tuple(sorted([ch for ch in ALPHABET if ch.islower()]))


@dataclass(frozen=True, slots=True)
class VocabularyPro:
    """Fadhma-Pro 40-class CTC Vocabulary."""

    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN
    word_delimiter: str = WORD_DELIMITER
    letters: tuple[str, ...] = CANONICAL_LETTERS

    def get_tokens(self) -> list[str]:
        """Return the contiguous 40 tokens."""
        tokens = [self.pad_token, self.unk_token, self.word_delimiter, HYPHEN]
        tokens.extend([ch for ch in self.letters if ch != HYPHEN])
        return tokens

    def token_to_id(self) -> dict[str, int]:
        """Token string to index ID."""
        return {t: i for i, t in enumerate(self.get_tokens())}

    def id_to_token(self) -> dict[int, str]:
        """Index ID to token string."""
        return dict(enumerate(self.get_tokens()))

    def save_vocab(self, out_path: Path) -> None:
        """Save vocabulary mapping to JSON file."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id(), f, indent=2, ensure_ascii=False)


def slice_omniasr_v2_head(
    base_model: Wav2Vec2ForCTC,
    base_tokenizer: Wav2Vec2CTCTokenizer,
    target_vocab: VocabularyPro,
) -> nn.Linear:
    """Slice the 40 Kabyle classes directly from Meta's 10,288-token omniASR-v2 head.

    Copies the pre-trained weights for the 36 Kabyle letters, hyphen, and space from Meta's
    supervised head, initializing [PAD] and [UNK] safely.
    """
    base_head = base_model.lm_head
    in_features = base_head.in_features
    out_features = len(target_vocab.get_tokens())

    new_head = nn.Linear(in_features, out_features, bias=base_head.bias is not None)

    # Initialize new head with base head's mean/std
    with torch.no_grad():
        new_head.weight.copy_(torch.randn_like(new_head.weight) * 0.02)
        if new_head.bias is not None:
            new_head.bias.zero_()

        base_vocab = base_tokenizer.get_vocab()
        target_mapping = target_vocab.token_to_id()

        copied = 0
        for token, new_id in target_mapping.items():
            lookup_token = " " if token == WORD_DELIMITER else token
            if lookup_token in base_vocab:
                old_id = base_vocab[lookup_token]
                new_head.weight[new_id].copy_(base_head.weight[old_id])
                if new_head.bias is not None and base_head.bias is not None:
                    new_head.bias[new_id].copy_(base_head.bias[old_id])
                copied += 1

    return new_head
