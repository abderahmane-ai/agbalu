"""Configuration for the Feraoun-Pro document OCR model."""

from __future__ import annotations

from typing import Any, Final

from transformers import PreTrainedConfig

SYMBOLS: Final[str] = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ɣƔɛƐčČǧǦţŢ"
    "ḍḌḥḤṛṚṣṢṭṬẓẒ"
    "éèêëàâîïôùûçÉÈÊËÀÂÎÏÔÙÛÇ"
    "0123456789"
    " .,?!«»-—:;'\"/()[]{}%*+=_~–…\n\t"
    "ⴰⴱⴳⴷⴹⴻⴼⴽⵀⵃⵄⵅⵇⵉⵊⵍⵎⵏⵓⵔⵕⵙⵚⵛⵜⵟⵡⵢⵣⵥⵖ"
)
"""The 169 printable classes after the four special tokens, spelled out rather than
imported: nothing under `hub/` may import `agbalu`, because these modules are loaded from
the published repository where this package does not exist.
`tests/unit/test_ocr_pro.py` pins them against `agbalu.ocr.vocabulary_pro`, so the two
copies cannot drift apart silently."""

DEFAULT_VOCABULARY: Final[tuple[str, ...]] = ("<pad>", "<s>", "</s>", "<unk>", *SYMBOLS)


class FeraounProConfig(PreTrainedConfig):
    model_type = "feraoun_pro"

    def __init__(
        self,
        vocabulary: list[str] | tuple[str, ...] | None = None,
        vocab_size: int = 173,
        d_model: int = 512,
        num_encoder_layers: int = 16,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        max_width_frames: int = 256,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # The symbol table ships in `config.json`: a CTC id means nothing without it, and a
        # downloader who has to reconstruct it from this repository does not have a usable
        # artifact. `vocab_size` stays a separate field because it is the width of
        # `ctc_head` and a saved config carries it.
        self.vocabulary = list(vocabulary) if vocabulary is not None else list(DEFAULT_VOCABULARY)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.max_width_frames = max_width_frames
        self.dropout = dropout

        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("unk_token_id", 3)
        super().__init__(**kwargs)


__all__ = ["DEFAULT_VOCABULARY", "SYMBOLS", "FeraounProConfig"]
