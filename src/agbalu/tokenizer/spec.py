"""Every parameter that decides the AƔBALU-Tok vocabulary.

Settled in `docs/tokenizer_design.md` §11.6 against the corpus itself, which overturned
two decisions of the literature survey in §10: the hyphen segments rather than being
protected, and vocabulary pressure never factors the annexed state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

from agbalu.normalise.rules import ALPHABET, HYPHEN, load_rules

TOKENIZER_VERSION: Final = "1.0.0"
"""Bump on any change to the produced vocabulary. Stamped into every model's metadata
beside the normaliser version, because a tokenizer trained on differently-normalised
text is a different tokenizer."""

MODEL_TYPE: Final = "unigram"
NORMALIZATION_RULE: Final = "identity"
"""The corpus is already normaliser output. SentencePiece's own NMT normalisation would
re-fold characters Phase 2 deliberately preserves, `ţ` above all."""

APOSTROPHE: Final = "'"

PAD_PIECE: Final = "[PAD]"
UNK_PIECE: Final = "[UNK]"
CLS_PIECE: Final = "[CLS]"
SEP_PIECE: Final = "[SEP]"
MASK_PIECE: Final = "[MASK]"

PAD_ID: Final = 0
UNK_ID: Final = 1
CLS_ID: Final = 2
SEP_ID: Final = 3

METASPACE: Final = "▁"
MAX_PIECE_LENGTH: Final = 16

MIN_STEM: Final = 3
"""A piece shorter than this shared between two forms is a letter cluster, not evidence
of a stem. Used both to seed stems and to score whether the annexed state was factored."""

SWEEP_SIZES: Final[tuple[int, ...]] = (8_000, 12_000, 16_000, 24_000, 32_000)
"""§11.6. The upper bound is not a budget: at d=384 a 32k table is 36.7% of a 30M-
parameter encoder, so vocabulary size is the largest single parameter lever we have."""

DEFAULT_VOCAB_SIZE: Final = 16_000
"""One of the swept points, so the default is always a measured configuration. The
BabyLM-winning LTG-BERT line uses 16,384 at our exact data scale (`ltgoslo/elc-bert`
and `ltgoslo/gpt-bert`, `configs/base.json`)."""

MIN_VOCAB_SIZE: Final = 1_000
MAX_VOCAB_SIZE: Final = 64_000

RANDOM_SEED: Final = 20260807


class TokenizerError(Exception):
    """A tokenizer could not be specified, trained, or loaded."""


@cache
def required_chars() -> str:
    """Characters guaranteed a vocabulary slot of their own.

    `rules.ALPHABET` plus the letters the rule table preserves rather than canonicalises
    — `ţ` U+0163, 21,058 attestations of the Dallet tradition that CLDR `kab.xml` omits.
    The hyphen is here because §11.2 wants it to *segment*: atomic and cheap, it lets
    Unigram break `yefka-yas-t-id` instead of memorising it as a hapax.

    Deliberately wider than `tools/vocabulary_evidence.py`'s classification set, which
    answers a different question. Being generous here costs one slot per character;
    being generous there would inflate the measured Kabyle share of the corpus.
    """
    preserved = {c for c in load_rules().preserved_chars if c.isalpha()}
    letters = set(ALPHABET) | preserved | {c.upper() for c in preserved}
    return "".join(sorted(letters | {HYPHEN, APOSTROPHE}))


TrainerArg = str | int | float | bool


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """A single trainable configuration. Two specs differing only in `seed_file` are the
    §11.5 experiment: does lexicon-seeded initialization represent the annexed state
    where the default initialization does not?"""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    seed_file: Path | None = None
    character_coverage: float = 0.9995
    input_sentence_size: int = 2_000_000
    num_threads: int = 8
    random_seed: int = RANDOM_SEED
    """Applied through `sentencepiece.set_random_generator_seed`, not through the trainer
    spec: `random_seed` is a process-global flag in SentencePiece, not a `TrainerSpec`
    field, and passing it as one is rejected outright. It decides which sentences
    `shuffle_input_sentence` draws, so without it a rebuild is not the same build."""

    def __post_init__(self) -> None:
        if not MIN_VOCAB_SIZE <= self.vocab_size <= MAX_VOCAB_SIZE:
            msg = f"vocab_size {self.vocab_size} outside [{MIN_VOCAB_SIZE}, {MAX_VOCAB_SIZE}]"
            raise TokenizerError(msg)
        if not 0.0 < self.character_coverage <= 1.0:
            msg = f"character_coverage {self.character_coverage} outside (0, 1]"
            raise TokenizerError(msg)
        if self.num_threads < 1:
            msg = f"num_threads {self.num_threads} must be positive"
            raise TokenizerError(msg)

    @property
    def seeded(self) -> bool:
        return self.seed_file is not None

    @property
    def name(self) -> str:
        thousands = self.vocab_size / 1_000
        size = f"{thousands:g}k"
        return f"agbalu-tok-{'seeded' if self.seeded else 'base'}-{size}"

    def trainer_kwargs(self, corpus: Path, prefix: Path) -> dict[str, TrainerArg | list[str]]:
        kwargs: dict[str, TrainerArg | list[str]] = {
            "input": str(corpus),
            "model_prefix": str(prefix),
            "vocab_size": self.vocab_size,
            "model_type": MODEL_TYPE,
            "character_coverage": self.character_coverage,
            "required_chars": required_chars(),
            "max_sentencepiece_length": MAX_PIECE_LENGTH,
            "byte_fallback": True,
            "split_digits": True,
            "normalization_rule_name": NORMALIZATION_RULE,
            "input_sentence_size": self.input_sentence_size,
            "shuffle_input_sentence": True,
            "num_threads": self.num_threads,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "bos_id": CLS_ID,
            "eos_id": SEP_ID,
            "pad_piece": PAD_PIECE,
            "unk_piece": UNK_PIECE,
            "bos_piece": CLS_PIECE,
            "eos_piece": SEP_PIECE,
            "user_defined_symbols": [MASK_PIECE],
        }
        if self.seed_file is not None:
            kwargs["seed_sentencepieces_file"] = str(self.seed_file)
        return kwargs
