"""What a trained vocabulary is worth, in the terms this corpus made necessary.

Fertility and compression are reported, never optimised: §11.4 measured our Unigram
losing compression to the community BPE (1.754 vs 1.542 at 48k), which is what Unigram
trades away.

The three Kabyle-specific criteria:

- **state-share** — do the free and annexed forms of a noun share a stem piece? 0/15 at
  every size under default initialization (§11.4).
- **clitic-split** — does a clitic survive the hyphen as one piece? 72/72 (6 hosts × 12
  clitics) once `-` is in `required_chars`.
- **byte-piece rate** — `byte_fallback=True` means UNK never appears, so an alphabet
  failure surfaces as a run of `<0x..>` pieces rather than as a count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import sentencepiece as spm

from agbalu.tokenizer.spec import METASPACE, MIN_STEM, TokenizerError

BYTE_PIECE: Final = re.compile(r"^<0x[0-9A-F]{2}>$")

EMBEDDING_DIMS: Final[tuple[int, ...]] = (384, 512, 768)
"""Candidate encoder widths for Phase 8. At d=384 a 32k table is 36.7% of a 30M-parameter
model and an 8k table is 12.6%, which is the parameter argument for sweeping small."""

STATE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("axxam", "wexxam"),
    ("argaz", "wergaz"),
    ("aman", "waman"),
    ("ass", "wass"),
    ("ayen", "wayen"),
    ("ul", "wul"),
    ("tamurt", "tmurt"),
    ("tamdint", "tmdint"),
    ("taddart", "teddart"),
    ("itri", "yitri"),
    ("ixef", "yixef"),
    ("iles", "yiles"),
    ("arrac", "warrac"),
    ("ayeḍ", "wayeḍ"),
    ("amdan", "wemdan"),
)
"""Free/annexed pairs with both members attested in the corpus. The alternation is
word-initial, which is the hardest case for a subword vocabulary to factor."""

CLITIC_HOSTS: Final[tuple[str, ...]] = ("ɣur", "fell", "yefka", "yenna", "ger", "deg")
CLITICS: Final[tuple[str, ...]] = (
    "s",
    "ak",
    "am",
    "as",
    "aɣ",
    "wen",
    "kent",
    "sen",
    "sent",
    "iw",
    "ik",
    "im",
)


@dataclass(frozen=True, slots=True)
class Evaluation:
    name: str
    vocab_size: int
    sentences: int
    words: int
    characters: int
    tokens: int
    fertility: float
    tokens_per_char: float
    byte_pieces: int
    byte_piece_rate: float
    roundtrip_failures: int
    whole_word_share: float
    state_share: int
    state_trials: int
    clitic_atomic: int
    clitic_trials: int
    embedding_params: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "sentences": self.sentences,
            "words": self.words,
            "characters": self.characters,
            "tokens": self.tokens,
            "fertility": round(self.fertility, 4),
            "tokens_per_char": round(self.tokens_per_char, 4),
            "byte_pieces": self.byte_pieces,
            "byte_piece_rate": round(self.byte_piece_rate, 6),
            "roundtrip_failures": self.roundtrip_failures,
            "whole_word_share": round(self.whole_word_share, 4),
            "state_share": f"{self.state_share}/{self.state_trials}",
            "clitic_atomic": f"{self.clitic_atomic}/{self.clitic_trials}",
            "embedding_params": {str(d): n for d, n in self.embedding_params.items()},
        }


def load(model: Path) -> spm.SentencePieceProcessor:
    if not model.is_file():
        msg = f"tokenizer model not found: {model}"
        raise TokenizerError(msg)
    processor = spm.SentencePieceProcessor()
    processor.load(str(model))
    return processor


def state_share(processor: spm.SentencePieceProcessor) -> int:
    """Pairs whose two states share a stem piece of at least `MIN_STEM` characters."""
    shared = 0
    for free, annexed in STATE_PAIRS:
        left = {p.lstrip(METASPACE) for p in processor.encode(free, out_type=str)}
        right = {p.lstrip(METASPACE) for p in processor.encode(annexed, out_type=str)}
        if {p for p in left if len(p) >= MIN_STEM} & {p for p in right if len(p) >= MIN_STEM}:
            shared += 1
    return shared


def clitic_atomic(processor: spm.SentencePieceProcessor) -> int:
    return sum(
        1
        for host in CLITIC_HOSTS
        for clitic in CLITICS
        if any(
            p.lstrip(METASPACE) == clitic
            for p in processor.encode(f"{host}-{clitic}", out_type=str)
        )
    )


def evaluate(model: Path, sentences: list[str], name: str | None = None) -> Evaluation:
    if not sentences:
        msg = "refusing to evaluate on an empty sample"
        raise TokenizerError(msg)
    processor = load(model)
    vocab = processor.get_piece_size()
    pieces = [processor.id_to_piece(i) for i in range(vocab)]

    tokens = 0
    byte_pieces = 0
    failures = 0
    for text in sentences:
        encoded = processor.encode(text, out_type=str)
        tokens += len(encoded)
        byte_pieces += sum(1 for p in encoded if BYTE_PIECE.match(p))
        if processor.decode(encoded) != text:
            failures += 1

    words = sum(len(text.split()) for text in sentences)
    characters = sum(len(text) for text in sentences)
    whole = sum(1 for p in pieces if p.startswith(METASPACE))

    return Evaluation(
        name=name or model.stem,
        vocab_size=vocab,
        sentences=len(sentences),
        words=words,
        characters=characters,
        tokens=tokens,
        fertility=tokens / words if words else 0.0,
        tokens_per_char=tokens / characters if characters else 0.0,
        byte_pieces=byte_pieces,
        byte_piece_rate=byte_pieces / tokens if tokens else 0.0,
        roundtrip_failures=failures,
        whole_word_share=whole / vocab,
        state_share=state_share(processor),
        state_trials=len(STATE_PAIRS),
        clitic_atomic=clitic_atomic(processor),
        clitic_trials=len(CLITIC_HOSTS) * len(CLITICS),
        embedding_params={d: vocab * d for d in EMBEDDING_DIMS},
    )
