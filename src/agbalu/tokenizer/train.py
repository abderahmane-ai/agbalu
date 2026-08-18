"""Train one AƔBALU-Tok model and record what produced it.

Every model carries the normaliser version in its metadata. A vocabulary built from
1.2.0 text and one built from 1.3.0 text are different vocabularies, and an artifact
that cannot say which it is cannot be compared with anything (CLAUDE.md §6.3).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import sentencepiece as spm

from agbalu.normalise import Normaliser
from agbalu.tokenizer.spec import TOKENIZER_VERSION, TokenizerError, TokenizerSpec

log: Final = logging.getLogger("agbalu.tokenizer")

DEFAULT_OUT_DIR: Final = Path("artifacts/tokenizer")

_READ_BLOCK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class BuildResult:
    spec: TokenizerSpec
    model: Path
    metadata: Path
    pieces: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def train(spec: TokenizerSpec, corpus: Path, out_dir: Path = DEFAULT_OUT_DIR) -> BuildResult:
    """Train `spec` over the flat-text `corpus`, writing model, vocab and metadata.

    `corpus` is the plain one-sentence-per-line file from `corpus.write_plain`, not the
    JSONL: SentencePiece reads no other format.
    """
    if not corpus.is_file():
        msg = f"training corpus not found: {corpus}"
        raise TokenizerError(msg)
    if spec.seed_file is not None and not spec.seed_file.is_file():
        msg = f"seed file not found: {spec.seed_file}; run `tokenizer seed` first"
        raise TokenizerError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / spec.name
    model = prefix.with_suffix(".model")

    log.info("training %s vocab=%d seeded=%s", spec.name, spec.vocab_size, spec.seeded)
    spm.set_random_generator_seed(spec.random_seed)
    spm.SentencePieceTrainer.train(**spec.trainer_kwargs(corpus, prefix))
    if not model.is_file():
        msg = f"trainer produced no model at {model}"
        raise TokenizerError(msg)

    processor = spm.SentencePieceProcessor()
    processor.load(str(model))
    pieces = int(processor.get_piece_size())

    metadata = prefix.with_suffix(".metadata.json")
    payload: dict[str, object] = {
        "name": spec.name,
        "tokenizer_version": TOKENIZER_VERSION,
        "normaliser_version": Normaliser().version,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "pieces": pieces,
        "model_sha256": sha256(model),
        "spec": {
            "vocab_size": spec.vocab_size,
            "seeded": spec.seeded,
            "seed_file": str(spec.seed_file) if spec.seed_file else None,
            "character_coverage": spec.character_coverage,
            "input_sentence_size": spec.input_sentence_size,
            "random_seed": spec.random_seed,
        },
    }
    metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s (%d pieces)", model, pieces)
    return BuildResult(spec=spec, model=model, metadata=metadata, pieces=pieces)
