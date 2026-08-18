"""Fill-mask inference against a trained encoder checkpoint.

The masked position is found by encoding the text and looking for the tokenizer's own
`[MASK]` piece id rather than a literal 4: the piece is a `user_defined_symbol`, so its id
is a property of the vocabulary and a rebuild could move it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import torch

from agbalu.model.checkpoint import BEST
from agbalu.model.checkpoint import load as load_checkpoint
from agbalu.model.config import PRESETS, ModelError, Preset
from agbalu.model.modeling import Encoder
from agbalu.tokenizer.spec import CLS_ID, MASK_PIECE, SEP_ID

TOP_K: Final = 5


class Tokenizer(Protocol):
    """The slice of the SentencePiece API this module drives.

    Structural rather than `spm.SentencePieceProcessor`: the package ships no stubs, so
    naming the concrete type puts `Any` in every signature here.
    """

    def piece_to_id(self, piece: str) -> int: ...

    def unk_id(self) -> int: ...

    def encode(self, text: str, out_type: type[int]) -> list[int]: ...

    def id_to_piece(self, token_id: int) -> str: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    piece: str
    probability: float


@dataclass(frozen=True, slots=True)
class MaskPrediction:
    """Predictions for one masked slot, ranked. `index` is the position in the encoding."""

    index: int
    candidates: tuple[Candidate, ...]


def mask_id(tokenizer: Tokenizer) -> int:
    """The vocabulary's own id for `[MASK]`, verified to be a real piece."""
    token_id = int(tokenizer.piece_to_id(MASK_PIECE))
    if token_id == tokenizer.unk_id():
        msg = f"this tokenizer has no {MASK_PIECE} piece; it cannot be used for fill-mask"
        raise ModelError(msg)
    return token_id


def encode(text: str, tokenizer: Tokenizer) -> tuple[list[int], list[int]]:
    """`(input_ids, masked_positions)` with `[CLS]` and `[SEP]` attached.

    Positions are indices into `input_ids`, so they already account for the leading `[CLS]`.
    """
    pieces: list[int] = tokenizer.encode(text, out_type=int)
    input_ids = [CLS_ID, *pieces, SEP_ID]
    target = mask_id(tokenizer)
    positions = [i for i, token in enumerate(input_ids) if token == target]
    if not positions:
        msg = f"no {MASK_PIECE} in {text!r}; write the mask exactly as {MASK_PIECE}"
        raise ModelError(msg)
    return input_ids, positions


def load_encoder(
    directory: Path,
    preset: Preset = "kab",
    *,
    name: str = BEST,
    device: torch.device | None = None,
) -> Encoder:
    """Build the encoder and restore weights through the checksum-verifying loader.

    Neither the optimizer nor the RNG is restored: inference needs neither, and replaying a
    training RNG state into the local process would perturb anything else running.
    """
    target = device or torch.device("cpu")
    model = Encoder(PRESETS[preset]).to(target)
    load_checkpoint(directory, model, None, name=name, restore_rng=False)
    return model.eval()


@torch.inference_mode()
def fill_mask(
    text: str,
    model: Encoder,
    tokenizer: Tokenizer,
    *,
    device: torch.device | None = None,
    top_k: int = TOP_K,
) -> list[MaskPrediction]:
    """Ranked candidates for every `[MASK]` in `text`, in the order they appear."""
    if top_k < 1:
        msg = f"top_k must be positive, got {top_k}"
        raise ModelError(msg)

    target = device or next(model.parameters()).device
    input_ids, positions = encode(text, tokenizer)
    ids = torch.tensor([input_ids], dtype=torch.long, device=target)
    attention = torch.ones_like(ids, dtype=torch.bool)

    hidden = model.contextualise(ids, attention)
    logits = model.classifier(hidden[0, torch.tensor(positions, device=target)])
    probabilities = torch.softmax(logits.float(), dim=-1)
    ranked = torch.topk(probabilities, k=min(top_k, probabilities.shape[-1]), dim=-1)

    return [
        MaskPrediction(
            index=position,
            candidates=tuple(
                Candidate(piece=str(tokenizer.id_to_piece(int(piece_id))), probability=float(score))
                for score, piece_id in zip(scores, piece_ids, strict=True)
            ),
        )
        for position, scores, piece_ids in zip(
            positions, ranked.values, ranked.indices, strict=True
        )
    ]
