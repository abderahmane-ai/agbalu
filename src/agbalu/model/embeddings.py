"""Geometry of the encoder's word-embedding table.

Nearest neighbours and the `a - b + c` analogy, over the input embedding rather than a
contextual vector: the table is what the vocabulary itself learned, and it is comparable
across checkpoints without re-encoding a corpus.

A word the vocabulary splits has no row of its own. The first piece is used, and
`piece_for` reports which piece that was, because `agellid` and `▁agellid` are different
rows and a silent fallback would answer about the wrong one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import torch
from torch import Tensor
from torch.nn import functional

TOP_K: Final = 10
ANALOGY_TERMS: Final = 3


class Tokenizer(Protocol):
    """The slice of the SentencePiece API this module drives."""

    def encode(self, text: str, out_type: type[int]) -> list[int]: ...

    def unk_id(self) -> int: ...

    def id_to_piece(self, piece_id: int) -> str: ...


@dataclass(frozen=True, slots=True)
class Neighbour:
    piece: str
    similarity: float


def piece_for(word: str, tokenizer: Tokenizer) -> int:
    """The row this word is read from: its first piece, or `unk` if it produces none."""
    pieces = tokenizer.encode(word, out_type=int)
    return pieces[0] if pieces else tokenizer.unk_id()


def neighbours(
    target: Tensor,
    table: Tensor,
    tokenizer: Tokenizer,
    *,
    top_k: int = TOP_K,
    exclude: Iterable[int] = (),
) -> list[Neighbour]:
    """The `top_k` rows closest to `target` by cosine similarity, excluded ids removed.

    `top_k + len(excluded)` rows are taken before filtering, so the result is never short
    of `top_k` because an excluded id ranked highly — which is the normal case for an
    analogy, where all three operands rank near their own sum.
    """
    dropped = set(exclude)
    unit_table = functional.normalize(table, dim=-1)
    unit_target = functional.normalize(target, dim=-1)
    scores = torch.mv(unit_table, unit_target)
    wanted = min(top_k + len(dropped), scores.shape[0])
    ranked = torch.topk(scores, k=wanted)

    found: list[Neighbour] = []
    for score, index in zip(ranked.values.tolist(), ranked.indices.tolist(), strict=True):
        if index in dropped:
            continue
        found.append(Neighbour(tokenizer.id_to_piece(index), float(score)))
        if len(found) >= top_k:
            break
    return found


def analogy(
    words: Sequence[str],
    table: Tensor,
    tokenizer: Tokenizer,
    *,
    top_k: int = TOP_K,
) -> tuple[list[Neighbour], list[int]]:
    """`a - b + c`, and the three rows it was computed from.

    The operands are excluded from the result: each sits close to their own combination,
    and returning them would report the inputs back as the answer.
    """
    if len(words) != ANALOGY_TERMS:
        message = f"an analogy takes exactly {ANALOGY_TERMS} words, got {len(words)}"
        raise ValueError(message)
    ids = [piece_for(word, tokenizer) for word in words]
    a, b, c = (table[index] for index in ids)
    return neighbours(a - b + c, table, tokenizer, top_k=top_k, exclude=ids), ids
