"""Loading a candidate encoder and giving it the characters Kabyle needs.

`widen` and `repair` split on what each needs: the coverage table is a property of a
vocabulary alone, so measuring four candidates downloads four tokenizers and no weights.

`resize_token_embeddings` appends, leaving every existing row where it was, so donor
vectors are read after the resize and no row written here carries the initialiser noise
transformers fills the new rows with.

The repair is verified rather than assumed: `widen` ends by re-running `assert_covered`,
so a character that was added but still tokenises through `<unk>` raises here rather than
surfacing later as a weak retrieval number with no obvious cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from tokenizers import AddedToken
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from agbalu.embed.vocabulary import (
    Encode,
    VocabularyError,
    assert_covered,
    donor_map,
    missing_characters,
)

WORD_BOUNDARY: str = "▁"
"""SentencePiece's word-start marker. It carries position, not identity, so it is
dropped before a donor's rows are averaged."""

CANDIDATES: Mapping[str, str] = {
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "LaBSE": "sentence-transformers/LaBSE",
    "mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "kabyle-st-mpnet": "data/raw/hf.boffire.kabyle-sentence-transformer-mpnet",
}
"""Every encoder measured before one is chosen.

`kabyle-st-mpnet` is addressed by local path, not repo id: it is the only published
Kabyle sentence transformer and its `boffire` namespace answers HTTP 401, so the copy
in `data/raw/` is the only one reachable. It is a baseline, never a base.
"""


@dataclass(frozen=True, slots=True)
class Repair:
    """What `repair` changed, for the stats file beside the trained model."""

    added: tuple[str, ...]
    donors: Mapping[str, str] = field(default_factory=dict)
    vocabulary_before: int = 0
    vocabulary_after: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added)


def encoder(tokenizer: PreTrainedTokenizerBase) -> Encode:
    """Text to ids without special tokens, which is the only form worth measuring."""

    def encode(text: str) -> Sequence[int]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return list(ids)

    return encode


def unknown_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """The `<unk>` id, refusing a tokenizer that declares none.

    Without one, every coverage number below reads as a clean zero on a tokenizer that
    may be dropping characters by another route.
    """
    unk = tokenizer.unk_token_id
    if unk is None:
        message = f"tokenizer {tokenizer.name_or_path!r} declares no unk_token_id"
        raise VocabularyError(message)
    return int(unk)


def embedding_weight(model: PreTrainedModel) -> torch.Tensor:
    """The input embedding matrix, refusing a model whose input head is not a table.

    A tied or factorised head returns something without a `weight`, and writing a donor
    row into it would silently do nothing.
    """
    weight = getattr(model.get_input_embeddings(), "weight", None)
    if not isinstance(weight, torch.Tensor):
        message = f"{type(model).__name__} input embeddings expose no weight tensor"
        raise VocabularyError(message)
    return weight


def _token_row(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    """The single row a token occupies, refusing a token that resolved to several."""
    row = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(row, int):
        message = f"token {token!r} resolved to {row!r} rather than one row"
        raise VocabularyError(message)
    return row


def _donor_vector(
    weight: torch.Tensor, encode: Encode, tokenizer: PreTrainedTokenizerBase, donor: str
) -> torch.Tensor:
    """Mean of the rows the donor characters already occupy."""
    rows: list[int] = []
    for char in donor:
        for token_id in encode(char):
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(token, str) and token.replace(WORD_BOUNDARY, "").strip():
                rows.append(int(token_id))
    if not rows:
        message = f"donor {donor!r} contributed no usable rows"
        raise VocabularyError(message)
    return weight[rows].mean(dim=0).clone()


def widen(tokenizer: PreTrainedTokenizerBase) -> Repair:
    """The tokenizer half: one token per missing required character.

    Separate from `repair` because the coverage table is a property of the vocabulary
    alone, and measuring it should not download four sets of weights.
    """
    encode = encoder(tokenizer)
    unk = unknown_id(tokenizer)
    before = len(tokenizer)
    missing = missing_characters(encode, unk)
    if not missing:
        return Repair(added=(), donors={}, vocabulary_before=before, vocabulary_after=before)

    donors = donor_map(missing, lambda char: unk not in encode(char))
    tokenizer.add_tokens([AddedToken(char, normalized=False, special=False) for char in missing])
    assert_covered(encoder(tokenizer), unk)
    return Repair(
        added=missing,
        donors=donors,
        vocabulary_before=before,
        vocabulary_after=len(tokenizer),
    )


def repair(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> Repair:
    """Widen the tokenizer, then give each new row its donor's embedding.

    `donor_map` resolves every donor before anything is added, so no donor is itself an
    added character and the rows read here are the base model's own.
    """
    widened = widen(tokenizer)
    if not widened.changed:
        return widened

    model.resize_token_embeddings(len(tokenizer))
    weight = embedding_weight(model)
    encode = encoder(tokenizer)
    with torch.no_grad():
        for char, donor in widened.donors.items():
            vector = _donor_vector(weight, encode, tokenizer, donor)
            weight[_token_row(tokenizer, char)] = vector.to(
                dtype=weight.dtype, device=weight.device
            )
    return widened


def load(name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """A candidate backbone and its tokenizer, unrepaired."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return model, tokenizer


__all__ = [
    "CANDIDATES",
    "Repair",
    "embedding_weight",
    "encoder",
    "load",
    "repair",
    "unknown_id",
    "widen",
]
