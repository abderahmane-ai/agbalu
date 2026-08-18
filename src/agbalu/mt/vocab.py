"""Trim NLLB's 256,206-token vocabulary to the tokens Kabyle, English and French use.

Measured over the fine-tuning corpus, which is what `modal_app.mt.trim` scans: 52,209
tokens, 20.4% of the vocabulary, taking the 1.3B to 1,161,745,408 parameters. The
embedding matrix is 42.7% of the 600M checkpoint and 19% of the 1.3B, so dropping the
unreachable rows costs nothing the three languages can produce *in that corpus* — see
`VocabTables` for what happens to text the scan never saw.

Rows are dropped, never renumbered in place: the kept ids are gathered into a new matrix
in ascending order, and `keep` is the mapping from new id back to old.

**The tokenizer is left alone.** Rebuilding a SentencePiece vocabulary to match would risk
changing segmentation; instead `Remap` translates between the tokenizer's ids and the
trimmed model's. Every path that meets both must go through it — `finetune.Dataset` on the
training side, `VocabTables` on the generation side, where the forced language token and
the generated ids are in different id spaces from the tokenizer that produced them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    import torch

LANGUAGE_CODES: Final[tuple[str, ...]] = ("kab_Latn", "eng_Latn", "fra_Latn")
"""Kept whatever the corpus uses them: they are how a direction is requested."""


class TrimError(Exception):
    """The trim would change what the model can produce."""


class Tokenizer(Protocol):
    """The slice of the tokenizer API this module drives.

    Declared structurally, as in `agbalu.bench.translate`: `transformers` types these
    loosely, and a Protocol both pins what is actually used and lets the tests exercise
    the logic without a 17 MB tokenizer on disk.
    """

    @property
    def all_special_ids(self) -> Sequence[int]: ...

    @property
    def unk_token_id(self) -> int | None: ...

    def __len__(self) -> int: ...

    def convert_tokens_to_ids(self, token: str) -> int | list[int]: ...

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool
    ) -> Mapping[str, list[list[int]]]: ...


class Embeddings(Protocol):
    weight: torch.Tensor
    embedding_dim: int


class Trimmable(Protocol):
    """What `trim_model` needs of a sequence-to-sequence checkpoint."""

    config: TrimmableConfig

    def get_input_embeddings(self) -> Embeddings: ...

    def resize_token_embeddings(self, new_num_tokens: int) -> object: ...

    def tie_weights(self) -> None: ...

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...


class TrimmableConfig(Protocol):
    vocab_size: int


@dataclass(frozen=True, slots=True)
class Coverage:
    keep: tuple[int, ...]
    """Old ids to retain, ascending. Index in this tuple is the new id."""

    scanned: int
    original_size: int

    @property
    def kept(self) -> int:
        return len(self.keep)

    @property
    def share(self) -> float:
        return self.kept / self.original_size if self.original_size else 0.0


def required_ids(tokenizer: Tokenizer) -> set[int]:
    """Ids that must survive whatever the corpus says: specials and the language codes.

    A language code that fell out would not raise — `convert_tokens_to_ids` answers
    `unk_token_id` for an unknown token, and the model would translate fluently into the
    wrong language.
    """
    ids: set[int] = {i for i in tokenizer.all_special_ids if isinstance(i, int)}
    for code in LANGUAGE_CODES:
        token_id = tokenizer.convert_tokens_to_ids(code)
        if not isinstance(token_id, int) or token_id == tokenizer.unk_token_id:
            message = f"{code!r} is not a token of this tokenizer; it cannot be trimmed safely"
            raise TrimError(message)
        ids.add(token_id)
    return ids


def coverage(texts: Iterable[str], tokenizer: Tokenizer, *, batch_size: int = 1_000) -> Coverage:
    """Which vocabulary rows the given text can reach."""
    used = required_ids(tokenizer)
    scanned = 0
    batch: list[str] = []

    def flush(items: list[str]) -> None:
        if not items:
            return
        for encoded in tokenizer(items, add_special_tokens=False)["input_ids"]:
            used.update(encoded)

    for text in texts:
        batch.append(text)
        scanned += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    return Coverage(tuple(sorted(used)), scanned, len(tokenizer))


@dataclass(frozen=True, slots=True)
class Remap:
    """Translation between the tokenizer's ids and a trimmed model's."""

    keep: tuple[int, ...]
    _to_new: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._to_new.update({old: new for new, old in enumerate(self.keep)})

    def to_new(self, ids: Iterable[int]) -> list[int]:
        """Tokenizer ids to model ids. An id outside the kept set is a bug in selection,
        not something to paper over with `unk`."""
        out: list[int] = []
        for old in ids:
            new = self._to_new.get(old)
            if new is None:
                message = f"token id {old} was trimmed away but the text still produces it"
                raise TrimError(message)
            out.append(new)
        return out

    def to_old(self, ids: Iterable[int]) -> list[int]:
        """Model ids back to tokenizer ids, for decoding."""
        size = len(self.keep)
        return [self.keep[i] for i in ids if 0 <= i < size]

    def tables(self, original_size: int, unk_id: int) -> VocabTables:
        """Dense lookup tables for the generation path, built once per run.

        `to_new`/`to_old` above are per-id and allocate a list; a batch of beams is
        hundreds of thousands of ids per step, so generation indexes a tensor instead.
        """
        return VocabTables(self.keep, original_size, unk_id)


class VocabTables:
    """Old<->new id translation as tensor indexing, on whatever device the ids are on.

    Generation meets text the trim never scanned, so a source id outside the kept set is
    an ordinary out-of-vocabulary token and becomes `unk`: on FLORES+ against the
    fine-tuning corpus's 52,209 tokens that is 0.07-0.22% of tokens, but it touches
    28-55 sentences of every thousand, so raising would end a scoring run rather than
    cost it a token. `Remap.to_new` still raises — on the training path the corpus *is*
    what was scanned, and an unseen id there is a selection bug.
    """

    def __init__(self, keep: Sequence[int], original_size: int, unk_id: int) -> None:
        import torch

        if original_size < len(keep):
            message = f"original_size {original_size} is smaller than the {len(keep)} kept ids"
            raise TrimError(message)
        self._old = torch.tensor(list(keep), dtype=torch.long)
        self._new = torch.full((original_size,), -1, dtype=torch.long)
        self._new[self._old] = torch.arange(len(keep), dtype=torch.long)
        self._unk = self.new_id(unk_id)

    def to_new(self, ids: torch.Tensor) -> torch.Tensor:
        """Tokenizer ids to model ids, unrepresentable ones to `unk`. The fill is what
        keeps a -1 out of the embedding index, which torch would read as its last row."""
        mapped = self._new.to(ids.device)[ids]
        return mapped.masked_fill(mapped < 0, self._unk)

    def to_old(self, ids: torch.Tensor) -> torch.Tensor:
        return self._old.to(ids.device)[ids]

    def new_id(self, old: int) -> int:
        """One id, for the forced language token and for `unk`. Out of range or trimmed
        away raises: `generate` would otherwise force a token the model cannot produce,
        and `to_new` would have no id to fall back to."""
        if not 0 <= old < len(self._new):
            message = f"token id {old} is outside the untrimmed vocabulary"
            raise TrimError(message)
        new = int(self._new[old])
        if new < 0:
            message = f"token id {old} was trimmed away and cannot be used"
            raise TrimError(message)
        return new


def trim_model(model: Trimmable, keep: Sequence[int]) -> Trimmable:
    """Gather the kept rows of the shared embedding and resize the model to match.

    NLLB ties the encoder embedding, the decoder embedding and the output projection, so
    one gather covers all three. The final-logits bias is a buffer over the vocabulary and
    is gathered with them.
    """
    import torch

    index = torch.tensor(list(keep), dtype=torch.long)
    gathered = model.get_input_embeddings().weight.index_select(0, index).clone()

    model.resize_token_embeddings(len(keep))
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(gathered)
    model.tie_weights()

    bias: torch.Tensor | None = getattr(model, "final_logits_bias", None)
    if bias is not None:
        model.register_buffer("final_logits_bias", bias.index_select(1, index))

    model.config.vocab_size = len(keep)
    return model


def write_keep(coverage_result: Coverage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keep": list(coverage_result.keep),
        "kept": coverage_result.kept,
        "original_size": coverage_result.original_size,
        "share": round(coverage_result.share, 6),
        "scanned": coverage_result.scanned,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def read_keep(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(int(i) for i in payload["keep"])
