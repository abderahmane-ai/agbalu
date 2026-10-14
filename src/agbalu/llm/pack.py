"""The training stream: every document tokenised once, scheduled, shuffled and packed.

Three inputs, each read whole: the Kabyle documents, the aligned pairs and the replay
sample. Each kind is repeated `recipe.*_PASSES` times, every pass in its own order, then
the whole schedule is shuffled once under `recipe.SHUFFLE_SEED` and written as
`SEQUENCE_LENGTH`-token blocks of `uint32`. The trainer reads the blocks in file order, so
the shuffle happens here and nowhere else.

Every document starts with `eos`, the only separator, and the trainer finds document
boundaries by it. Tokenising with `split_special_tokens` is what keeps that sound: a web
page containing the literal string `<|endoftext|>` would otherwise put a separator inside
a document.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from agbalu.llm.recipe import ALIGNED_PASSES, KABYLE_PASSES, SEQUENCE_LENGTH, SHUFFLE_SEED

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

TOKEN_DTYPE: Final = np.uint32
"""The vocabulary is 248,320, past `uint16`."""

ENCODE_CHUNK: Final = 4096
WRITE_CHUNK: Final = 200_000


class PackError(Exception):
    """An input is unreadable, or the stream does not account for its documents."""


@dataclass(frozen=True, slots=True)
class Kind:
    name: str
    path: Path
    passes: int


def kinds(kabyle: Path, aligned: Path, replay: Path) -> tuple[Kind, ...]:
    return (
        Kind("kabyle", kabyle, KABYLE_PASSES),
        Kind("aligned", aligned, ALIGNED_PASSES),
        Kind("replay", replay, 1),
    )


def read_texts(path: Path) -> tuple[list[str], list[str]]:
    """Every row's text and source, in file order."""
    if not path.is_file():
        message = f"{path} not found"
        raise PackError(message)
    texts: list[str] = []
    sources: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            text, source = row.get("text"), row.get("source")
            if not isinstance(text, str) or not isinstance(source, str):
                message = f"{path}:{number} lacks a string text and source"
                raise PackError(message)
            texts.append(text)
            sources.append(source)
    if not texts:
        message = f"{path} holds no document"
        raise PackError(message)
    return texts, sources


_WORKER: dict[str, PreTrainedTokenizerBase] = {}
"""One tokenizer per worker process, loaded by the pool's initializer."""


def _load(repo: str) -> None:
    from transformers import AutoTokenizer

    _WORKER["tokenizer"] = AutoTokenizer.from_pretrained(repo)


def _encode(texts: list[str]) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """One chunk as a flat token array and per-document lengths, never as Python ints."""
    tokenizer = _WORKER.get("tokenizer")
    if tokenizer is None:
        message = "worker tokenizer was not loaded"
        raise PackError(message)
    ids = tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    lengths = np.fromiter((len(row) for row in ids), dtype=np.int64, count=len(ids))
    flat = np.fromiter(
        itertools.chain.from_iterable(ids), dtype=TOKEN_DTYPE, count=int(lengths.sum())
    )
    return flat, lengths


def chunks(texts: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(texts), size):
        yield list(texts[start : start + size])


def prefix_eos(
    flat: npt.NDArray[np.uint32], lengths: npt.NDArray[np.int64], eos: int
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """Put `eos` before every document. Refuses an `eos` already inside one."""
    if int(lengths.sum()) != len(flat):
        message = f"lengths sum to {int(lengths.sum())} for {len(flat)} tokens"
        raise PackError(message)
    stray = int((flat == eos).sum())
    if stray:
        message = f"{stray} eos tokens inside documents; the boundaries would be wrong"
        raise PackError(message)
    sizes = lengths + 1
    starts = np.cumsum(sizes) - sizes
    out = np.empty(int(sizes.sum()), dtype=TOKEN_DTYPE)
    body = np.ones(len(out), dtype=bool)
    body[starts] = False
    out[starts] = eos
    out[body] = flat
    return out, sizes


@dataclass(frozen=True, slots=True)
class Encoded:
    """Every document of every kind, `eos`-prefixed, in one flat array."""

    tokens: npt.NDArray[np.uint32]
    offsets: npt.NDArray[np.int64]
    lengths: npt.NDArray[np.int64]
    kind_of: npt.NDArray[np.uint8]


def assemble(
    parts: Sequence[tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64], int]],
) -> Encoded:
    """`(tokens, lengths, kind code)` parts as one array with document offsets."""
    if not parts:
        message = "nothing was encoded"
        raise PackError(message)
    tokens = np.concatenate([p[0] for p in parts])
    lengths = np.concatenate([p[1] for p in parts])
    kind_of = np.concatenate([np.full(len(p[1]), p[2], dtype=np.uint8) for p in parts])
    offsets = np.cumsum(lengths) - lengths
    return Encoded(tokens, offsets, lengths, kind_of)


def schedule(
    kind_of: npt.NDArray[np.uint8], passes: Sequence[int], seed: int = SHUFFLE_SEED
) -> npt.NDArray[np.int64]:
    """Document indices, each kind repeated by its passes, then shuffled as one stream."""
    generator = np.random.default_rng(seed)
    entries: list[npt.NDArray[np.int64]] = []
    for code, count in enumerate(passes):
        members = np.flatnonzero(kind_of == code).astype(np.int64)
        entries.extend(generator.permutation(members) for _ in range(count))
    order = np.concatenate(entries) if entries else np.empty(0, dtype=np.int64)
    return generator.permutation(order)


def gather(
    tokens: npt.NDArray[np.uint32],
    offsets: npt.NDArray[np.int64],
    lengths: npt.NDArray[np.int64],
    documents: npt.NDArray[np.int64],
) -> npt.NDArray[np.uint32]:
    """The tokens of `documents`, concatenated, without a Python loop over them."""
    sizes = lengths[documents]
    starts = offsets[documents]
    ends = np.cumsum(sizes)
    within = np.arange(int(ends[-1]) if len(ends) else 0, dtype=np.int64)
    shift = np.repeat(starts - (ends - sizes), sizes)
    gathered: npt.NDArray[np.uint32] = tokens[within + shift]
    return gathered


@dataclass(frozen=True, slots=True)
class Written:
    blocks: int
    length: int
    tail: int
    stream_tokens: int


def write_blocks(
    encoded: Encoded, order: npt.NDArray[np.int64], out: Path, length: int = SEQUENCE_LENGTH
) -> Written:
    """The scheduled stream as `(blocks, length)` rows; the final partial block is dropped."""
    stream = int(encoded.lengths[order].sum())
    blocks = stream // length
    if blocks < 1:
        message = f"the stream holds {stream} tokens, short of one {length}-token block"
        raise PackError(message)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    target = np.lib.format.open_memmap(
        partial, mode="w+", dtype=TOKEN_DTYPE, shape=(blocks, length)
    )
    flat = target.reshape(-1)
    position = 0
    capacity = blocks * length
    for start in range(0, len(order), WRITE_CHUNK):
        piece = gather(
            encoded.tokens, encoded.offsets, encoded.lengths, order[start : start + WRITE_CHUNK]
        )
        take = min(len(piece), capacity - position)
        flat[position : position + take] = piece[:take]
        position += take
        if position == capacity:
            break
    target.flush()
    del target
    partial.replace(out)
    if position != capacity:
        message = f"wrote {position} tokens into a stream sized for {capacity}"
        raise PackError(message)
    return Written(blocks=blocks, length=length, tail=stream - capacity, stream_tokens=stream)


def encode_all(
    texts: Sequence[str], repo: str, workers: int, eos: int
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """Every text `eos`-prefixed, in input order, tokenised across `workers` processes."""
    tokens: list[npt.NDArray[np.uint32]] = []
    lengths: list[npt.NDArray[np.int64]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_load, initargs=(repo,)) as pool:
        for flat, sizes in pool.map(_encode, chunks(texts, ENCODE_CHUNK)):
            prefixed, prefixed_sizes = prefix_eos(flat, sizes, eos)
            tokens.append(prefixed)
            lengths.append(prefixed_sizes)
    joined = np.concatenate(lengths) if lengths else np.empty(0, dtype=np.int64)
    if len(joined) != len(texts):
        message = f"{len(joined)} encodings for {len(texts)} texts"
        raise PackError(message)
    return np.concatenate(tokens), joined


def summary(
    encoded: Encoded,
    order: npt.NDArray[np.int64],
    written: Written,
    names: Sequence[str],
    passes: Sequence[int],
    sources: Sequence[str],
) -> dict[str, object]:
    """Per-kind and per-source tokens, and the conservation the stream must satisfy."""
    per_kind: list[dict[str, object]] = []
    stream_lengths = encoded.lengths[order]
    stream_kinds = encoded.kind_of[order]
    for code, (name, count) in enumerate(zip(names, passes, strict=True)):
        mask = encoded.kind_of == code
        stream = int(stream_lengths[stream_kinds == code].sum())
        expected = count * int(encoded.lengths[mask].sum())
        if stream != expected:
            message = f"{name}: {stream} stream tokens, {count} passes predict {expected}"
            raise PackError(message)
        per_kind.append(
            {
                "kind": name,
                "documents": int(mask.sum()),
                "unique_tokens": int(encoded.lengths[mask].sum()),
                "passes": count,
                "stream_tokens": stream,
                "share": round(stream / written.stream_tokens, 4),
            }
        )
    by_source: dict[str, int] = {}
    for source, size in zip(sources, encoded.lengths.tolist(), strict=True):
        by_source[source] = by_source.get(source, 0) + size
    if written.blocks * written.length + written.tail != written.stream_tokens:
        message = (
            f"{written.blocks} blocks plus a {written.tail}-token tail do not make "
            f"the {written.stream_tokens}-token stream"
        )
        raise PackError(message)
    return {
        "kinds": per_kind,
        "unique_tokens_by_source": dict(sorted(by_source.items())),
        "blocks": written.blocks,
        "sequence_length": written.length,
        "stream_tokens": written.stream_tokens,
        "written_tokens": written.blocks * written.length,
        "dropped_tail_tokens": written.tail,
        "seed": SHUFFLE_SEED,
    }
