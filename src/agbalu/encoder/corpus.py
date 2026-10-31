"""Convert the existing HPC Kabyle text into memory-mapped encoder inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
import sentencepiece as spm
from numpy.typing import NDArray
from sentencepiece import sentencepiece_pb2

from agbalu.encoder.configuration import SPECIAL_TOKENS, VOCABULARY_SIZE, Locations
from agbalu.encoder.packing import word_groups
from agbalu.llm.decontaminate import WINDOW, Decontaminator
from agbalu.llm.progress import Tracker, stage
from agbalu.llm.provenance import code_stamp, file_digest
from agbalu.treebank import Split, read_split

TOKENIZER_SHA = "c8094fccd936d2e8954809bd9cf45331679e9550d2c7bbabe5c38c3bf365dc4e"
FORMAT_VERSION = 1


def integer_field(piece: object, name: str) -> int:
    value: object = getattr(piece, name)
    if not isinstance(value, int):
        message = f"SentencePiece {name}: expected integer, got {value!r}"
        raise TypeError(message)
    return value


def read_json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        message = f"{path}: expected JSON object with string keys"
        raise ValueError(message)
    return {str(key): value for key, value in payload.items()}


def write_json(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.partial")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def tokenizer(path: Path) -> spm.SentencePieceProcessor:
    digest = file_digest(path)
    if digest != TOKENIZER_SHA:
        message = f"{path}: tokenizer SHA256 {digest} != required Mammeri-16k {TOKENIZER_SHA}"
        raise ValueError(message)
    return spm.SentencePieceProcessor(model_file=str(path))


def encode_groups(text: str, processor: spm.SentencePieceProcessor) -> tuple[list[int], list[int]]:
    serialized = processor.encode(text, out_type="serialized_proto", enable_sampling=False)
    if not isinstance(serialized, bytes):
        message = f"SentencePiece returned {type(serialized).__name__}, expected serialized bytes"
        raise TypeError(message)
    # Protobuf creates message classes dynamically; the generated module has no class stub.
    proto: object = getattr(sentencepiece_pb2, "SentencePieceText")()
    getattr(proto, "ParseFromString")(serialized)
    pieces: object = getattr(proto, "pieces")
    if not isinstance(pieces, Sequence):
        message = "SentencePieceText does not expose an offset sequence"
        raise TypeError(message)
    ids = [integer_field(piece, "id") for piece in pieces]
    offsets = [(integer_field(piece, "begin"), integer_field(piece, "end")) for piece in pieces]
    return ids, word_groups(text, offsets)


def treebank_exclusion(locations: Locations) -> Decontaminator:
    splits: tuple[Split, ...] = ("train", "test")
    sentences = [s.text for split in splits for s in read_split(locations.treebank, split)]
    result = Decontaminator.build(sentences, whole=sentences)
    positive = next(text for text in sentences if len(text.split()) >= WINDOW)
    result.require_positive(positive)
    return result


def write_row(
    text: str, processor: spm.SentencePieceProcessor, tokens: BinaryIO, groups: BinaryIO
) -> int:
    ids, ownership = encode_groups(text, processor)
    if any(value < SPECIAL_TOKENS or value >= VOCABULARY_SIZE for value in ids):
        message = f"invalid control/out-of-vocabulary ID in text {text[:120]!r}"
        raise ValueError(message)
    tokens.write(np.asarray(ids, dtype=np.uint16).tobytes())
    origin = groups.tell() // np.dtype(np.int32).itemsize
    groups.write((np.asarray(ownership, dtype=np.int32) + origin).astype(np.int32).tobytes())
    return len(ids)


def prepare(locations: Locations) -> dict[str, object]:
    if not locations.corpus.is_file():
        message = f"missing existing Jugurtha Kabyle corpus: {locations.corpus}"
        raise FileNotFoundError(message)
    processor = tokenizer(locations.tokenizer)
    exclusion = treebank_exclusion(locations)
    identity = {"input_sha256": file_digest(locations.corpus), "tokenizer_sha256": TOKENIZER_SHA}
    stats_path = locations.data / "corpus.stats.json"
    if stats_path.is_file():
        stored = read_json(stats_path)
        if all(stored.get(key) == value for key, value in identity.items()):
            verify_data(locations.data)
            return stored
        message = f"{locations.data}: input/tokenizer changed; use a new data directory"
        raise ValueError(message)
    staging = locations.data.with_name(f".{locations.data.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    documents: list[tuple[int, int, int]] = []
    read = rejected = offset = 0
    tracker = Tracker("Mammeri corpus documents", None)
    input_stat = locations.corpus.stat()
    with (
        locations.corpus.open(encoding="utf-8") as source,
        (staging / "tokens.bin").open("wb") as token_file,
        (staging / "groups.bin").open("wb") as group_file,
        (staging / "rejects.jsonl").open("w", encoding="utf-8") as rejects,
        stage("prepare existing Kabyle corpus"),
    ):
        for number, line in enumerate(source, 1):
            read += 1
            row = json.loads(line)
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                message = (
                    f"{locations.corpus}:{number}: expected nonempty string text, got {text!r}"
                )
                raise ValueError(message)
            if exclusion.matches(text):
                rejected += 1
                rejects.write(
                    json.dumps({"row": number, "reason": "encoder_evaluation_overlap"}) + "\n"
                )
                continue
            count = write_row(text, processor, token_file, group_file)
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            validation = int.from_bytes(digest[:4], "big") % 100 == 0
            documents.append((offset, offset + count, int(validation)))
            offset += count
            tracker.advance()
        for handle in (token_file, group_file, rejects):
            handle.flush()
            os.fsync(handle.fileno())
    after = locations.corpus.stat()
    if (input_stat.st_size, input_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        message = f"{locations.corpus}: input changed during conversion"
        raise ValueError(message)
    if len(documents) + rejected != read:
        message = f"document conservation failed: {len(documents)} + {rejected} != {read}"
        raise ValueError(message)
    table = np.asarray(documents, dtype=np.int64)
    if table.size == 0 or not np.any(table[:, 2] == 1) or not np.any(table[:, 2] == 0):
        message = f"{locations.corpus}: conversion needs nonempty train and validation partitions"
        raise ValueError(message)
    np.save(staging / "documents.npy", table, allow_pickle=False)
    train_tokens = int(np.sum((table[:, 1] - table[:, 0])[table[:, 2] == 0]))
    manifest = {
        name: file_digest(staging / name) for name in ("tokens.bin", "groups.bin", "documents.npy")
    }
    stats: dict[str, object] = {
        **identity,
        "format_version": FORMAT_VERSION,
        "code": code_stamp(),
        "documents_read": read,
        "documents_kept": len(documents),
        "documents_rejected": rejected,
        "text_tokens": offset,
        "train_text_tokens": train_tokens,
        "validation_text_tokens": offset - train_tokens,
        "files": manifest,
    }
    write_json(staging / "corpus.stats.json", stats)
    if locations.data.exists():
        message = f"refusing to replace existing data directory {locations.data}"
        raise FileExistsError(message)
    staging.rename(locations.data)
    verify_data(locations.data)
    return stats


def verify_data(directory: Path) -> dict[str, object]:
    payload = read_json(directory / "corpus.stats.json")
    files = payload["files"]
    if not isinstance(files, dict):
        message = f"{directory}: missing file checksums"
        raise TypeError(message)
    for name, digest in files.items():
        actual = file_digest(directory / name)
        if actual != digest:
            message = f"{directory / name}: checksum {actual} != {digest}"
            raise ValueError(message)
    count = int(str(payload["text_tokens"]))
    if (directory / "tokens.bin").stat().st_size != count * 2:
        message = f"{directory}: token bytes disagree with text_tokens={count}"
        raise ValueError(message)
    if (directory / "groups.bin").stat().st_size != count * 4:
        message = f"{directory}: group bytes disagree with text_tokens={count}"
        raise ValueError(message)
    return payload


def documents(directory: Path) -> NDArray[np.int64]:
    result: NDArray[np.int64] = np.load(
        directory / "documents.npy", mmap_mode="r", allow_pickle=False
    )
    return result
