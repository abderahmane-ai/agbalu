"""Read a landed artifact into flat string records, whatever its container.

Every reader yields `Mapping[str, str]`; nested JSON is flattened with dotted keys
so a Kabyle field buried in a `translation` object is still a candidate column.
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import pyarrow.parquet as pq

csv.field_size_limit(sys.maxsize)

DELIMITED: Final[dict[str, str]] = {".tsv": "\t", ".csv": ",", ".txt": "\t"}
PARQUET: Final = frozenset({".parquet"})
JSON_LINES: Final = frozenset({".jsonl", ".ndjson"})
JSON_WHOLE: Final = frozenset({".json"})
ARCHIVE: Final = frozenset({".zip"})

READABLE: Final = PARQUET | JSON_LINES | JSON_WHOLE | ARCHIVE | frozenset(DELIMITED)

PARQUET_BATCH: Final = 8192
MAX_FLATTEN_DEPTH: Final = 3


class ReadError(Exception):
    """An artifact could not be read as records."""


def _flatten(value: Any, prefix: str = "", depth: int = 0) -> Iterator[tuple[str, str]]:  # noqa: ANN401
    if depth > MAX_FLATTEN_DEPTH:
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _flatten(inner, f"{prefix}.{key}" if prefix else str(key), depth + 1)
    elif isinstance(value, str):
        yield prefix or "text", value
    elif isinstance(value, (int, float, bool)):
        yield prefix or "text", str(value)


def _record(obj: Any) -> Mapping[str, str]:  # noqa: ANN401
    return dict(_flatten(obj))


def read_parquet(path: Path) -> Iterator[Mapping[str, str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=PARQUET_BATCH):
        for row in batch.to_pylist():
            yield _record(row)


def read_jsonl(path: Path) -> Iterator[Mapping[str, str]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield _record(json.loads(stripped))
            except json.JSONDecodeError:
                continue


def read_json(path: Path) -> Iterator[Mapping[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError, MemoryError):
        return
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        yield _record(row)


def read_delimited(path: Path, delimiter: str) -> Iterator[Mapping[str, str]]:
    """Read a delimited file with quoting disabled.

    `QUOTE_NONE` is mandatory: the default swallows every line after a `"` inside a
    sentence and loses 5,904 rows of the seed corpus without raising (CLAUDE.md 2.1a).
    """
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        header = sample.rstrip("\r\n").split(delimiter)
        looks_named = bool(header) and not any(_looks_like_sentence(c) for c in header)
        reader = csv.reader(handle, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        if looks_named:
            next(reader, None)
            names = header
        else:
            names = [f"col{i}" for i in range(len(header))]
        for row in reader:
            yield {names[i] if i < len(names) else f"col{i}": v for i, v in enumerate(row)}


HEADER_MAX_CHARS: Final = 40


def _looks_like_sentence(field: str) -> bool:
    return " " in field.strip() or len(field) > HEADER_MAX_CHARS


def read_zip(path: Path) -> Iterator[Mapping[str, str]]:
    """Read the Kabyle side of an OPUS moses bundle.

    A bundle holds one line-aligned plain-text file per language, so the `.kab`
    member is the Kabyle column and needs no detection.
    """
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith(".kab")]
        if not members:
            return
        for name in members:
            with archive.open(name) as raw:
                for line in raw:
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        yield {"kab": text}


def read_lines(path: Path) -> Iterator[Mapping[str, str]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield {"text": text}


def read_records(path: Path) -> Iterator[Mapping[str, str]]:
    """Dispatch on suffix. Unreadable containers yield nothing rather than raising."""
    suffix = path.suffix.lower()
    try:
        if suffix in PARQUET:
            yield from read_parquet(path)
        elif suffix in JSON_LINES:
            yield from read_jsonl(path)
        elif suffix in JSON_WHOLE:
            yield from read_json(path)
        elif suffix in ARCHIVE:
            yield from read_zip(path)
        elif suffix == ".txt":
            yield from read_lines(path)
        elif suffix in DELIMITED:
            yield from read_delimited(path, DELIMITED[suffix])
    except (OSError, zipfile.BadZipFile, ValueError, UnicodeError) as exc:
        msg = f"{path}: {exc}"
        raise ReadError(msg) from exc
