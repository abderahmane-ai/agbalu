"""The SFT data files on disk: where each one lives and how it is read and written (11.22).

Generated files are written whole or not at all; a file the operator writes (a reply, a
review) is read strictly and never rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Final

from agbalu.llm.teacher import REPLY_DIR, Candidate

INTERIM: Final = Path("data/interim/llm/sft")
SEEDS: Final = INTERIM / "seeds.jsonl"
CANDIDATES: Final = INTERIM / "candidates.jsonl"
LANGUAGE_SIGNALS: Final = INTERIM / "signals-language.jsonl"
FLUENCY_SIGNALS: Final = INTERIM / "signals-fluency.jsonl"
REVIEW_SHEET: Final = INTERIM / "review-sheet.jsonl"
LEDGER: Final = INTERIM / "ledger.jsonl"
REVIEWS: Final = REPLY_DIR / "reviews.jsonl"
"""The operator's verdicts, beside the replies: both are human input, kept append-only."""

PILOT: Final = INTERIM / "pilot"
REQUESTS_PROMPT: Final = INTERIM / "requests.txt"
REQUESTS_REPLY: Final = REPLY_DIR / "requests.txt"
TRANSLATIONS: Final = INTERIM / "translations.jsonl"


class FilesError(Exception):
    """A file is missing or holds what it should not."""


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    partial.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if not path.is_file():
        message = f"{path} not found"
        raise FilesError(message)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                message = f"{path}:{number} is not a JSON object"
                raise FilesError(message)
            yield row


def candidate_from(row: Mapping[str, object], where: str) -> Candidate:
    turns = row.get("turns")
    if not isinstance(turns, list) or not all(isinstance(t, dict) for t in turns):
        message = f"{where}: turns is not a list of objects"
        raise FilesError(message)
    fields = ("id", "card", "batch", "model", "date")
    values = [row.get(name) for name in fields]
    number, version = row.get("n"), row.get("version")
    if not all(isinstance(v, str) for v in values) or not isinstance(number, int):
        message = f"{where}: a candidate field is missing or mistyped"
        raise FilesError(message)
    if not isinstance(version, int):
        message = f"{where}: version is {version!r}"
        raise FilesError(message)
    return Candidate(
        str(values[0]),
        str(values[1]),
        str(values[2]),
        number,
        tuple({str(k): str(v) for k, v in turn.items()} for turn in turns),
        str(values[3]),
        str(values[4]),
        version,
    )


def read_candidates(path: Path = CANDIDATES) -> list[Candidate]:
    return [candidate_from(row, f"{path}") for row in read_jsonl(path)] if path.is_file() else []


def candidate_rows(candidates: Iterable[Candidate]) -> Iterator[dict[str, object]]:
    for candidate in candidates:
        yield {**asdict(candidate), "turns": list(candidate.turns)}


def read_signals(path: Path) -> dict[str, dict[str, object]]:
    """Signals per candidate id; an absent file is an absent pass."""
    if not path.is_file():
        return {}
    found: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        name = row.pop("candidate", None)
        if not isinstance(name, str):
            message = f"{path}: a signal row has no candidate id"
            raise FilesError(message)
        found[name] = row
    return found


def signal_rows(signals: Mapping[str, Mapping[str, object]]) -> Iterator[dict[str, object]]:
    for name, values in sorted(signals.items()):
        yield {"candidate": name, **values}


def manifests(directory: Path, kind: str) -> list[tuple[str, list[str], int]]:
    """Every batch of `kind`: its name, the card ids it asked for and the prompt version."""
    found: list[tuple[str, list[str], int]] = []
    for path in sorted(directory.glob(f"{kind}-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        asked = row.get("cards")
        version = row.get("version")
        if not isinstance(asked, list) or not all(isinstance(x, str) for x in asked):
            message = f"{path}: no list of ids"
            raise FilesError(message)
        if not isinstance(version, int):
            message = f"{path}: version is {version!r}"
            raise FilesError(message)
        found.append((path.stem, [str(x) for x in asked], version))
    return found


def next_number(directory: Path, kind: str) -> int:
    numbers = [int(path.stem.split("-")[-1]) for path in directory.glob(f"{kind}-*.json")]
    return max(numbers, default=0) + 1
