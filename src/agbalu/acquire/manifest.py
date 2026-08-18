"""The provenance record: append-only JSONL beside the bytes it describes.

Two homogeneous files rather than one tagged stream, so either can be read with
`json.loads` per line without dispatching on a discriminator:

    manifest.jsonl   one row per artifact actually landed
    deferrals.jsonl  one row per source deliberately not landed, with the reason
    removals.jsonl   one row per landed artifact deliberately deleted, with the reason

All three are append-only. A correction is a new row plus a note in the phase record,
never an edit — `data/raw/` is immutable (CLAUDE.md 2.7). Deleting bytes is therefore
recorded, not erased: the manifest row stays, a removal row joins it, and `verify`
derives the current state from both.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from agbalu.acquire.models import Deferral, ManifestEntry, Removal

MANIFEST_NAME: Final = "manifest.jsonl"
DEFERRALS_NAME: Final = "deferrals.jsonl"
REMOVALS_NAME: Final = "removals.jsonl"


class ManifestError(Exception):
    """The manifest is unreadable or holds a row that does not validate."""


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_lines(path: Path) -> list[tuple[int, str]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = f"manifest is not valid UTF-8: {path}"
        raise ManifestError(msg) from exc
    return [(n, line) for n, line in enumerate(text.splitlines(), start=1) if line.strip()]


class Manifest:
    """Reader/writer for one `data/raw/` provenance pair."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries_path = root / MANIFEST_NAME
        self.deferrals_path = root / DEFERRALS_NAME
        self.removals_path = root / REMOVALS_NAME

    def append(self, entry: ManifestEntry) -> None:
        _append_line(self.entries_path, entry.model_dump_json())

    def defer(self, deferral: Deferral) -> None:
        _append_line(self.deferrals_path, deferral.model_dump_json())

    def record_removal(self, removal: Removal) -> None:
        _append_line(self.removals_path, removal.model_dump_json())

    def entries(self) -> tuple[ManifestEntry, ...]:
        rows: list[ManifestEntry] = []
        for number, line in _read_lines(self.entries_path):
            try:
                rows.append(ManifestEntry.model_validate_json(line))
            except ValidationError as exc:
                msg = f"{self.entries_path}:{number} is not a valid manifest entry\n{exc}"
                raise ManifestError(msg) from exc
        return tuple(rows)

    def deferrals(self) -> tuple[Deferral, ...]:
        rows: list[Deferral] = []
        for number, line in _read_lines(self.deferrals_path):
            try:
                rows.append(Deferral.model_validate_json(line))
            except ValidationError as exc:
                msg = f"{self.deferrals_path}:{number} is not a valid deferral\n{exc}"
                raise ManifestError(msg) from exc
        return tuple(rows)

    def removals(self) -> tuple[Removal, ...]:
        rows: list[Removal] = []
        for number, line in _read_lines(self.removals_path):
            try:
                rows.append(Removal.model_validate_json(line))
            except ValidationError as exc:
                msg = f"{self.removals_path}:{number} is not a valid removal\n{exc}"
                raise ManifestError(msg) from exc
        return tuple(rows)

    def removed_artifacts(self) -> frozenset[tuple[str, str]]:
        """`(source_id, path)` for every artifact recorded as deleted and not re-fetched.

        A later manifest row for the same artifact supersedes the removal: re-fetching is
        how a removal is undone, and both rows stay on disk.
        """
        removals = self.removals()
        if not removals:
            return frozenset()
        latest_fetch: dict[tuple[str, str], datetime] = {}
        for entry in self.entries():
            key = (entry.source_id, entry.path)
            seen = latest_fetch.get(key)
            if seen is None or entry.fetched_at > seen:
                latest_fetch[key] = entry.fetched_at
        return frozenset(
            key
            for removal in removals
            if (key := (removal.source_id, removal.path)) not in latest_fetch
            or latest_fetch[key] < removal.removed_at
        )

    def active_deferrals(self) -> tuple[Deferral, ...]:
        """Deferrals not superseded by a later successful fetch.

        The files are append-only, so a source deferred in one run and landed in
        the next keeps its old deferral row. History is preserved; the current
        state is derived, and the most recent row per source wins.
        """
        landed = {e.source_id for e in self.entries()}
        current: dict[str, Deferral] = {}
        for deferral in self.deferrals():
            if deferral.source_id not in landed:
                current[deferral.source_id] = deferral
        return tuple(current.values())

    def by_source(self, source_id: str) -> tuple[ManifestEntry, ...]:
        return tuple(e for e in self.entries() if e.source_id == source_id)

    def recorded_source_ids(self) -> frozenset[str]:
        """Sources that are either landed or explicitly deferred."""
        return frozenset(e.source_id for e in self.entries()) | frozenset(
            d.source_id for d in self.deferrals()
        )
