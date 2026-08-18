"""Integration checks against the real acquired corpus in `data/raw/`.

Skipped when the corpus has not been acquired, so a fresh clone still runs green.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agbalu.acquire.manifest import Manifest
from agbalu.acquire.storage import sha256_file
from agbalu.registry.loader import load_registry

pytestmark = pytest.mark.integration

RAW = Path("data/raw")
REGISTRY = Path("resources/corpus_registry.yaml")
PARALLEL = RAW / "tatoeba" / "tatoeba_kab_eng_2026-08-05.tsv"
MONO = RAW / "tatoeba" / "tatoeba_kab_mono_2026-08-05.tsv"


def manifest_or_skip() -> Manifest:
    manifest = Manifest(RAW)
    if not manifest.entries_path.is_file():
        pytest.skip("corpus not acquired; run `make acquire`")
    return manifest


def test_manifest_rows_all_validate() -> None:
    assert len(manifest_or_skip().entries()) > 0


def test_every_core_source_is_landed_or_deferred() -> None:
    manifest = manifest_or_skip()
    registry = load_registry(REGISTRY)
    recorded = manifest.recorded_source_ids()
    missing = [s.id for s in registry.sources if s.tier == "core" and s.id not in recorded]
    assert missing == [], f"core sources neither landed nor deferred: {missing}"


def test_no_deferral_lacks_a_reason() -> None:
    for deferral in manifest_or_skip().active_deferrals():
        assert deferral.detail.strip()


def test_landed_checksums_match_disk() -> None:
    manifest = manifest_or_skip()
    removed = manifest.removed_artifacts()
    local = [
        e
        for e in manifest.entries()
        if e.target == "local" and (e.source_id, e.path) not in removed
    ]
    # Hash the ten largest: enough to catch a truncated or swapped artifact cheaply.
    for entry in sorted(local, key=lambda e: -e.bytes)[:10]:
        path = RAW / entry.source_id / entry.path
        assert path.is_file(), f"missing: {path}"
        digest, size = sha256_file(path)
        assert digest == entry.sha256, f"checksum drift: {path}"
        assert size == entry.bytes


def test_removed_artifacts_are_actually_gone() -> None:
    """A removal row claims bytes are deleted. If the file is back, the record is stale."""
    manifest = manifest_or_skip()
    for source_id, path in sorted(manifest.removed_artifacts()):
        assert not (RAW / source_id / path).is_file(), f"recorded as removed but present: {path}"


def test_manifest_paths_never_escape_their_source() -> None:
    for entry in manifest_or_skip().entries():
        resolved = (RAW / entry.source_id / entry.path).resolve()
        assert (RAW / entry.source_id).resolve() in resolved.parents


def test_every_entry_records_a_revision_where_upstream_offers_one() -> None:
    # `manual` sources have no upstream revision; everything fetched does.
    manifest = manifest_or_skip()
    registry = load_registry(REGISTRY)
    by_id = {s.id: s for s in registry.sources}
    for entry in manifest.entries():
        source = by_id.get(entry.source_id)
        if source is None or source.access == "manual":
            continue
        assert entry.revision, f"{entry.source_id} landed without a revision pin"


def read_tsv(path: Path, encoding: str, columns: int) -> list[list[str]]:
    with path.open(encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [row for row in reader if len(row) == columns]


@pytest.mark.skipif(not PARALLEL.is_file(), reason="seed parallel corpus absent")
def test_parallel_seed_parses_to_its_documented_row_count() -> None:
    assert len(read_tsv(PARALLEL, "utf-8-sig", 4)) == 140_324


@pytest.mark.skipif(not MONO.is_file(), reason="seed mono corpus absent")
def test_mono_seed_parses_to_its_documented_row_count() -> None:
    assert len(read_tsv(MONO, "utf-8", 3)) == 791_018


@pytest.mark.skipif(not MONO.is_file(), reason="seed mono corpus absent")
def test_default_csv_quoting_silently_loses_rows() -> None:
    """Regression guard for a defect found in Phase 1.

    Python's default QUOTE_MINIMAL treats `"` inside a Tatoeba sentence as a
    quoting character and swallows the following lines. It raises nothing: 5,904
    monolingual sentences simply vanish. Every reader of these files must pass
    `quoting=csv.QUOTE_NONE`.
    """
    csv.field_size_limit(10**9)
    with MONO.open(encoding="utf-8", newline="") as handle:
        lenient = sum(1 for row in csv.reader(handle, delimiter="\t") if len(row) == 3)
    strict = len(read_tsv(MONO, "utf-8", 3))
    assert strict == 791_018
    assert lenient < strict, "default quoting no longer loses rows; re-check the reader policy"
