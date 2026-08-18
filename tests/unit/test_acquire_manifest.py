from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agbalu.acquire.manifest import Manifest, ManifestError
from agbalu.acquire.models import Deferral, ManifestEntry, Removal

DIGEST = "b" * 64


def make_entry(
    source_id: str = "hf.example.kab",
    path: str = "train.parquet",
    fetched_at: datetime | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        source_id=source_id,
        path=path,
        bytes=7,
        sha256=DIGEST,
        kind="text",
        target="local",
        uri="https://example.invalid/kab",
        fetched_at=fetched_at or datetime.now(UTC),
    )


def make_removal(
    source_id: str = "hf.example.kab",
    path: str = "train.parquet",
    removed_at: datetime | None = None,
) -> Removal:
    return Removal(
        source_id=source_id,
        path=path,
        sha256=DIGEST,
        reason="redundant",
        detail="byte-identical to another landed artifact",
        removed_at=removed_at or datetime.now(UTC),
    )


def make_deferral(source_id: str, reason: str = "remote-target") -> Deferral:
    return Deferral.model_validate(
        {
            "source_id": source_id,
            "reason": reason,
            "detail": "not landed locally",
            "recorded_at": datetime.now(UTC),
        }
    )


def test_reading_an_absent_manifest_yields_nothing(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    assert manifest.entries() == ()
    assert manifest.deferrals() == ()


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    entry = make_entry()
    manifest.append(entry)
    assert manifest.entries() == (entry,)


def test_appends_accumulate_in_order(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    for index in range(5):
        manifest.append(make_entry(path=f"part-{index}.parquet"))
    assert [e.path for e in manifest.entries()] == [f"part-{i}.parquet" for i in range(5)]


def test_manifest_creates_its_parent_directory(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "deep" / "raw")
    manifest.append(make_entry())
    assert manifest.entries_path.is_file()


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry())
    with manifest.entries_path.open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    assert len(manifest.entries()) == 1


def test_a_malformed_row_names_its_line_number(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry())
    with manifest.entries_path.open("a", encoding="utf-8") as handle:
        handle.write('{"source_id": "x"}\n')
    with pytest.raises(ManifestError, match=r":2 is not a valid manifest entry"):
        manifest.entries()


def test_a_truncated_row_is_an_error_not_a_silent_drop(tmp_path: Path) -> None:
    # A kill mid-append must be loud: provenance silently lost is worse than a crash.
    manifest = Manifest(tmp_path)
    manifest.append(make_entry())
    with manifest.entries_path.open("a", encoding="utf-8") as handle:
        handle.write('{"source_id": "hf.x", "path": "a.txt", "byt')
    with pytest.raises(ManifestError):
        manifest.entries()


def test_non_utf8_bytes_raise(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.entries_path.write_bytes(b"\xff\xfe not utf-8\n")
    with pytest.raises(ManifestError, match="not valid UTF-8"):
        manifest.entries()


def test_by_source_filters(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry(source_id="hf.a"))
    manifest.append(make_entry(source_id="hf.b"))
    assert [e.source_id for e in manifest.by_source("hf.b")] == ["hf.b"]


def test_recorded_source_ids_covers_both_files(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry(source_id="hf.landed"))
    manifest.defer(make_deferral("hf.deferred"))
    assert manifest.recorded_source_ids() == frozenset({"hf.landed", "hf.deferred"})


def test_a_landed_source_supersedes_its_earlier_deferral(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.defer(make_deferral("opus.tatoeba-kab", reason="unavailable"))
    manifest.append(make_entry(source_id="opus.tatoeba-kab"))
    assert manifest.deferrals()  # history is preserved
    assert manifest.active_deferrals() == ()  # current state is derived


def test_the_latest_deferral_per_source_wins(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.defer(make_deferral("hf.x", reason="unavailable"))
    manifest.defer(make_deferral("hf.x", reason="gated"))
    active = manifest.active_deferrals()
    assert len(active) == 1
    assert active[0].reason == "gated"


def test_unicode_survives_a_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry(path="tazwart/aɣbalu-ɛ-ḥ.txt"))
    assert manifest.entries()[0].path == "tazwart/aɣbalu-ɛ-ḥ.txt"


def test_no_removals_is_an_empty_set(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry())
    assert manifest.removals() == ()
    assert manifest.removed_artifacts() == frozenset()


def test_removal_round_trips(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry())
    manifest.record_removal(make_removal())
    assert manifest.entries()  # the provenance row is never erased
    assert manifest.removed_artifacts() == frozenset({("hf.example.kab", "train.parquet")})


def test_a_removal_is_scoped_to_one_artifact(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry(path="a.bin"))
    manifest.append(make_entry(path="b.bin"))
    manifest.record_removal(make_removal(path="a.bin"))
    assert manifest.removed_artifacts() == frozenset({("hf.example.kab", "a.bin")})


def test_a_refetch_after_removal_clears_it(tmp_path: Path) -> None:
    """Re-fetching is how a removal is undone; both rows stay on disk."""
    manifest = Manifest(tmp_path)
    early = datetime(2026, 1, 1, tzinfo=UTC)
    manifest.append(make_entry(fetched_at=early))
    manifest.record_removal(make_removal(removed_at=datetime(2026, 2, 1, tzinfo=UTC)))
    manifest.append(make_entry(fetched_at=datetime(2026, 3, 1, tzinfo=UTC)))
    assert manifest.removed_artifacts() == frozenset()


def test_a_fetch_before_the_removal_does_not_clear_it(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.append(make_entry(fetched_at=datetime(2026, 1, 1, tzinfo=UTC)))
    manifest.record_removal(make_removal(removed_at=datetime(2026, 2, 1, tzinfo=UTC)))
    assert manifest.removed_artifacts() == frozenset({("hf.example.kab", "train.parquet")})


def test_a_malformed_removal_names_its_line_number(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    manifest.record_removal(make_removal())
    manifest.removals_path.write_text(
        manifest.removals_path.read_text(encoding="utf-8") + '{"source_id": "x"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match=re.escape("removals.jsonl:2")):
        manifest.removals()


def test_a_naive_removal_timestamp_is_refused() -> None:
    with pytest.raises(ValueError, match="removed_at must be timezone-aware"):
        make_removal(removed_at=datetime(2026, 2, 1))
