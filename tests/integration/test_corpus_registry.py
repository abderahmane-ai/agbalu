"""The shipped registry must always be valid and internally consistent."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agbalu.registry import Registry, load_registry
from agbalu.registry.cli import main, summarise

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "resources" / "corpus_registry.yaml"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def registry() -> Registry:
    return load_registry(REGISTRY_PATH)


def test_registry_validates(registry: Registry) -> None:
    assert len(registry.sources) > 0


def test_every_source_covers_kabyle(registry: Registry) -> None:
    for source in registry.sources:
        assert any(lang.startswith("kab") for lang in source.languages), source.id


def test_ids_are_unique(registry: Registry) -> None:
    ids = [s.id for s in registry.sources]
    assert len(ids) == len(set(ids))


def test_no_source_silently_lacks_a_licence(registry: Registry) -> None:
    for source in registry.sources:
        assert source.licence.strip(), source.id


def test_unclear_licences_are_documented(registry: Registry) -> None:
    # An unclear licence is acceptable only if the note says what must be resolved.
    for source in registry.filter(redistribution="unclear"):
        assert source.notes.strip(), f"{source.id} has an unclear licence and no note"


def test_excluded_sources_state_a_reason(registry: Registry) -> None:
    for source in registry.filter(tier="excluded"):
        assert source.notes.strip(), source.id


def test_benchmarks_are_never_core(registry: Registry) -> None:
    # FLORES+/SIB-200 must stay out of training data; tier is the guard.
    for source_id in ("hf.flores-plus-kab", "hf.sib200-kab"):
        assert registry.by_id(source_id).tier == "reference"


def test_common_voice_is_the_largest_speech_source(registry: Registry) -> None:
    speech = registry.filter(modality="speech")
    assert speech
    largest = max(speech, key=lambda s: s.size.hours or 0.0)
    assert largest.id == "cv.common-voice-26-kab"
    assert (largest.size.hours or 0.0) > 500.0


@pytest.mark.parametrize(
    ("source_id", "rows"),
    [("local.tatoeba-kab-eng", 140324), ("local.tatoeba-kab-mono", 791018)],
)
def test_local_seed_corpora_are_registered(registry: Registry, source_id: str, rows: int) -> None:
    seed = registry.by_id(source_id)
    assert seed.access == "manual"
    assert seed.size.rows == rows


@pytest.mark.parametrize("source_id", ["local.tatoeba-kab-eng", "local.tatoeba-kab-mono"])
def test_seed_files_referenced_by_registry_exist(registry: Registry, source_id: str) -> None:
    seed = registry.by_id(source_id)
    path = REGISTRY_PATH.parents[1] / seed.uri
    assert path.is_file(), f"registry points at a missing file: {path}"
    assert path.stat().st_size == seed.size.bytes


def test_local_sources_use_repo_relative_paths(registry: Registry) -> None:
    for source in registry.sources:
        if source.access == "manual":
            assert not source.uri.startswith(("/", "http")), source.id


@pytest.mark.slow
def test_local_source_bytes_match_their_declared_checksum(registry: Registry) -> None:
    """The registry claims to pin these bytes. Verify it actually does."""
    for source in registry.sources:
        if source.access != "manual" or source.checksum is None:
            continue
        path = REGISTRY_PATH.parents[1] / source.uri
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        assert digest.hexdigest() == source.checksum, f"{source.id} bytes changed on disk"


def test_summary_reports_every_source(registry: Registry) -> None:
    text = summarise(registry)
    assert f"{len(registry.sources)} sources" in text
    assert "by modality:" in text
    assert "speech hours:" in text


def test_cli_succeeds_on_the_shipped_registry() -> None:
    assert main([str(REGISTRY_PATH)]) == 0


def test_cli_fails_on_a_missing_file(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope.yaml")]) == 1
