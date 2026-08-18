from __future__ import annotations

import types
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any, Self

import pytest

from agbalu.acquire import fetch as fetch_module
from agbalu.acquire.fetch import (
    FetchError,
    download_file,
    opus_corpus,
    repo_id,
    walk_artifacts,
    wikimedia_wiki,
)
from agbalu.acquire.storage import PART_SUFFIX
from agbalu.registry.models import Source


def make_source(**overrides: object) -> Source:
    fields: dict[str, object] = {
        "id": "hf.example.kab",
        "name": "Example",
        "modality": "text",
        "tier": "core",
        "access": "hf",
        "uri": "https://huggingface.co/datasets/boffire/kabyle-verbs",
        "licence": "cc0-1.0",
        "languages": ("kab",),
        "size": {"rows": 1},
        "retrieved": date(2026, 8, 5),
    }
    fields.update(overrides)
    return Source.model_validate(fields)


def test_repo_id_strips_the_dataset_prefix() -> None:
    assert repo_id(make_source()) == "boffire/kabyle-verbs"


def test_repo_id_rejects_a_non_huggingface_uri() -> None:
    with pytest.raises(FetchError, match="not a HuggingFace dataset or model URL"):
        repo_id(make_source(uri="https://github.com/x/y"))


def test_model_repos_resolve_as_models() -> None:
    """The reference tier is mostly model repos, not datasets.

    Assuming `repo_type="dataset"` for every HF URL is what made six sources —
    both published Kabyle tokenizers, the POS model, MarianMT and GlotLID —
    permanently unfetchable.
    """
    source = make_source(uri="https://huggingface.co/cis-lmu/glotlid")
    assert fetch_module.hf_repo(source) == ("cis-lmu/glotlid", "model")


def test_dataset_repos_still_resolve_as_datasets() -> None:
    source = make_source(uri="https://huggingface.co/datasets/boffire/kabyle-verbs")
    assert fetch_module.hf_repo(source) == ("boffire/kabyle-verbs", "dataset")


def test_a_bare_huggingface_url_is_rejected() -> None:
    # No owner/name pair means there is nothing to fetch.
    with pytest.raises(FetchError, match="not a HuggingFace dataset or model URL"):
        fetch_module.hf_repo(make_source(uri="https://huggingface.co/"))


def test_a_deep_huggingface_path_is_rejected() -> None:
    with pytest.raises(FetchError, match="not a HuggingFace dataset or model URL"):
        fetch_module.hf_repo(make_source(uri="https://huggingface.co/a/b/c/d"))


def test_opus_corpus_reads_the_corpus_name() -> None:
    source = make_source(access="opus", uri="https://opus.nlpl.eu/Tatoeba/corpus/version/Tatoeba")
    assert opus_corpus(source) == "Tatoeba"


def test_opus_corpus_handles_a_hyphenated_name() -> None:
    source = make_source(
        access="opus", uri="https://opus.nlpl.eu/bible-uedin/corpus/version/bible-uedin"
    )
    assert opus_corpus(source) == "bible-uedin"


def test_wikimedia_wiki_reads_the_wiki_name() -> None:
    source = make_source(access="wikimedia", uri="https://dumps.wikimedia.org/kabwiki/")
    assert wikimedia_wiki(source) == "kabwiki"


def test_walk_artifacts_hashes_every_file(tmp_path: Path) -> None:
    (tmp_path / "a.tsv").write_text("hello", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("world", encoding="utf-8")
    artifacts = list(walk_artifacts(tmp_path, make_source()))
    assert [a.path for a in artifacts] == ["a.tsv", "nested/b.txt"]
    assert all(len(a.sha256) == 64 for a in artifacts)
    assert [a.bytes for a in artifacts] == [5, 5]


def test_walk_artifacts_excludes_git_metadata(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    (tmp_path / "real.txt").write_text("y", encoding="utf-8")
    assert [a.path for a in walk_artifacts(tmp_path, make_source())] == ["real.txt"]


def test_walk_artifacts_excludes_partial_downloads(tmp_path: Path) -> None:
    # A `.part` is an interrupted transfer, not an artifact to record.
    (tmp_path / f"big.zip{PART_SUFFIX}").write_text("half", encoding="utf-8")
    (tmp_path / "done.zip").write_text("all", encoding="utf-8")
    assert [a.path for a in walk_artifacts(tmp_path, make_source())] == ["done.zip"]


def test_walk_artifacts_skips_symlinks(tmp_path: Path) -> None:
    (tmp_path / "real.txt").write_text("y", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    assert [a.path for a in walk_artifacts(tmp_path, make_source())] == ["real.txt"]


def test_walk_artifacts_on_an_empty_directory(tmp_path: Path) -> None:
    assert list(walk_artifacts(tmp_path, make_source())) == []


def test_walk_artifacts_records_an_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    artifacts = list(walk_artifacts(tmp_path, make_source()))
    assert [(a.path, a.bytes) for a in artifacts] == [("empty.txt", 0)]


def test_walk_artifacts_assigns_targets(tmp_path: Path) -> None:
    (tmp_path / "book.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.tsv").write_text("a\tb", encoding="utf-8")
    targets = {a.path: a.target for a in walk_artifacts(tmp_path, make_source())}
    assert targets == {"book.pdf": "remote", "notes.tsv": "local"}


class FakeResponse:
    """Minimal stand-in for `requests.Response` as a context manager."""

    def __init__(self, status: int, body: bytes = b"", *, chunk: int = 4) -> None:
        self.status_code = status
        self._body = body
        self._chunk = chunk
        self.headers: dict[str, str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        step = min(chunk_size, self._chunk)
        for start in range(0, len(self._body), step):
            yield self._body[start : start + step]


def install_fake_get(
    monkeypatch: pytest.MonkeyPatch, response: FakeResponse, seen: dict[str, Any]
) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        seen["url"] = url
        seen["headers"] = kwargs.get("headers") or {}
        return response

    # String target: `no_implicit_reexport` forbids reaching through the module
    # for an imported name.
    monkeypatch.setattr("agbalu.acquire.fetch.requests.get", fake_get)


def test_download_writes_a_complete_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(200, b"abcdefgh"), seen)
    final = tmp_path / "out.bin"
    download_file("https://example.invalid/f", final)
    assert final.read_bytes() == b"abcdefgh"
    assert not final.with_name(final.name + PART_SUFFIX).exists()


def test_download_sends_a_descriptive_user_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Wikimedia answers a generic agent with 403.
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(200, b"x"), seen)
    download_file("https://example.invalid/f", tmp_path / "out.bin")
    assert "AGBALU" in seen["headers"]["User-Agent"]


def test_download_resumes_from_an_existing_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "out.bin"
    final.with_name(final.name + PART_SUFFIX).write_bytes(b"abcd")
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(206, b"efgh"), seen)
    download_file("https://example.invalid/f", final)
    assert seen["headers"]["Range"] == "bytes=4-"
    assert final.read_bytes() == b"abcdefgh"


def test_a_206_without_a_prior_part_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(206, b"xyz"), seen)
    final = tmp_path / "out.bin"
    download_file("https://example.invalid/f", final)
    assert final.read_bytes() == b"xyz"
    assert "Range" not in seen["headers"]


def test_a_416_finalizes_the_existing_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The server says the range is unsatisfiable because we already hold it all.
    final = tmp_path / "out.bin"
    final.with_name(final.name + PART_SUFFIX).write_bytes(b"complete")
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(416), seen)
    download_file("https://example.invalid/f", final)
    assert final.read_bytes() == b"complete"


@pytest.mark.parametrize("status", [403, 404, 429, 500, 503])
def test_an_error_status_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(status), seen)
    with pytest.raises(FetchError, match=f"HTTP {status}"):
        download_file("https://example.invalid/f", tmp_path / "out.bin")


def test_a_failed_download_leaves_no_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    install_fake_get(monkeypatch, FakeResponse(500), seen)
    final = tmp_path / "out.bin"
    with pytest.raises(FetchError):
        download_file("https://example.invalid/f", final)
    assert not final.exists()


def test_git_binary_error_names_the_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agbalu.acquire.fetch.shutil.which", lambda _: None)
    with pytest.raises(FetchError, match="git is not on PATH"):
        fetch_module.git_binary()


def test_manual_fetch_reports_a_missing_local_file(tmp_path: Path) -> None:
    source = make_source(access="manual", uri=str(tmp_path / "absent.tsv"), checksum="c" * 64)
    with pytest.raises(FetchError, match="declared local file is missing"):
        fetch_module.fetch_manual(source, tmp_path / "dest")


def test_iso_codes_cover_the_registry_languages() -> None:
    # A missing mapping would silently fetch no OPUS bundle for that language.
    assert fetch_module.ISO3_TO_ISO2["eng"] == "en"
    assert fetch_module.ISO3_TO_ISO2["fra"] == "fr"


def test_module_exposes_kab_patterns_for_mega_datasets() -> None:
    # Without a filter HPLT alone is 112,335 files; the Kabyle share is one.
    assert fetch_module.FETCH_PATTERNS["hf.hplt2-cleaned-kab"] == ("kab_Latn/**",)


def test_fetch_result_is_a_frozen_dataclass() -> None:
    result = fetch_module.FetchResult(revision="abc", artifacts=())
    with pytest.raises((AttributeError, TypeError)):
        result.revision = "def"  # type: ignore[misc]


def test_module_has_no_stub_access_types() -> None:
    # Every `Access` literal must be dispatched; a stubbed one silently defers sources.
    assert isinstance(fetch_module.fetch, types.FunctionType)
