"""Verify speech release selection and corrupted artifact rejection."""

import json
from pathlib import Path

import pytest
from tools.stage_speech import link_weight, select_formats, verify_digest, verify_existing


def test_default_release_contains_only_qualified_formats() -> None:
    assert select_formats(None) == ("bf16", "q8_0", "q4_k")


@pytest.mark.parametrize("value", ["bf16", "q8_0", "q4_k"])
def test_individual_format_selection(value: str) -> None:
    assert select_formats(value) == (value,)


@pytest.mark.parametrize("value", ["q4_0", "nf4", "int8", "", "../model"])
def test_unqualified_format_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="unsupported release format"):
        select_formats(value)


@pytest.mark.integration
def test_checksum_positive_control_reports_corrupted_path(tmp_path: Path) -> None:
    path = tmp_path / "model"
    path.write_bytes(b"abc")
    verify_digest(path, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="checksum mismatch: path="):
        verify_digest(path, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


@pytest.mark.integration
def test_release_weights_share_storage_without_copying(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"weights")
    staged = tmp_path / "staged"
    link_weight(original, staged)
    assert staged.stat().st_ino == original.stat().st_ino
    original.unlink()
    assert staged.read_bytes() == b"weights"


@pytest.mark.integration
def test_existing_release_rejects_changed_source_manifest(tmp_path: Path) -> None:
    (tmp_path / "export.stats.json").write_text(json.dumps({"files": {"README.md": "old"}}))
    with pytest.raises(FileExistsError, match="preserve it before restaging"):
        verify_existing(tmp_path, {"README.md": "new"})


@pytest.mark.integration
def test_existing_release_rejects_missing_artifact(tmp_path: Path) -> None:
    files = {"model.safetensors": "a" * 64}
    (tmp_path / "export.stats.json").write_text(json.dumps({"files": files}))
    with pytest.raises(FileNotFoundError):
        verify_existing(tmp_path, files)
