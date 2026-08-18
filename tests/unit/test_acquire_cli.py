from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agbalu.acquire.cli import build_parser, main
from agbalu.acquire.manifest import Manifest
from agbalu.acquire.models import ManifestEntry
from agbalu.acquire.storage import sha256_file

REGISTRY = """
version: "1.0.0"
surveyed: 2026-08-05
sources:
  - id: hf.text.kab
    name: Text
    modality: text
    tier: core
    access: hf
    uri: https://huggingface.co/datasets/x/y
    licence: cc0-1.0
    languages: [kab]
    size: { bytes: 1000 }
    retrieved: 2026-08-05
  - id: hf.speech.kab
    name: Speech
    modality: speech
    tier: core
    access: hf
    uri: https://huggingface.co/datasets/x/z
    licence: cc0-1.0
    languages: [kab]
    size: { bytes: 2000000000 }
    retrieved: 2026-08-05
  - id: hf.bench.kab
    name: Benchmark
    modality: text
    tier: reference
    access: hf
    uri: https://huggingface.co/datasets/x/b
    licence: cc0-1.0
    languages: [kab]
    size: { rows: 1 }
    retrieved: 2026-08-05
"""


@pytest.fixture
def registry_file(tmp_path: Path) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(REGISTRY, encoding="utf-8")
    return path


def test_shared_flags_may_follow_the_subcommand() -> None:
    args = build_parser().parse_args(["plan", "--tier", "core"])
    assert args.command == "plan"
    assert args.tier == "core"


def test_id_flag_accumulates() -> None:
    args = build_parser().parse_args(["fetch", "--id", "a", "--id", "b"])
    assert args.id == ["a", "b"]


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_an_unknown_tier_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan", "--tier", "nonsense"])


def test_plan_separates_local_from_remote(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["plan", "--registry", str(registry_file), "--root", str(tmp_path / "raw")])
    out = capsys.readouterr().out
    assert code == 0
    assert "local   core           hf        hf.text.kab" in out
    assert "remote  core           hf        hf.speech.kab" in out
    # `reference` lands too — it is consulted, never ingested into a release.
    assert "local   reference      hf        hf.bench.kab" in out


def test_plan_totals_declared_bytes(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["plan", "--registry", str(registry_file), "--root", str(tmp_path / "raw")])
    out = capsys.readouterr().out
    assert "remote    1 sources      2.00 GB declared" in out


def test_plan_honours_the_id_filter(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "plan",
            "--registry",
            str(registry_file),
            "--root",
            str(tmp_path / "raw"),
            "--id",
            "hf.text.kab",
        ]
    )
    out = capsys.readouterr().out
    assert "hf.text.kab" in out
    assert "hf.speech.kab" not in out


def test_a_missing_registry_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["plan", "--registry", str(tmp_path / "absent.yaml")])
    assert code == 1
    assert "registry not found" in capsys.readouterr().err


def test_verify_reports_every_unfetched_source_that_should_have_bytes(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["verify", "--registry", str(registry_file), "--root", str(tmp_path / "raw")])
    out = capsys.readouterr().out
    assert code == 1
    assert "unfetched" in out
    assert "hf.text.kab" in out
    # Tier governs release membership, not whether bytes exist: a `reference` source
    # is consulted, and consulting it requires having it. Scoping this check to
    # `core` is what let 14 sources go missing unnoticed in Phase 1.
    assert "hf.bench.kab" in out


def landed(root: Path, source_id: str, path: str, payload: bytes = b"bytes") -> None:
    """A source with one artifact on disk and its manifest row, as `fetch` leaves it."""
    artifact = root / source_id / path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    digest, size = sha256_file(artifact)
    Manifest(root).append(
        ManifestEntry(
            source_id=source_id,
            path=path,
            bytes=size,
            sha256=digest,
            kind="text",
            target="local",
            uri="https://example.invalid/kab",
            fetched_at=datetime.now(UTC),
        )
    )


def remove_argv(registry_file: Path, root: Path, path: str) -> list[str]:
    return [
        "remove",
        "--registry",
        str(registry_file),
        "--root",
        str(root),
        "--source",
        "hf.text.kab",
        "--path",
        path,
        "--reason",
        "redundant",
        "--detail",
        "byte-identical to another landed artifact",
    ]


def test_remove_refuses_an_artifact_still_on_disk(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The record must never claim a file is gone while something can still read it."""
    root = tmp_path / "raw"
    landed(root, "hf.text.kab", "train.parquet")
    code = main(remove_argv(registry_file, root, "train.parquet"))
    assert code == 1
    assert "still on disk" in capsys.readouterr().err
    assert Manifest(root).removed_artifacts() == frozenset()


def test_remove_refuses_a_path_not_in_the_manifest(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "raw"
    landed(root, "hf.text.kab", "train.parquet")
    code = main(remove_argv(registry_file, root, "absent.parquet"))
    assert code == 1
    assert "not in the manifest" in capsys.readouterr().err


def test_remove_refuses_an_unregistered_source(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "raw"
    argv = remove_argv(registry_file, root, "train.parquet")
    argv[argv.index("hf.text.kab")] = "hf.invented.kab"
    code = main(argv)
    assert code == 1
    assert "not in the registry" in capsys.readouterr().err


def test_remove_records_a_deleted_artifact(registry_file: Path, tmp_path: Path) -> None:
    root = tmp_path / "raw"
    landed(root, "hf.text.kab", "train.parquet")
    (root / "hf.text.kab" / "train.parquet").unlink()
    assert main(remove_argv(registry_file, root, "train.parquet")) == 0
    assert Manifest(root).removed_artifacts() == frozenset({("hf.text.kab", "train.parquet")})


def test_verify_does_not_report_a_recorded_removal_as_missing(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "raw"
    landed(root, "hf.text.kab", "train.parquet")
    (root / "hf.text.kab" / "train.parquet").unlink()
    main(remove_argv(registry_file, root, "train.parquet"))
    capsys.readouterr()

    main(["verify", "--registry", str(registry_file), "--root", str(root)])
    out = capsys.readouterr().out
    assert "missing" not in out
    assert "1 removed" in out


def test_verify_reports_a_removed_artifact_that_came_back(
    registry_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A removal row plus the bytes on disk is a stale record, not a clean state."""
    root = tmp_path / "raw"
    landed(root, "hf.text.kab", "train.parquet")
    (root / "hf.text.kab" / "train.parquet").unlink()
    main(remove_argv(registry_file, root, "train.parquet"))
    (root / "hf.text.kab" / "train.parquet").write_bytes(b"bytes")
    capsys.readouterr()

    code = main(["verify", "--registry", str(registry_file), "--root", str(root)])
    out = capsys.readouterr().out
    assert code == 1
    assert "resurrected" in out
