from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from agbalu.acquire.models import Artifact, Deferral, ManifestEntry

DIGEST = "a" * 64


def artifact(**overrides: object) -> Artifact:
    fields: dict[str, object] = {
        "path": "data/train.parquet",
        "bytes": 10,
        "sha256": DIGEST,
        "kind": "text",
        "target": "local",
    }
    fields.update(overrides)
    return Artifact.model_validate(fields)


def test_accepts_a_plain_relative_path() -> None:
    assert artifact().path == "data/train.parquet"


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../../.ssh/authorized_keys",
        "data/../../escape.txt",
        "..",
        "a/../../b",
    ],
)
def test_rejects_paths_that_escape_the_source_root(path: str) -> None:
    # Archive members and provider filenames are untrusted input.
    with pytest.raises(ValidationError):
        artifact(path=path)


def test_rejects_a_null_byte_in_the_path() -> None:
    with pytest.raises(ValidationError):
        artifact(path="data/train\x00.parquet")


def test_rejects_an_empty_path() -> None:
    with pytest.raises(ValidationError):
        artifact(path="")


def test_allows_a_dotdot_inside_a_filename() -> None:
    # `..` is only traversal as a whole path segment.
    assert artifact(path="weird..name.txt").path == "weird..name.txt"


def test_preserves_kabyle_orthography_in_paths() -> None:
    assert artifact(path="tazwart/aɣbalu-ɛ-ḥ-ḍ.txt").path.endswith("ɛ-ḥ-ḍ.txt")


def test_rejects_a_short_digest() -> None:
    with pytest.raises(ValidationError):
        artifact(sha256="abc")


def test_rejects_an_uppercase_digest() -> None:
    with pytest.raises(ValidationError):
        artifact(sha256="A" * 64)


def test_rejects_negative_bytes() -> None:
    with pytest.raises(ValidationError):
        artifact(bytes=-1)


def test_allows_a_zero_byte_artifact() -> None:
    # Empty files exist upstream; they must be recorded, not silently dropped.
    assert artifact(bytes=0).bytes == 0


def test_rejects_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        artifact(colour="red")


def test_artifact_is_frozen() -> None:
    with pytest.raises(ValidationError):
        artifact().path = "other.txt"


def entry(**overrides: object) -> ManifestEntry:
    fields: dict[str, object] = {
        "source_id": "hf.example.kab",
        "path": "train.parquet",
        "bytes": 1,
        "sha256": DIGEST,
        "kind": "text",
        "target": "local",
        "uri": "https://example.invalid/kab",
        "fetched_at": datetime.now(UTC),
    }
    fields.update(overrides)
    return ManifestEntry.model_validate(fields)


def test_manifest_entry_requires_a_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        entry(fetched_at=datetime(2026, 8, 5, 12, 0, 0))


def test_manifest_entry_accepts_a_non_utc_timezone() -> None:
    stamp = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone(timedelta(hours=1)))
    assert entry(fetched_at=stamp).fetched_at.tzinfo is not None


def test_manifest_entry_revision_defaults_to_none() -> None:
    assert entry().revision is None


def test_manifest_entry_round_trips_through_json() -> None:
    original = entry(revision="af9c13333eb9")
    assert ManifestEntry.model_validate_json(original.model_dump_json()) == original


def test_deferral_requires_a_known_reason() -> None:
    with pytest.raises(ValidationError):
        Deferral.model_validate(
            {
                "source_id": "hf.example.kab",
                "reason": "because-i-said-so",
                "detail": "no",
                "recorded_at": datetime.now(UTC),
            }
        )


def test_deferral_requires_a_detail() -> None:
    with pytest.raises(ValidationError):
        Deferral.model_validate(
            {
                "source_id": "hf.example.kab",
                "reason": "gated",
                "detail": "",
                "recorded_at": datetime.now(UTC),
            }
        )
