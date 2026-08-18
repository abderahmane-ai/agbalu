"""Typed records produced by acquisition.

Provenance is per *artifact*, not per source: a speech source splits into
transcripts that belong on the workstation and audio that belongs on a remote
volume, so a single storage target per source cannot express reality.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from agbalu.registry.models import Sha256, SourceId

StorageTarget = Literal["local", "remote", "none"]
"""Where an artifact's bytes live.

`local` is the working tree under `data/raw/`; `remote` is a Modal Volume, read
only by the phase that needs it; `none` means the source is catalogued but never
ingested (`reference` and `excluded` tiers).
"""

ArtifactKind = Literal["text", "audio", "document", "opaque"]
"""What an artifact is, as far as placement is concerned."""

DeferralReason = Literal[
    "remote-target", "gated", "unresolved-licence", "manual-step", "unavailable"
]
"""Why a source was not landed locally. Every value is auditable after the fact."""

RemovalReason = Literal["redundant", "superseded", "space", "licence"]
"""Why landed bytes were deleted. `redundant` means another landed file has the same
digest; `superseded` means an upstream version replaced it."""


def _relative_posix_path(value: str) -> str:
    """Reject anything that is not a plain relative path inside a source directory.

    A validator rather than a `pattern`: pydantic v2 compiles patterns with the
    Rust `regex` crate, which has no look-around, so the traversal check cannot
    be expressed as a regex.
    """
    if value.startswith("/"):
        msg = f"artifact path must be relative: {value!r}"
        raise ValueError(msg)
    if "\x00" in value:
        msg = f"artifact path contains a null byte: {value!r}"
        raise ValueError(msg)
    if ".." in PurePosixPath(value).parts:
        msg = f"artifact path escapes its source root: {value!r}"
        raise ValueError(msg)
    return value


ArtifactPath = Annotated[
    str, Field(min_length=1, max_length=1024), AfterValidator(_relative_posix_path)
]


class Artifact(BaseModel):
    """One file belonging to a source, as landed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: ArtifactPath
    bytes: int = Field(ge=0)
    sha256: Sha256
    kind: ArtifactKind
    target: StorageTarget


class ManifestEntry(BaseModel):
    """An artifact plus the acquisition that produced it. One JSONL row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SourceId
    path: ArtifactPath
    bytes: int = Field(ge=0)
    sha256: Sha256
    kind: ArtifactKind
    target: StorageTarget
    uri: str = Field(min_length=1)
    fetched_at: datetime
    revision: str | None = Field(default=None, max_length=200)
    """Commit sha, HF revision, or dump date — whatever pins this fetch in time."""

    @model_validator(mode="after")
    def _require_timezone(self) -> Self:
        # A naive timestamp is unorderable against one from another machine.
        if self.fetched_at.tzinfo is None:
            msg = f"fetched_at must be timezone-aware: {self.source_id}/{self.path}"
            raise ValueError(msg)
        return self


class Deferral(BaseModel):
    """A source deliberately not landed. The exit criterion admits these; silence it does not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SourceId
    reason: DeferralReason
    detail: str = Field(min_length=1, max_length=500)
    recorded_at: datetime

    @model_validator(mode="after")
    def _require_timezone(self) -> Self:
        if self.recorded_at.tzinfo is None:
            msg = f"recorded_at must be timezone-aware: {self.source_id}"
            raise ValueError(msg)
        return self


class Removal(BaseModel):
    """Landed bytes deliberately deleted. `data/raw/` is immutable, so this is a new row
    rather than an edit, and the digest is kept so a re-fetch can be checked against what
    was removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SourceId
    path: ArtifactPath
    sha256: Sha256
    reason: RemovalReason
    detail: str = Field(min_length=1, max_length=500)
    removed_at: datetime

    @model_validator(mode="after")
    def _require_timezone(self) -> Self:
        if self.removed_at.tzinfo is None:
            msg = f"removed_at must be timezone-aware: {self.source_id}/{self.path}"
            raise ValueError(msg)
        return self


class VerifyFinding(BaseModel):
    """A discrepancy between the manifest, the disk, and the registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SourceId
    path: ArtifactPath | None
    kind: Literal[
        "missing", "checksum-mismatch", "size-drift", "unrecorded", "unfetched", "resurrected"
    ]
    detail: str = Field(min_length=1, max_length=500)
