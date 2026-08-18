"""Decide where an artifact's bytes belong.

The policy follows the phase that reads the data. Phases 2-4 and 6 iterate over
text on the workstation, so text lands locally. Phase 3.2 OCRs PDFs and Phase 5
processes audio, both on GPU, so those land on the remote volume and are never
copied down.

Tier controls release membership, not whether bytes exist on disk. A `reference`
source is consulted and never ingested, but consulting it requires having it, so
it lands like anything else. Only `excluded` resolves to `none`.

A pure function of (tier, modality, path, size), with no per-source override table.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final

from agbalu.acquire.models import ArtifactKind, StorageTarget
from agbalu.registry.models import Modality, Tier

AUDIO_SUFFIXES: Final = frozenset(
    {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff"}
)

DOCUMENT_SUFFIXES: Final = frozenset({".pdf", ".djvu", ".epub"})

TEXT_SUFFIXES: Final = frozenset(
    {
        ".tsv", ".csv", ".txt", ".json", ".jsonl", ".ndjson", ".yaml", ".yml",
        ".md", ".conllu", ".xml", ".dic", ".aff", ".po", ".vtt", ".srt", ".sql",
    }
)  # fmt: skip

LOCAL_MAX_BYTES: Final = 2 * 1024**3
"""Above this a single artifact goes remote regardless of kind.

The workstation had ~32 GiB free when this policy was set. Nothing the iterative
phases do holds a >2 GiB artifact in memory, so a file that large is batch input,
and batch input belongs where the batch runs.
"""


REMOTE_SOURCES: Final = frozenset(
    {
        # Scanned books. Phase 3.2 OCRs these on GPU; they are never read locally.
        "hf.boffire.adlis-pdfs",
        "hf.boffire.ayamun-pdfs",
        # 8.58 GB and 157,307,939 rows covering thousands of languages, with no
        # kab-named file to filter on. Kabyle is a sliver of it, so a bulk pull
        # spends 8.5 GB of a 26 GB budget to reach a word list. Phase 6 extracts
        # the Kabyle rows server-side instead.
        "hf.panlex-meanings-kab",
    }
)


def source_target(*, tier: Tier, modality: Modality, source_id: str) -> StorageTarget:
    """Coarse target for a whole source, decided *before* fetching.

    `classify` is exact but needs an artifact's name and size, which requires
    downloading it first. This predicate answers the question that avoids the
    download: is this source's payload something the workstation should hold?
    """
    if tier == "excluded":
        return "none"
    if modality == "speech" or source_id in REMOTE_SOURCES:
        return "remote"
    return "local"


def artifact_kind(path: str, modality: Modality) -> ArtifactKind:
    """Classify an artifact by extension, falling back to its source's modality."""
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in DOCUMENT_SUFFIXES:
        return "document"
    # An unrecognised extension inside a speech source is an audio container or a
    # shard of them — Common Voice ships `.tar.gz`, whose suffix is `.gz`.
    if modality == "speech":
        return "audio"
    return "opaque"


def classify(
    *,
    tier: Tier,
    modality: Modality,
    path: str,
    size_bytes: int,
) -> StorageTarget:
    """Return the storage target for one artifact."""
    if tier == "excluded":
        return "none"

    kind = artifact_kind(path, modality)
    if kind in ("audio", "document"):
        return "remote"
    if size_bytes > LOCAL_MAX_BYTES:
        return "remote"
    return "local"
