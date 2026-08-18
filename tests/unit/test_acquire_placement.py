from __future__ import annotations

import pytest

from agbalu.acquire.placement import LOCAL_MAX_BYTES, artifact_kind, classify, source_target


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("train.parquet", "opaque"),
        ("validated.tsv", "text"),
        ("notes.TXT", "text"),
        ("kab.conllu", "text"),
        ("hunspell.dic", "text"),
        ("clip.mp3", "audio"),
        ("clip.FLAC", "audio"),
        ("book.pdf", "document"),
        ("archive.tar.gz", "opaque"),
        ("no_extension", "opaque"),
    ],
)
def test_artifact_kind_reads_the_extension(path: str, expected: str) -> None:
    assert artifact_kind(path, "text") == expected


def test_unknown_extension_in_a_speech_source_is_audio() -> None:
    # Common Voice ships `.tar.gz`, whose suffix is `.gz`, not `.mp3`.
    assert artifact_kind("kab.tar.gz", "speech") == "audio"


def test_a_transcript_in_a_speech_source_is_still_text() -> None:
    # Extension wins over modality, so Common Voice's TSVs stay on the workstation.
    assert artifact_kind("validated.tsv", "speech") == "text"


def test_reference_tier_lands_so_it_can_be_consulted() -> None:
    # `reference` means "never enters a release", not "never on disk" — the
    # orthography spec and Dallet must be readable to be consulted at all.
    assert classify(tier="reference", modality="text", path="a.tsv", size_bytes=1) == "local"


def test_excluded_tier_is_never_ingested() -> None:
    assert classify(tier="excluded", modality="text", path="a.tsv", size_bytes=1) == "none"


def test_small_text_lands_locally() -> None:
    assert classify(tier="core", modality="text", path="a.tsv", size_bytes=1024) == "local"


def test_audio_goes_remote_however_small() -> None:
    assert classify(tier="core", modality="speech", path="a.mp3", size_bytes=1) == "remote"


def test_pdfs_go_remote_however_small() -> None:
    assert classify(tier="core", modality="text", path="a.pdf", size_bytes=1) == "remote"


def test_a_speech_transcript_lands_locally() -> None:
    assert classify(tier="core", modality="speech", path="validated.tsv", size_bytes=10) == "local"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (LOCAL_MAX_BYTES - 1, "local"),
        (LOCAL_MAX_BYTES, "local"),
        (LOCAL_MAX_BYTES + 1, "remote"),
    ],
)
def test_size_threshold_is_inclusive(size: int, expected: str) -> None:
    assert classify(tier="core", modality="text", path="a.tsv", size_bytes=size) == expected


def test_a_zero_byte_artifact_lands_locally() -> None:
    assert classify(tier="core", modality="text", path="empty.txt", size_bytes=0) == "local"


def test_source_target_sends_speech_remote() -> None:
    assert source_target(tier="core", modality="speech", source_id="cv.x") == "remote"


def test_source_target_sends_pdf_archives_remote() -> None:
    assert (
        source_target(tier="core", modality="text", source_id="hf.boffire.adlis-pdfs") == "remote"
    )


def test_source_target_keeps_text_local() -> None:
    assert source_target(tier="core", modality="text", source_id="hf.fineweb2-kab") == "local"


def test_source_target_lands_reference_tier() -> None:
    assert source_target(tier="reference", modality="text", source_id="anything") == "local"


def test_only_excluded_resolves_to_none() -> None:
    assert source_target(tier="excluded", modality="text", source_id="anything") == "none"
    assert classify(tier="excluded", modality="text", path="a.tsv", size_bytes=1) == "none"


def test_source_target_agrees_with_classify_for_text() -> None:
    # The coarse pre-fetch predicate must not contradict the exact one.
    coarse = source_target(tier="core", modality="text", source_id="hf.fineweb2-kab")
    exact = classify(tier="core", modality="text", path="data.parquet", size_bytes=1000)
    assert coarse == exact
