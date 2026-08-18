"""Document sidecars are the only provenance that survives a clean checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tools.stamp_documents import DOCUMENTS, MEASURED, REQUIRED, measure, sidecars, stamp


def document(tmp_path: Path, text: str, **fields: object) -> Path:
    (tmp_path / "book.txt").write_text(text, encoding="utf-8")
    sidecar = tmp_path / "book.meta.json"
    sidecar.write_text(json.dumps(fields), encoding="utf-8")
    return sidecar


def test_the_hash_is_of_the_bytes_on_disk() -> None:
    path = Path(__file__)
    assert measure(path)["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_paragraphs_are_blank_line_separated_and_empty_ones_do_not_count(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("one\n\n\n\ntwo\n\nthree\n", encoding="utf-8")
    assert measure(tmp_path / "a.txt")["paragraphs"] == 3


def test_characters_and_bytes_differ_where_the_text_is_not_ascii(tmp_path: Path) -> None:
    """`ɣ` is one character and two bytes. Reporting one as the other is a factual error
    in a card that quotes a corpus size."""
    (tmp_path / "a.txt").write_text("aɣ", encoding="utf-8")
    stats = measure(tmp_path / "a.txt")
    assert stats["characters"] == 2
    assert stats["bytes"] == 3


def test_a_stale_hash_is_reported_rather_than_trusted(tmp_path: Path) -> None:
    """The failure this exists for: a document cleaned in place after its sidecar was
    written keeps a hash that no longer matches, which reads as verified."""
    sidecar = document(tmp_path, "azul\n", title="A", sha256="0" * 64)
    assert "sha256" in stamp(sidecar, write=False).stale


def test_writing_corrects_every_measured_field(tmp_path: Path) -> None:
    sidecar = document(tmp_path, "azul d ameqqran\n", title="A", bytes=1, words=99)
    assert stamp(sidecar, write=True).stale == ()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["words"] == 3
    assert payload["bytes"] == len("azul d ameqqran\n")


def test_provenance_is_reported_missing_and_never_invented(tmp_path: Path) -> None:
    """A source, a licence and a retrieval date cannot be derived from the bytes. A tool
    that filled them in would be manufacturing provenance, which is the one thing §2.1
    rule 1 forbids."""
    sidecar = document(tmp_path, "azul\n", title="A", language="kab")
    report = stamp(sidecar, write=True)
    assert set(report.missing) == set(REQUIRED) - {"title", "language"}
    assert set(json.loads(sidecar.read_text(encoding="utf-8"))) == {
        "title",
        "language",
        *MEASURED,
    }


def test_a_sidecar_whose_bytes_are_absent_is_reported_not_skipped(tmp_path: Path) -> None:
    """The bytes are git-ignored, so this is the normal state on a fresh checkout — and it
    must be visible rather than read as a pass."""
    sidecar = tmp_path / "book.meta.json"
    sidecar.write_text(json.dumps({"title": "A"}), encoding="utf-8")
    report = stamp(sidecar, write=False)
    assert not report.present
    assert not report.ok


@pytest.mark.integration
def test_every_document_in_the_tree_carries_a_licence_and_a_title() -> None:
    """Provenance or it does not enter — §2.1 rule 1. `source` and `reproduce` are checked
    by `python3 -m tools.stamp_documents`, which currently reports three gaps that cannot be
    closed without knowing where those documents came from."""
    found = sidecars()
    assert found, "no documents staged"
    for sidecar in found:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        for field in ("title", "language", "licence"):
            assert payload.get(field), f"{sidecar.relative_to(DOCUMENTS)} has no {field}"


@pytest.mark.integration
def test_no_document_sidecar_records_a_hash_that_does_not_match_its_bytes() -> None:
    for sidecar in sidecars():
        report = stamp(sidecar, write=False)
        assert not report.stale, f"{sidecar.relative_to(DOCUMENTS)}: {list(report.stale)}"
