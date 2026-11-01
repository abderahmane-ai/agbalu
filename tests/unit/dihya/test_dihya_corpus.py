"""Verify candidate accounting and split isolation with positive controls."""

from __future__ import annotations

import pytest

from agbalu.dihya.audit import verify_conservation
from agbalu.dihya.corpus import RawRow, evaluation_key, select_rows, split_rows, text_key
from agbalu.dihya.taxonomy import COMPOSITE_CLASSES


def test_evaluation_exclusion_matches_punctuation_and_aligned_siblings() -> None:
    rows = [
        RawRow("Reserved evaluation text!", "kab_Latn", "web", "test", "Latn", family="shared"),
        RawRow("ⴰⵣⵓⵍ ⴰⵣⵓⵍ ⴰⵣⵓⵍ", "kab_Tfng", "web", "test", "Tfng", family="shared"),
    ]
    kept, rejected, _ = select_rows(rows, 10, 42, {evaluation_key("Reserved evaluation text.")})
    assert kept == []
    assert {reason for _, reason in rejected} == {
        "external_evaluation_overlap",
        "external_evaluation_family_overlap",
    }


def test_conflicting_labels_are_quarantined_in_both_directions() -> None:
    rows = [
        RawRow("Azul fell awen", label, "web", "test", "Latn") for label in ("kab_Latn", "shy_Latn")
    ]
    kept, rejected, counts = select_rows(rows, 10, 42, set())
    assert kept == []
    assert [reason for _, reason in rejected] == ["conflicting_labels"] * 2
    assert counts == {"candidate_rows": 2, "accepted_rows": 0, "rejected_rows": 2}


def test_normalized_duplicates_script_errors_and_external_exclusions_are_counted() -> None:
    rows = [
        RawRow(text, "kab_Latn", "web", "test", "Latn")
        for text in (
            "Amék tellid",
            "Amék tellid",
            "Azul ⴰⵣⵓⵍ",
            "Reserved evaluation text",
        )
    ]
    kept, rejected, counts = select_rows(rows, 10, 42, {text_key("Reserved evaluation text")})
    assert [row.text for row in kept] == ["Amék tellid"]
    assert len(rejected) == 3
    assert {reason for _, reason in rejected} >= {"duplicate_text", "external_evaluation_overlap"}
    verify_conservation(counts["candidate_rows"], counts["accepted_rows"], counts["rejected_rows"])


def test_conservation_positive_control_rejects_synthetic_loss() -> None:
    with pytest.raises(ValueError, match="input 5 != kept 2"):
        verify_conservation(5, 2, 2)


def test_reservoir_cap_is_reproducible_and_sees_the_whole_stream() -> None:
    rows = [
        RawRow(f"Azul fell awen {index}", "kab_Latn", "web", "test", "Latn") for index in range(100)
    ]
    first = select_rows(rows, 10, 42, set())
    assert first == select_rows(rows, 10, 42, set())
    assert len(first[0]) == 10
    assert len(first[1]) == 90
    assert any(int(row.text.split()[-1]) > 50 for row in first[0])


def test_split_stratifies_every_class_and_keeps_families_together() -> None:
    rows: list[RawRow] = []
    for label in COMPOSITE_CLASSES:
        script = label.rsplit("_", 1)[1]
        text = {"Latn": "Azul", "Arabizi": "salam", "Arab": "سلام", "Tfng": "ⴰⵣⵓⵍ", "Hani": "你好"}[
            script
        ]
        rows.extend(
            RawRow(
                f"{text} {index}",
                label,
                "web",
                "test",
                script,
                family=f"{label.split('_')[0]}:{index}",
            )
            for index in range(40)
        )
    splits = split_rows(rows, 42)
    ownership: dict[str, str] = {}
    for name, pool in splits.items():
        assert {row.label for row in pool} == set(COMPOSITE_CLASSES)
        for row in pool:
            assert ownership.setdefault(row.family, name) == name
    assert sum(map(len, splits.values())) == len(rows)
