from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.bench.audit import audit, describe, token_divergence
from agbalu.bench.cli import main
from agbalu.bench.flores import FloresError, Sentence, read_split, revisions

CLEAN = "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a."
CORRUPT = "Tura nesεa tiɣeṛdayin n 4 n wagguren di leεmeṛ-nsent."
REPAIRED = "Tura nesɛa tiɣeṛdayin n 4 n wagguren di leɛmeṛ-nsent."


def sentence(text: str, ident: int = 1, split: str = "dev", updated: str = "1.0") -> Sentence:
    return Sentence(
        id=ident,
        text=text,
        split=split,  # type: ignore[arg-type]
        domain="wikinews",
        topic="News",
        url="",
        last_updated=updated,
        iso_639_3="kab",
        iso_15924="Latn",
    )


def test_greek_epsilon_is_found_and_repaired() -> None:
    report = audit([sentence(CORRUPT)], "kab_Latn")
    assert report.changed == 1
    assert report.diffs[0].corrected == REPAIRED
    assert report.by_codepoint["ε"] == 2


def test_clean_sentences_are_left_alone() -> None:
    report = audit([sentence(CLEAN)], "kab_Latn")
    assert report.changed == 0
    assert report.diffs == []
    assert report.changed_rate == 0.0
    assert report.mean_token_divergence == 0.0


def test_rates_are_over_all_sentences_and_divergence_over_changed_only() -> None:
    """Oktem et al. report divergence among corrected items; ours must match."""
    report = audit([sentence(CORRUPT, 1), *[sentence(CLEAN, i) for i in range(2, 5)]], "kab_Latn")
    assert report.sentences == 4
    assert report.changed == 1
    assert report.changed_rate == pytest.approx(0.25)
    assert report.mean_token_divergence == pytest.approx(report.diffs[0].token_divergence)
    assert report.mean_token_divergence > 0.0


def test_revisions_expose_an_unrevised_language() -> None:
    report = audit([sentence(CLEAN, 1), sentence(CLEAN, 2)], "kab_Latn")
    assert report.revisions == {"1.0": 2}


def test_split_attribution() -> None:
    report = audit([sentence(CORRUPT, 1, "dev"), sentence(CORRUPT, 2, "devtest")], "kab_Latn")
    assert dict(report.by_split) == {"dev": 1, "devtest": 1}


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("a b c d", "a b c d", 0.0),
        ("a b c d", "a b c X", 0.25),
        ("a b c d", "X Y Z W", 1.0),
        ("a b", "a b c d", 0.5),
        ("", "", 0.0),
        ("", "a", 1.0),
    ],
)
def test_token_divergence(before: str, after: str, expected: float) -> None:
    assert token_divergence(before, after) == pytest.approx(expected)


def test_describe_names_the_codepoint() -> None:
    assert describe("ε") == "ε U+03B5 GREEK SMALL LETTER EPSILON"
    assert "U+0263" in describe("ɣ")


def test_empty_audit_does_not_divide_by_zero() -> None:
    report = audit([], "kab_Latn")
    assert report.changed_rate == 0.0
    assert report.mean_token_divergence == 0.0


def write_split(root: Path, split: str, rows: list[dict[str, object]]) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "kab_Latn.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_read_split_parses_the_release_fields(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "dev",
        [{"id": 3, "text": CLEAN, "last_updated": "1.0", "domain": "wikinews", "topic": "News"}],
    )
    got = read_split(tmp_path, "dev")
    assert got[0].id == 3
    assert got[0].text == CLEAN
    assert got[0].last_updated == "1.0"
    assert revisions(got) == {"1.0": 1}


def test_a_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FloresError, match="not found"):
        read_split(tmp_path, "dev")


def test_an_empty_split_raises(tmp_path: Path) -> None:
    write_split(tmp_path, "dev", [])
    with pytest.raises(FloresError, match="empty"):
        read_split(tmp_path, "dev")


def test_blank_lines_in_a_split_are_skipped(tmp_path: Path) -> None:
    directory = tmp_path / "dev"
    directory.mkdir(parents=True)
    (directory / "kab_Latn.jsonl").write_text(
        "\n" + json.dumps({"id": 1, "text": CLEAN}) + "\n\n", encoding="utf-8"
    )
    assert len(read_split(tmp_path, "dev")) == 1


def test_a_foreign_proper_noun_is_not_rewritten() -> None:
    """`Chişinău` is Romanian, not a corrupted Kabyle `ṣ`.

    Rewriting `ş` there invents `Chiṣinău`. The same mistake counted Romanian `ț`
    as 176,494 corruptions during the Phase 2 census.
    """
    text = "Tamanaɣt n Muldufa d Chişinău."
    report = audit([sentence(text)], "kab_Latn")
    assert report.changed == 0
    assert report.protected == 1
    assert report.diffs == []


def test_a_protected_token_does_not_veto_repairs_beside_it() -> None:
    text = "Nesεa Chişinău d tamanaɣt n leεmeṛ."
    report = audit([sentence(text)], "kab_Latn")
    assert report.changed == 1
    corrected = report.diffs[0].corrected
    assert "Chişinău" in corrected
    assert "nesɛa" in corrected.lower()
    assert "leɛmeṛ" in corrected


def test_lowercase_kabyle_is_never_protected() -> None:
    report = audit([sentence("Mi yesεa Fidal 28 n yiseggasen di leεmeṛ-is.")], "kab_Latn")
    assert report.changed == 1
    assert report.protected == 0
    assert "yesɛa" in report.diffs[0].corrected


def test_corrections_are_keyed_by_split_not_id_alone(tmp_path: Path) -> None:
    """FLORES+ ids restart at 0 in every split; all 997 dev ids collide with devtest.

    Keying on `id` alone applied devtest corrections to dev sentences and marked
    534 of 2,009 lines corrected when only 326 were.
    """
    for split, text in (("dev", CLEAN), ("devtest", CORRUPT)):
        write_split(tmp_path, split, [{"id": 7, "text": text, "last_updated": "1.0"}])

    out = tmp_path / "out"
    assert main(["audit", "--root", str(tmp_path), "--out", str(out)]) == 0
    rows = [
        json.loads(line)
        for line in (out / "flores-plus-kab_Latn-corrected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_split = {r["split"]: r for r in rows}
    assert by_split["dev"]["corrected"] is False
    assert by_split["dev"]["text"] == CLEAN
    assert by_split["devtest"]["corrected"] is True
    assert by_split["devtest"]["text"] == REPAIRED
