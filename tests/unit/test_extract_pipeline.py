from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agbalu.acquire.manifest import Manifest
from agbalu.acquire.models import ManifestEntry
from agbalu.extract.pipeline import (
    MAX_CHARS,
    CorpusBuilder,
    SourceStats,
    _letter_ratio,
    fingerprint,
    summary,
)
from agbalu.registry.models import Registry, Source, SourceSize

KAB = "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a."
KAB2 = "Ɣef waya i d-nusa ɣer da, imi nezmer ad nemmeslay akken ilaq."


def make_source(source_id: str = "test.src", **kwargs: object) -> Source:
    defaults: dict[str, object] = {
        "id": source_id,
        "name": "Test source",
        "modality": "text",
        "tier": "core",
        "access": "hf",
        "uri": "https://huggingface.co/datasets/x/y",
        "licence": "cc0-1.0",
        "languages": ("kab",),
        "size": SourceSize(rows=2),
        "retrieved": date(2026, 8, 6),
    }
    return Source.model_validate(defaults | kwargs)


def make_registry(*sources: Source) -> Registry:
    return Registry(version="1.0.0", surveyed=date(2026, 8, 6), sources=sources)


def land(root: Path, source: Source, name: str, text: str) -> None:
    directory = root / source.id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")
    Manifest(root).append(
        ManifestEntry(
            source_id=source.id,
            path=name,
            bytes=len(text.encode()),
            sha256="0" * 64,
            kind="text",
            target="local",
            uri=source.uri,
            fetched_at=datetime.now(UTC),
            revision=None,
        )
    )


def build(tmp_path: Path, registry: Registry) -> tuple[list[dict[str, str]], list[SourceStats]]:
    out = tmp_path / "out" / "corpus.jsonl"
    stats = CorpusBuilder(registry, tmp_path / "raw").build(out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return lines, stats


def test_extracts_normalises_and_carries_provenance(tmp_path: Path) -> None:
    source = make_source()
    # Greek epsilon where ɛ belongs — the defect the normaliser exists to repair.
    land(tmp_path / "raw", source, "a.tsv", f"kab\nteεyiḍ tafsut\n{KAB}\n")
    lines, stats = build(tmp_path, make_registry(source))

    assert lines[0]["text"] == "teɛyiḍ tafsut"
    assert lines[0]["source"] == "test.src"
    assert lines[0]["licence"] == "cc0-1.0"
    assert lines[0]["redistribution"] == "permissive"
    assert stats[0].repaired == 1


def test_duplicates_are_dropped_across_sources(tmp_path: Path) -> None:
    a = make_source("test.a")
    b = make_source("test.b")
    land(tmp_path / "raw", a, "a.tsv", f"kab\n{KAB}\n")
    land(tmp_path / "raw", b, "b.tsv", f"kab\n{KAB}\n")
    lines, stats = build(tmp_path, make_registry(a, b))

    assert len(lines) == 1
    assert lines[0]["source"] == "test.a"
    assert {s.source_id: s.duplicate for s in stats}["test.b"] == 1


def test_dedup_ignores_case_and_punctuation() -> None:
    assert fingerprint("Azul fell-awen!") == fingerprint("azul fell awen")
    assert fingerprint("Azul  fell-awen") == fingerprint("azul fell-awen")
    assert fingerprint("azul") != fingerprint("ddunit")


def test_short_long_and_non_prose_lines_are_dropped(tmp_path: Path) -> None:
    source = make_source()
    land(
        tmp_path / "raw",
        source,
        "a.tsv",
        "kab\n" + f"a\n{'x' * (MAX_CHARS + 1)}\n1234567890 ||| 42\n{KAB}\n",
    )
    lines, stats = build(tmp_path, make_registry(source))

    assert [line["text"] for line in lines] == [KAB]
    assert stats[0].too_short == 1
    assert stats[0].too_long == 1
    assert stats[0].not_prose == 1


def test_benchmark_sources_are_never_ingested(tmp_path: Path) -> None:
    flores = make_source("hf.flores-plus-kab", tier="reference", modality="annotation")
    land(tmp_path / "raw", flores, "a.tsv", f"kab\n{KAB}\n")
    lines, _ = build(tmp_path, make_registry(flores))
    assert lines == []


def test_excluded_tier_and_speech_are_skipped(tmp_path: Path) -> None:
    excluded = make_source("test.excluded", tier="excluded", notes="left out on purpose")
    speech = make_source("test.speech", modality="speech")
    for source in (excluded, speech):
        land(tmp_path / "raw", source, "a.tsv", f"kab\n{KAB}\n")
    lines, _ = build(tmp_path, make_registry(excluded, speech))
    assert lines == []


def test_source_with_no_kabyle_column_contributes_nothing(tmp_path: Path) -> None:
    source = make_source()
    land(
        tmp_path / "raw",
        source,
        "a.tsv",
        "a\tb\nI am at home and will not go\tThis is why we came here today\n",
    )
    lines, stats = build(tmp_path, make_registry(source))
    assert lines == []
    assert stats[0].kept == 0


def test_unreadable_artifact_is_recorded_not_fatal(tmp_path: Path) -> None:
    source = make_source()
    root = tmp_path / "raw"
    land(root, source, "good.tsv", f"kab\n{KAB}\n")
    land(root, source, "broken.zip", "not a zip at all")
    lines, stats = build(tmp_path, make_registry(source))

    assert [line["text"] for line in lines] == [KAB]
    assert stats[0].errors


def test_empty_corpus_writes_an_empty_file(tmp_path: Path) -> None:
    out = tmp_path / "out" / "corpus.jsonl"
    CorpusBuilder(make_registry(make_source()), tmp_path / "raw").build(out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_output_is_written_atomically(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\n{KAB}\n")
    out = tmp_path / "out" / "corpus.jsonl"
    CorpusBuilder(make_registry(source), tmp_path / "raw").build(out)
    assert not out.with_name(out.name + ".part").exists()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("azul", 1.0), ("", 0.0), ("1234", 0.0), ("ab12", 0.5)],
)
def test_letter_ratio(text: str, expected: float) -> None:
    assert _letter_ratio(text) == pytest.approx(expected)


def test_summary_totals_every_counter() -> None:
    stats = [
        SourceStats("a", read=10, kept=6, duplicate=2, repaired=1, too_short=2),
        SourceStats("b", read=5, kept=5, repaired=3),
    ]
    result = summary(stats)
    assert result["read"] == 15
    assert result["kept"] == 11
    assert result["duplicate"] == 2
    assert result["repaired"] == 4
    assert result["sources"] == 2


def test_tifinagh_rows_are_dropped_by_the_script_gate(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\nⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ⴰⵔ ⵜⵉⵎⵍⵉⵍⵉⵜ\n{KAB}\n")
    lines, stats = build(tmp_path, make_registry(source))
    assert [line["text"] for line in lines] == [KAB]
    assert stats[0].wrong_script == 1


def test_lexicon_sources_are_not_text(tmp_path: Path) -> None:
    lexicon = make_source("test.lex", modality="lexicon")
    land(tmp_path / "raw", lexicon, "a.tsv", "kab\nittemnunnuḍen\nteḥḍimemt\n")
    lines, _ = build(tmp_path, make_registry(lexicon))
    assert lines == []


def test_permissive_sources_claim_shared_sentences_first(tmp_path: Path) -> None:
    nc = make_source("a.noncommercial", licence="cc-by-nc-4.0", tier="supplementary")
    permissive = make_source("z.permissive", licence="cc0-1.0")
    for source in (nc, permissive):
        land(tmp_path / "raw", source, "a.tsv", f"kab\n{KAB}\n")
    lines, _ = build(tmp_path, make_registry(nc, permissive))

    assert len(lines) == 1
    assert lines[0]["source"] == "z.permissive"
    assert lines[0]["redistribution"] == "permissive"


def test_core_tier_outranks_supplementary_at_equal_licence(tmp_path: Path) -> None:
    supplementary = make_source("a.supp", tier="supplementary")
    core = make_source("z.core", tier="core")
    for source in (supplementary, core):
        land(tmp_path / "raw", source, "a.tsv", f"kab\n{KAB}\n")
    lines, _ = build(tmp_path, make_registry(supplementary, core))
    assert lines[0]["source"] == "z.core"
