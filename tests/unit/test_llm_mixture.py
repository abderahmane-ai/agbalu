"""The aligned corpus: each human pair once, tagged, held-out and contaminated pairs out."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agbalu.llm import mixture
from agbalu.llm.corpus import Record, Source, is_held_out
from agbalu.llm.decontaminate import Decontaminator

KAB = "Tamurt-nni tella deg yidurar"
ENG = "That country was in the mountains"
EVAL = "the committee met on tuesday to decide the budget for next year in the capital"


class WordCounter:
    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def mt_source(path: Path, rows: Sequence[tuple[str, str]]) -> Source:
    with path.open("w", encoding="utf-8") as handle:
        for kab, eng in rows:
            for direction, first, second in (("kab-eng", kab, eng), ("eng-kab", eng, kab)):
                row = {"source": first, "target": second, "direction": direction}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return Source("mt", path, "aligned", ("source", "target"), direction_field="direction")


def run(source: Source, rate: int = 10_000) -> tuple[list[mixture.SourceTally], list[str]]:
    sink = io.StringIO()
    writer = mixture.AlignedWriter(Decontaminator.build([EVAL]), WordCounter(), sink, rate=rate)
    tallies = mixture.build([source], None, writer)
    return tallies, [json.loads(line)["text"] for line in sink.getvalue().splitlines()]


class TestDocument:
    def test_kabyle_first_on_even_rows(self) -> None:
        text = mixture.aligned_document(KAB, ENG, "eng_Latn", kabyle_first=True)
        assert text == f"kab_Latn: {KAB}\neng_Latn: {ENG}"

    def test_the_other_side_first_on_odd_rows(self) -> None:
        text = mixture.aligned_document(KAB, ENG, "eng_Latn", kabyle_first=False)
        assert text == f"eng_Latn: {ENG}\nkab_Latn: {KAB}"


class TestBuild:
    def test_the_mirror_row_of_a_pair_is_dropped(self, tmp_path: Path) -> None:
        pairs = [(f"{KAB} {i}", f"{ENG} {i}") for i in range(10)]
        tallies, texts = run(mt_source(tmp_path / "mt.jsonl", pairs))
        assert len(texts) == 10
        assert tallies[0].rejected["duplicate_pair"] == 10

    def test_every_row_read_is_kept_or_rejected(self, tmp_path: Path) -> None:
        pairs = [(f"{KAB} {i}", f"{ENG} {i}") for i in range(30)] + [(KAB, EVAL)]
        tallies, _ = run(mt_source(tmp_path / "mt.jsonl", pairs), rate=7)
        tally = tallies[0]
        assert tally.read == 62
        assert tally.read == tally.kept + sum(tally.rejected.values())
        assert tally.rejected["held_out"] > 0

    def test_a_pair_containing_an_evaluation_sentence_is_rejected(self, tmp_path: Path) -> None:
        tallies, texts = run(mt_source(tmp_path / "mt.jsonl", [(KAB, f"So {EVAL}.")]))
        assert texts == []
        assert tallies[0].rejected["contaminated"] == 1

    def test_held_out_records_never_reach_the_corpus(self, tmp_path: Path) -> None:
        pairs = [(f"{KAB} {i}", f"{ENG} {i}") for i in range(60)]
        _, texts = run(mt_source(tmp_path / "mt.jsonl", pairs), rate=3)
        for kab, eng in pairs:
            if is_held_out(kab, rate=3) or is_held_out(eng, rate=3):
                assert not any(kab + "\n" in t or t.endswith(kab) for t in texts)

    def test_a_monolingual_source_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text('{"text": "azul"}\n', encoding="utf-8")
        with pytest.raises(mixture.MixtureError, match="aligned pairs only"):
            run(Source("m", path, "kabyle", ("text",)))


class TestTranslations:
    def test_rows_become_records_with_english_beside_kabyle(self, tmp_path: Path) -> None:
        table = pa.table({"og_full_text": [KAB, None, "x"], "translated_text": [ENG, "y", ""]})
        pq.write_table(table, tmp_path / "000.parquet")
        records = list(mixture.translation_records(tmp_path))
        assert records == [Record(kabyle=KAB, index=0, other=ENG, code="eng_Latn")]

    def test_an_empty_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(mixture.MixtureError):
            list(mixture.translation_records(tmp_path))
