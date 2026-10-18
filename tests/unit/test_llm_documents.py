"""The document corpus: filters in order, every record kept or rejected with a reason."""

from __future__ import annotations

import bz2
import io
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agbalu.llm import documents
from agbalu.llm.corpus import is_held_out
from agbalu.llm.decontaminate import Decontaminator
from agbalu.normalise import Normaliser

EVAL = "the committee met on tuesday to decide the budget for next year in the capital"
PARAGRAPH = (
    "Tamurt n Leqbayel tella deg ugafa n Lezzayer, gar yidurar d yill. "
    "Imezdaɣ-is ttmeslayen taqbaylit, tutlayt tamaziɣt yettwarun s yisekkilen ilatiniyen. "
)


class WordCounter:
    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


class FakeScorer:
    """Kabyle unless the window contains the word `English`."""

    def __init__(self) -> None:
        self.calls = 0

    def probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]:
        self.calls += 1
        return [
            {"kab_Latn": 0.1, "NOT_AMAZIGH": 0.9}
            if "English" in text
            else {"kab_Latn": 0.9, "NOT_AMAZIGH": 0.1}
            for text in texts
        ]


def builder(batch: int = 4) -> tuple[documents.Builder, io.StringIO, io.StringIO]:
    sink, rejects = io.StringIO(), io.StringIO()
    index = Decontaminator.build([EVAL])
    return (
        documents.Builder(FakeScorer(), index, WordCounter(), sink, rejects, batch=batch),
        sink,
        rejects,
    )


class TestHelpers:
    def test_clean_unifies_line_endings_and_blank_runs(self) -> None:
        assert documents.clean("a\r\nb\r\r\n\n\n\nc  ", Normaliser()) == "a\nb\n\nc"

    def test_windows_cover_the_start_and_the_end(self) -> None:
        text = "".join(chr(97 + i % 26) for i in range(1000))
        parts = documents.windows(text, size=100, count=4)
        assert parts[0] == text[:100]
        assert parts[-1] == text[-100:]
        assert len(parts) == 4

    def test_a_short_text_is_one_window(self) -> None:
        assert documents.windows("azul", size=100) == ["azul"]

    def test_the_verdict_is_the_highest_mean(self) -> None:
        rows = [{"a": 0.9, "b": 0.1}, {"a": 0.2, "b": 0.8}, {"a": 0.3, "b": 0.7}]
        assert documents.verdict(rows) == "b"

    def test_wiki_markup_categories_and_files_are_dropped(self) -> None:
        text = "'''Azul''' d [[awal]].\n[[Taggayt:Tutlayin]]\n[[Afaylu:x.png]]"
        plain = documents.wiki_plaintext(text, "Azul")
        assert plain.startswith("Azul\n\nAzul d awal.")
        assert "Taggayt" not in plain
        assert "Afaylu" not in plain


class TestBuilder:
    def test_each_filter_names_its_reason(self) -> None:
        build, sink, rejects = builder()
        stats = build.source_stats("s", "cc0-1.0", "permissive")
        long_doc = PARAGRAPH * 3
        build.offer(long_doc, "1", stats, is_document=True)
        build.offer(long_doc, "2", stats, is_document=True)
        build.offer("short", "3", stats, is_document=True)
        build.offer("", "4", stats, is_document=True)
        build.offer(PARAGRAPH * 2 + EVAL, "5", stats, is_document=True)
        build.offer("English " * 60, "6", stats, is_document=True)
        build.offer(long_doc + " Ih.", "7", stats, is_document=True)
        build.flush()
        reasons = [json.loads(line)["reason"] for line in rejects.getvalue().splitlines()]
        assert sorted(r.split(":")[0] for r in reasons) == sorted(
            ["duplicate", "too_short", "empty", "contaminated", "language", "near_duplicate"]
        )
        assert stats.kept == 1
        assert stats.read == stats.kept + sum(stats.rejected.values())
        assert len(sink.getvalue().splitlines()) == 1

    def test_a_held_out_sentence_row_is_rejected(self) -> None:
        build, _, _ = builder()
        stats = build.source_stats("s", "cc0-1.0", "permissive")
        held = next(
            f"Azul {i} fell-awen." for i in range(5000) if is_held_out(f"Azul {i} fell-awen.")
        )
        build.offer(held, "1", stats, is_document=False)
        assert stats.rejected["held_out"] == 1

    def test_kept_rows_carry_source_and_licence(self) -> None:
        build, sink, _ = builder()
        stats = build.source_stats("s", "cc0-1.0", "permissive")
        build.offer(PARAGRAPH * 3, "1", stats, is_document=True)
        build.flush()
        row = json.loads(sink.getvalue())
        assert (row["source"], row["licence"], row["redistribution"]) == (
            "s",
            "cc0-1.0",
            "permissive",
        )
        assert stats.tokens == len(row["text"].split())


class TestReaders:
    def test_parquet_fields_are_joined_title_first(self, tmp_path: Path) -> None:
        directory = tmp_path / "src.id" / "data"
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table({"t": ["Azul", None], "b": ["Tafat", "Aman"]}), directory / "0.parquet"
        )
        source = documents.DocumentSource("src.id", ("data/*.parquet",), "parquet", ("t", "b"))
        found = list(documents.read_source(source, tmp_path))
        assert [d.text for d in found] == ["Azul\n\nTafat", "Aman"]
        assert found[0].key == "data/0.parquet:0"

    def test_files_of_the_same_name_in_different_folders_get_different_keys(
        self, tmp_path: Path
    ) -> None:
        for split in ("train", "test"):
            directory = tmp_path / "src.id" / split
            directory.mkdir(parents=True)
            pq.write_table(pa.table({"text": ["a", "b"]}), directory / "000.parquet")
        source = documents.DocumentSource("src.id", ("*/*.parquet",), "parquet", ("text",))
        keys = [d.key for d in documents.read_source(source, tmp_path)]
        assert len(keys) == len(set(keys)) == 4

    def test_a_missing_column_is_named(self, tmp_path: Path) -> None:
        directory = tmp_path / "src.id"
        directory.mkdir()
        pq.write_table(pa.table({"x": ["a"]}), directory / "0.parquet")
        source = documents.DocumentSource("src.id", ("*.parquet",), "parquet", ("text",))
        with pytest.raises(documents.DocumentError, match="no column"):
            list(documents.read_source(source, tmp_path))

    def test_nothing_matching_is_refused(self, tmp_path: Path) -> None:
        source = documents.DocumentSource("absent", ("*.jsonl",), "jsonl", ("text",))
        with pytest.raises(documents.DocumentError, match="nothing matches"):
            list(documents.read_source(source, tmp_path))

    def test_wiki_dumps_yield_main_namespace_articles_without_redirects(
        self, tmp_path: Path
    ) -> None:
        xml = (
            '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
            "<page><title>Azul</title><ns>0</ns>"
            "<revision><text>Azul d awal.</text></revision></page>"
            "<page><title>Talk</title><ns>1</ns><revision><text>x</text></revision></page>"
            '<page><title>R</title><ns>0</ns><redirect title="Azul"/><revision><text>#R</text>'
            "</revision></page></mediawiki>"
        )
        directory = tmp_path / "wiki"
        directory.mkdir()
        (directory / "kabwiki-pages-articles.xml.bz2").write_bytes(bz2.compress(xml.encode()))
        source = documents.DocumentSource(
            "wiki", ("*-pages-articles.xml.bz2",), "wikitext", ("text",)
        )
        found = list(documents.read_source(source, tmp_path))
        assert [d.text for d in found] == ["Azul\n\nAzul d awal."]
