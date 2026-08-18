from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.tokenizer.corpus import (
    read_texts,
    sample_sentences,
    word_frequencies,
    write_plain,
)
from agbalu.tokenizer.spec import TokenizerError

KAB = "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a."


def write_corpus(path: Path, texts: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )
    return path


class TestReadTexts:
    def test_missing_corpus_names_the_rebuild_command(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizerError, match="make extract"):
            list(read_texts(tmp_path / "absent.jsonl"))

    def test_reads_every_record(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [KAB, "Azul"])
        assert list(read_texts(corpus)) == [KAB, "Azul"]

    def test_empty_corpus_yields_nothing(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [])
        assert list(read_texts(corpus)) == []

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        corpus = tmp_path / "c.jsonl"
        corpus.write_text('\n\n{"text": "Azul"}\n   \n', encoding="utf-8")
        assert list(read_texts(corpus)) == ["Azul"]

    def test_truncated_final_line_reports_its_position(self, tmp_path: Path) -> None:
        corpus = tmp_path / "c.jsonl"
        corpus.write_text('{"text": "Azul"}\n{"text": "Aq', encoding="utf-8")
        with pytest.raises(TokenizerError, match=r":2 is not JSON"):
            list(read_texts(corpus))

    def test_record_without_a_text_field_reports_its_position(self, tmp_path: Path) -> None:
        corpus = tmp_path / "c.jsonl"
        corpus.write_text('{"text": "Azul"}\n{"sentence": "Aql-i"}\n', encoding="utf-8")
        with pytest.raises(TokenizerError, match=r":2 has no `text` field"):
            list(read_texts(corpus))

    def test_a_bare_json_array_is_not_a_record(self, tmp_path: Path) -> None:
        corpus = tmp_path / "c.jsonl"
        corpus.write_text('["Azul"]\n', encoding="utf-8")
        with pytest.raises(TokenizerError, match="has no `text` field"):
            list(read_texts(corpus))


class TestWritePlain:
    def test_writes_one_sentence_per_line(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [KAB, "Azul"])
        dest = tmp_path / "plain" / "corpus.txt"
        assert write_plain(corpus, dest) == 2
        assert dest.read_text(encoding="utf-8").splitlines() == [KAB, "Azul"]

    def test_flattens_embedded_newlines_and_tabs(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", ["Azul\nfell-awen\tAql-i"])
        dest = tmp_path / "corpus.txt"
        assert write_plain(corpus, dest) == 1
        assert dest.read_text(encoding="utf-8") == "Azul fell-awen Aql-i\n"

    def test_collapses_exotic_whitespace(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", ["Azul  fell-awen"])
        dest = tmp_path / "corpus.txt"
        write_plain(corpus, dest)
        assert dest.read_text(encoding="utf-8") == "Azul fell-awen\n"

    def test_drops_records_that_flatten_to_nothing(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", ["", "   ", "\u200b", "Azul"])
        dest = tmp_path / "corpus.txt"
        assert write_plain(corpus, dest) == 2
        assert dest.read_text(encoding="utf-8").splitlines() == ["\u200b", "Azul"]

    def test_leaves_no_staging_file_behind(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [KAB])
        dest = tmp_path / "corpus.txt"
        write_plain(corpus, dest)
        assert list(tmp_path.glob("*.partial")) == []

    def test_a_failed_run_does_not_replace_a_good_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "corpus.txt"
        dest.write_text("previous\n", encoding="utf-8")
        broken = tmp_path / "c.jsonl"
        broken.write_text('{"text": "Azul"}\n{oops\n', encoding="utf-8")
        with pytest.raises(TokenizerError):
            write_plain(broken, dest)
        assert dest.read_text(encoding="utf-8") == "previous\n"


class TestWordFrequencies:
    def test_counts_whitespace_tokens(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", ["azul azul fell-awen"])
        assert word_frequencies(corpus) == {"azul": 2, "fell-awen": 1}

    def test_empty_corpus_gives_an_empty_table(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [])
        assert word_frequencies(corpus) == {}

    def test_punctuation_stays_attached(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", ["azul, azul"])
        assert word_frequencies(corpus) == {"azul,": 1, "azul": 1}


class TestSampleSentences:
    def test_rejects_a_non_positive_size(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [KAB])
        with pytest.raises(TokenizerError, match="must be positive"):
            sample_sentences(corpus, 0)

    def test_returns_everything_when_the_corpus_is_smaller(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [KAB, "Azul"])
        assert sorted(sample_sentences(corpus, 100)) == sorted([KAB, "Azul"])

    def test_is_deterministic_for_a_fixed_seed(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [f"s{i}" for i in range(500)])
        assert sample_sentences(corpus, 20) == sample_sentences(corpus, 20)

    def test_a_different_seed_gives_a_different_sample(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [f"s{i}" for i in range(500)])
        assert sample_sentences(corpus, 20, seed=1) != sample_sentences(corpus, 20, seed=2)

    def test_draws_from_beyond_the_first_window(self, tmp_path: Path) -> None:
        corpus = write_corpus(tmp_path / "c.jsonl", [f"s{i}" for i in range(2_000)])
        sample = sample_sentences(corpus, 50)
        assert len(sample) == 50
        assert any(int(s[1:]) >= 50 for s in sample)
