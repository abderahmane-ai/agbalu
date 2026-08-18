from __future__ import annotations

import json
from pathlib import Path

from agbalu.bench.contamination import NGRAM, ngrams, scan
from agbalu.bench.flores import Sentence

LONG = "Deg wass n Arim imusnawen seg uɣerbaz n tujjya berrḥen-d s usnulfu n wallal amaynut"
CORRUPT = LONG.replace("ɣ", "γ")
OTHER = "Tameṭṭut-nni tenna-yas belli acu i yellan d ayen ilaqen i wemdan akken ad yidir"


def sentence(text: str, ident: int = 1, split: str = "dev") -> Sentence:
    return Sentence(
        id=ident,
        text=text,
        split=split,  # type: ignore[arg-type]
        domain="wikinews",
        topic="News",
        url="",
        last_updated="1.0",
        iso_639_3="kab",
        iso_15924="Latn",
    )


def corpus(tmp_path: Path, *texts: str, source: str = "hf.some.source") -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(
                json.dumps(
                    {"text": text, "source": source, "licence": "cc0-1.0"}, ensure_ascii=False
                )
                + "\n"
            )
    return path


def test_an_exactly_leaked_sentence_is_caught(tmp_path: Path) -> None:
    path = corpus(tmp_path, OTHER, LONG)
    report = scan(path, [sentence(LONG)])
    assert len(report.leaked) == 1
    assert report.by_kind["exact"] == 1
    assert report.leaked[0].source == "hf.some.source"


def test_a_leak_is_caught_across_the_homoglyph_difference(tmp_path: Path) -> None:
    """The corpus is normalised and FLORES+ is not.

    Indexing only the raw benchmark text would miss every leaked sentence carrying
    the Greek-epsilon corruption, and report a clean corpus that is not clean.
    """
    path = corpus(tmp_path, LONG)
    report = scan(path, [sentence(CORRUPT)])
    assert len(report.leaked) == 1


def test_an_edited_leak_is_caught_by_ngram(tmp_path: Path) -> None:
    embedded = f"Yenna-yas wemdan-nni : {LONG} , dɣa yeffeɣ seg wexxam."
    path = corpus(tmp_path, embedded)
    report = scan(path, [sentence(LONG)])
    assert len(report.leaked) == 1
    assert report.by_kind["ngram"] == 1


def test_case_differences_do_not_hide_a_leak(tmp_path: Path) -> None:
    path = corpus(tmp_path, LONG.upper())
    report = scan(path, [sentence(LONG)])
    assert len(report.leaked) == 1


def test_a_clean_corpus_reports_zero(tmp_path: Path) -> None:
    path = corpus(tmp_path, OTHER)
    report = scan(path, [sentence(LONG)])
    assert report.leaked == []
    assert report.leak_rate == 0.0
    assert report.corpus_lines == 1


def test_each_benchmark_sentence_is_counted_once(tmp_path: Path) -> None:
    path = corpus(tmp_path, LONG, LONG, LONG)
    report = scan(path, [sentence(LONG)])
    assert len(report.leaked) == 1


def test_leak_rate_is_over_the_benchmark_not_the_corpus(tmp_path: Path) -> None:
    path = corpus(tmp_path, LONG, *[OTHER] * 99)
    report = scan(path, [sentence(LONG, 1), sentence(OTHER, 2)])
    assert report.leak_rate == 1.0


def test_an_empty_corpus_is_not_an_error(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("", encoding="utf-8")
    report = scan(path, [sentence(LONG)])
    assert report.corpus_lines == 0
    assert report.leaked == []


def test_blank_corpus_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n" + json.dumps({"text": LONG, "source": "s"}, ensure_ascii=False) + "\n\n",
        encoding="utf-8",
    )
    report = scan(path, [sentence(LONG)])
    assert report.corpus_lines == 1
    assert len(report.leaked) == 1


def test_ngrams_of_a_short_sentence_are_empty() -> None:
    assert ngrams("azul fell-awen") == set()
    assert len(ngrams(" ".join(["w"] * NGRAM))) == 1


def test_ngrams_are_case_folded() -> None:
    assert ngrams(LONG) == ngrams(LONG.upper())


def test_short_benchmark_sentences_still_match_exactly(tmp_path: Path) -> None:
    short = "Azul fell-awen"
    path = corpus(tmp_path, short)
    report = scan(path, [sentence(short)])
    assert len(report.leaked) == 1
    assert report.by_kind["exact"] == 1


def test_same_id_in_two_splits_counts_as_two_leaks(tmp_path: Path) -> None:
    """FLORES+ ids restart per split, so `id` alone cannot dedupe leaks."""
    path = corpus(tmp_path, LONG, OTHER)
    report = scan(path, [sentence(LONG, 7, "dev"), sentence(OTHER, 7, "devtest")])
    assert len(report.leaked) == 2
    assert dict(report.by_split) == {"dev": 1, "devtest": 1}
