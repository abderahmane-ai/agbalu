"""Selection and splitting for the MT fine-tuning corpus.

The two properties that matter: NLLB's own mined output stays out by default, since
fine-tuning it on that teaches it what it already knows; and the dev split holds out
*pairs*, not examples, because both directions of a pair are the same sentence pair and
splitting them would put the dev target in the training source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.mt.data import (
    Example,
    build,
    directions_for,
    is_mined,
    read,
    select,
    split,
)


def row(kab: str, other: str, source: str = "hf.tatoeba", defects: list[str] | None = None) -> str:
    payload = {
        "kab": kab,
        "eng": other,
        "source": source,
        "licence": "cc-by-2.0",
        "redistribution": "permissive",
        "defects": defects or [],
        "length_ratio": 1.0,
    }
    return json.dumps(payload, ensure_ascii=False)


def write_corpus(directory: Path, rows: list[str], language: str = "eng") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"agbalu-parallel-v1.kab-{language}.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestIsMined:
    @pytest.mark.parametrize(
        "source_id",
        ["opus.nllb-kab", "hf.boffire.nllb-en-kab", "hf.imsidag.nllb-en-kab", "HF.NLLB.Mixed"],
    )
    def test_every_reupload_of_nllb_is_recognised(self, source_id: str) -> None:
        """The marker is on the id, not on one uploader: three separate sources ship the
        same mined output."""
        assert is_mined(source_id)

    @pytest.mark.parametrize("source_id", ["opus.tatoeba-kab", "opus.bible-uedin-kab", ""])
    def test_human_sources_are_not(self, source_id: str) -> None:
        assert not is_mined(source_id)


class TestDirectionsFor:
    def test_english_and_french_both_expand_to_two(self) -> None:
        assert directions_for("eng") == ("kab-eng", "eng-kab")
        assert directions_for("fra") == ("kab-fra", "fra-kab")

    def test_an_unknown_language_is_refused_not_guessed(self) -> None:
        with pytest.raises(ValueError, match="no directions"):
            directions_for("deu")


class TestSelect:
    def test_mined_pairs_are_excluded_by_default(self, tmp_path: Path) -> None:
        write_corpus(tmp_path, [row("a", "b", "opus.nllb-kab"), row("c", "d", "opus.tatoeba-kab")])
        examples, stats = select(tmp_path)
        assert stats.mined == 1
        assert stats.kept_pairs == 1
        assert {e.source_id for e in examples} == {"opus.tatoeba-kab"}

    def test_mined_pairs_can_be_asked_for(self, tmp_path: Path) -> None:
        write_corpus(tmp_path, [row("a", "b", "opus.nllb-kab")])
        _, stats = select(tmp_path, include_mined=True)
        assert stats.kept_pairs == 1
        assert stats.mined == 0

    def test_hard_defects_are_dropped_and_soft_ones_kept(self, tmp_path: Path) -> None:
        write_corpus(
            tmp_path,
            [
                row("a", "b", defects=["untranslated-copy"]),
                row("c", "d", defects=["number-mismatch"]),
            ],
        )
        _, stats = select(tmp_path)
        assert stats.defective == 1
        assert stats.kept_pairs == 1

    def test_each_pair_becomes_both_directions(self, tmp_path: Path) -> None:
        write_corpus(tmp_path, [row("azul", "hello")])
        examples, stats = select(tmp_path)
        assert stats.by_direction == {"kab-eng": 1, "eng-kab": 1}
        assert Example("azul", "hello", "kab-eng", "hf.tatoeba") in examples
        assert Example("hello", "azul", "eng-kab", "hf.tatoeba") in examples

    def test_duplicates_are_removed_across_case_and_composition(self, tmp_path: Path) -> None:
        """The same pair arrives from several uploaders with different capitalisation;
        counting each as new data would inflate the corpus with nothing."""
        write_corpus(tmp_path, [row("Azul", "Hello"), row("azul", "hello"), row("AZUL", "HELLO")])
        _, stats = select(tmp_path)
        assert stats.kept_pairs == 1
        assert stats.duplicate == 2

    def test_an_empty_corpus_yields_nothing_rather_than_failing(self, tmp_path: Path) -> None:
        write_corpus(tmp_path, [])
        examples, stats = select(tmp_path)
        assert examples == []
        assert stats.read == 0

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = write_corpus(tmp_path, [row("a", "b")])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        _, stats = select(tmp_path)
        assert stats.read == 1

    def test_both_language_files_are_read(self, tmp_path: Path) -> None:
        write_corpus(tmp_path, [row("a", "b")], "eng")
        french = tmp_path / "agbalu-parallel-v1.kab-fra.jsonl"
        french.write_text(
            json.dumps(
                {
                    "kab": "c",
                    "fra": "d",
                    "source": "hf.sifal",
                    "licence": "cc",
                    "redistribution": "unclear",
                    "defects": [],
                    "length_ratio": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _, stats = select(tmp_path)
        assert set(stats.by_direction) == {"kab-eng", "eng-kab", "kab-fra", "fra-kab"}


class TestSplit:
    def test_both_directions_of_a_pair_land_on_the_same_side(self) -> None:
        """Splitting on the example would put the dev target into a training source."""
        examples = [Example(f"kab{i}", f"eng{i}", "kab-eng", "s") for i in range(50)] + [
            Example(f"eng{i}", f"kab{i}", "eng-kab", "s") for i in range(50)
        ]
        train, dev = split(examples, dev_pairs=10, seed=0)

        def pairs(items: list[Example]) -> set[str]:
            return {e.source if e.direction.startswith("kab") else e.target for e in items}

        assert not pairs(train) & pairs(dev)
        assert len(dev) == 20

    def test_it_is_deterministic_for_a_seed(self) -> None:
        examples = [Example(f"a{i}", f"b{i}", "kab-eng", "s") for i in range(100)]
        first, _ = split(examples, dev_pairs=10, seed=7)
        second, _ = split(examples, dev_pairs=10, seed=7)
        assert [e.source for e in first] == [e.source for e in second]

    def test_a_dev_request_larger_than_the_corpus_takes_everything(self) -> None:
        examples = [Example("a", "b", "kab-eng", "s")]
        train, dev = split(examples, dev_pairs=500)
        assert train == []
        assert len(dev) == 1

    def test_an_empty_corpus_splits_into_two_empties(self) -> None:
        assert split([], dev_pairs=10) == ([], [])


class TestBuild:
    def test_it_writes_both_splits_and_the_stats_beside_them(self, tmp_path: Path) -> None:
        corpus = tmp_path / "parallel"
        write_corpus(corpus, [row(f"kab{i}", f"eng{i}") for i in range(20)])
        out = tmp_path / "mt"

        stats = build(corpus, out, dev_pairs=5)

        assert (out / "train.jsonl").is_file()
        assert (out / "dev.jsonl").is_file()
        written = json.loads((out / "agbalu-mt-v1.stats.json").read_text(encoding="utf-8"))
        assert written["kept_pairs"] == 20
        assert written["examples"] == 40
        assert stats["train"] == 30
        assert stats["dev"] == 10

    def test_the_written_rows_round_trip(self, tmp_path: Path) -> None:
        corpus = tmp_path / "parallel"
        write_corpus(corpus, [row("azul", "hello")])
        out = tmp_path / "mt"
        build(corpus, out, dev_pairs=0)

        restored = read(out / "train.jsonl")
        assert len(restored) == 2
        assert {e.direction for e in restored} == {"kab-eng", "eng-kab"}
