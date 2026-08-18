"""The evaluation sets withheld from continued pretraining (task 11.2).

The load-bearing assertion is the last class: what `holdout` writes and what `mixture`
writes must be disjoint. Everything else in Phase 11 is measured against these files, and
a leak would show up as an improvement rather than as a failure.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.llm import holdout, mixture
from agbalu.llm.corpus import LANGUAGE_TAG, CorpusError, Source

KAB = "Aɣbalu n tmaziɣt d ayen ɣef nettmeslay deg umezruy-nneɣ."
ENG = "The source of Tamazight is what we speak about in our history."
FRA = "La source du tamazight est ce dont nous parlons dans notre histoire."


class WordCounter:
    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def mono(path: Path, rows: int) -> Source:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(json.dumps({"text": f"{KAB} {i}"}, ensure_ascii=False) + "\n")
    return Source(name="mono", path=path, kind="kabyle", fields=("text",))


def mt(path: Path, rows: int) -> Source:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(
                json.dumps(
                    {"source": f"{KAB} {i}", "target": f"{ENG} {i}", "direction": "kab-eng"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {"source": f"{ENG} {i}", "target": f"{KAB} {i}", "direction": "eng-kab"},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return Source(
        name="mt",
        path=path,
        kind="aligned",
        fields=("source", "target"),
        direction_field="direction",
    )


def fra_pairs(path: Path, rows: int) -> Source:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(
                json.dumps({"kab": f"{KAB} {i}", "fra": f"{FRA} {i}"}, ensure_ascii=False) + "\n"
            )
    return Source(name="fra", path=path, kind="aligned", fields=("kab", "fra"), code="fra_Latn")


def texts_of(path: Path) -> list[str]:
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]


class TestBuild:
    def test_a_monolingual_source_feeds_the_kabyle_set(self, tmp_path: Path) -> None:
        result = holdout.build([mono(tmp_path / "m.jsonl", 20)], tmp_path / "out", rate=1)
        path = tmp_path / "out" / "heldout-kab.jsonl"
        assert path.is_file()
        assert len(texts_of(path)) == 20
        assert result.documents("kab_Latn") == 20

    def test_an_aligned_source_feeds_the_other_language(self, tmp_path: Path) -> None:
        """The Kabyle side of an aligned row is training data's shape, not the eval set's;
        what the retention measurement needs is the English and French sides."""
        holdout.build([mt(tmp_path / "mt.jsonl", 6)], tmp_path / "out", rate=1)
        written = texts_of(tmp_path / "out" / "heldout-eng.jsonl")
        assert not (tmp_path / "out" / "heldout-kab.jsonl").exists()
        assert all(text.startswith(ENG) for text in written)

    def test_the_mirrored_copy_of_a_pair_is_not_written_twice(self, tmp_path: Path) -> None:
        holdout.build([mt(tmp_path / "mt.jsonl", 6)], tmp_path / "out", rate=1)
        written = texts_of(tmp_path / "out" / "heldout-eng.jsonl")
        assert len(written) == len(set(written)) == 6

    def test_three_languages_land_in_three_files(self, tmp_path: Path) -> None:
        result = holdout.build(
            [
                mono(tmp_path / "m.jsonl", 5),
                mt(tmp_path / "mt.jsonl", 5),
                fra_pairs(tmp_path / "f.jsonl", 5),
            ],
            tmp_path / "out",
            rate=1,
        )
        assert {p.name for p in result.paths} == {
            "heldout-kab.jsonl",
            "heldout-eng.jsonl",
            "heldout-fra.jsonl",
        }

    def test_each_row_carries_its_language_and_source(self, tmp_path: Path) -> None:
        holdout.build([fra_pairs(tmp_path / "f.jsonl", 3)], tmp_path / "out", rate=1)
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "heldout-fra.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert all(row["language"] == "fra_Latn" and row["source"] == "fra" for row in rows)

    def test_the_cap_bounds_each_source_and_language(self, tmp_path: Path) -> None:
        result = holdout.build([mono(tmp_path / "m.jsonl", 50)], tmp_path / "out", rate=1, cap=7)
        assert result.documents("kab_Latn") == 7

    def test_the_cap_applies_per_source_so_one_does_not_starve_another(
        self, tmp_path: Path
    ) -> None:
        result = holdout.build(
            [mt(tmp_path / "mt.jsonl", 20), fra_pairs(tmp_path / "f.jsonl", 20)],
            tmp_path / "out",
            rate=1,
            cap=4,
        )
        assert result.documents("eng_Latn") == 4
        assert result.documents("fra_Latn") == 4

    def test_nothing_outside_the_length_bounds_is_selected(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for text in ("short", "x" * (holdout.MAX_CHARS + 1), KAB):
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        source = Source(name="m", path=path, kind="kabyle", fields=("text",))
        holdout.build([source], tmp_path / "out", rate=1)
        assert texts_of(tmp_path / "out" / "heldout-kab.jsonl") == [KAB]

    def test_a_repeated_sentence_is_written_once(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text(
            (json.dumps({"text": KAB}, ensure_ascii=False) + "\n") * 4, encoding="utf-8"
        )
        source = Source(name="m", path=path, kind="kabyle", fields=("text",))
        holdout.build([source], tmp_path / "out", rate=1)
        assert texts_of(tmp_path / "out" / "heldout-kab.jsonl") == [KAB]

    def test_a_zero_cap_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="cap must be positive"):
            holdout.build([mono(tmp_path / "m.jsonl", 2)], tmp_path / "out", rate=1, cap=0)

    def test_the_split_is_recorded_beside_the_counts(self, tmp_path: Path) -> None:
        result = holdout.build([mono(tmp_path / "m.jsonl", 4)], tmp_path / "out", rate=1, cap=3)
        payload = result.as_dict()
        assert payload["rate"] == 1
        assert payload["cap"] == 3
        assert payload["min_chars"] == holdout.MIN_CHARS


class TestShortCode:
    def test_every_tag_round_trips(self) -> None:
        assert all(holdout.short_code(code) == short for short, code in LANGUAGE_TAG.items())

    def test_an_unregistered_tag_is_refused(self) -> None:
        with pytest.raises(CorpusError, match="no short code"):
            holdout.short_code("arb_Arab")


class TestRead:
    def test_documents_come_back_in_file_order(self, tmp_path: Path) -> None:
        holdout.build([mono(tmp_path / "m.jsonl", 4)], tmp_path / "out", rate=1)
        path = tmp_path / "out" / "heldout-kab.jsonl"
        assert holdout.read(path) == texts_of(path)

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="not found"):
            holdout.read(tmp_path / "heldout-kab.jsonl")

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "heldout-kab.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(CorpusError, match="empty"):
            holdout.read(path)

    def test_a_row_without_text_names_its_line(self, tmp_path: Path) -> None:
        path = tmp_path / "heldout-kab.jsonl"
        path.write_text(json.dumps({"language": "kab_Latn"}) + "\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="carries no text"):
            holdout.read(path)


class TestDisjointness:
    """What is evaluated must be absent from what is trained on, at any rate."""

    def sides_written(self, path: Path) -> set[str]:
        found: set[str] = set()
        for text in texts_of(path):
            for line in text.split("\n"):
                head, _, rest = line.partition(": ")
                found.add(rest if head in LANGUAGE_TAG.values() else line)
        return found

    @pytest.mark.parametrize("rate", [2, 3, 7])
    def test_no_evaluation_document_appears_in_the_corpus(self, tmp_path: Path, rate: int) -> None:
        sources = [
            mono(tmp_path / "m.jsonl", 60),
            mt(tmp_path / "mt.jsonl", 60),
            fra_pairs(tmp_path / "f.jsonl", 60),
        ]
        corpus = tmp_path / "cpt.jsonl"
        mixture.build(sources, WordCounter(), corpus, epochs=1, rate=rate)
        result = holdout.build(sources, tmp_path / "out", rate=rate, cap=1000)

        trained = self.sides_written(corpus)
        assert trained, "the corpus must not be empty, or the test proves nothing"
        for path in result.paths:
            evaluated = set(holdout.read(path))
            assert evaluated
            assert not (evaluated & trained), path.name

    def test_the_kabyle_side_of_a_held_out_pair_also_leaves_training(self, tmp_path: Path) -> None:
        """The English side is what is evaluated, but the pair is what is withheld — the
        Kabyle sentence must not come back through the other direction's row."""
        sources = [mt(tmp_path / "mt.jsonl", 40)]
        corpus = tmp_path / "cpt.jsonl"
        mixture.build(sources, WordCounter(), corpus, epochs=1, rate=3)
        trained = self.sides_written(corpus)
        held = [f"{KAB} {i}" for i in range(40) if f"{KAB} {i}" not in trained]
        assert held
        assert all(text not in trained for text in held)

    @pytest.mark.parametrize("rate", [2, 3, 5])
    def test_a_sentence_with_several_partners_leaves_through_every_one(
        self, tmp_path: Path, rate: int
    ) -> None:
        """One English sentence against many Kabyle ones — the real corpus's shape, and the
        case a fixture of unique pairs cannot exhibit.

        Selecting on the Kabyle side alone put **274 of 932** English evaluation sentences
        back into the built corpus, each through a partner that had not been selected. The
        exclusion is per record on either side for exactly this.
        """
        path = tmp_path / "many.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i in range(60):
                for partner in range(4):
                    fh.write(
                        json.dumps(
                            {
                                "source": f"{KAB} {i}.{partner}",
                                "target": f"{ENG} {i}",
                                "direction": "kab-eng",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        source = Source(
            name="many",
            path=path,
            kind="aligned",
            fields=("source", "target"),
            direction_field="direction",
        )
        corpus = tmp_path / "cpt.jsonl"
        mixture.build([source], WordCounter(), corpus, epochs=1, rate=rate)
        result = holdout.build([source], tmp_path / "out", rate=rate, cap=1000)

        evaluated = set(holdout.read(result.paths[0]))
        assert evaluated
        assert not (evaluated & self.sides_written(corpus))
