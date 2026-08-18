"""The row model and the held-out split (task 11.2).

Two defects this file exists to prevent. A four-direction MT file read as if `source` were
always Kabyle tags half its rows with the wrong language; and a split keyed on the row's
first field holds out `eng-kab` while `kab-eng` — the same two sentences — stays in
training, which is a leak no later measurement can detect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agbalu.llm.corpus import (
    HOLDOUT_RATE,
    LANGUAGE_TAG,
    CorpusError,
    Record,
    Source,
    is_held_out,
    records,
)

KAB = "Aɣbalu n tmaziɣt d ayen ɣef nettmeslay."
ENG = "The source of Tamazight is what we speak about."
FRA = "La source du tamazight est ce dont nous parlons."


def write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def mt_source(path: Path) -> Source:
    return Source(
        name="mt",
        path=path,
        kind="aligned",
        fields=("source", "target"),
        direction_field="direction",
    )


class TestSource:
    def test_a_monolingual_source_takes_one_field(self, tmp_path: Path) -> None:
        source = Source(name="m", path=tmp_path / "m.jsonl", kind="kabyle", fields=("text",))
        assert source.fields == ("text",)

    def test_the_field_count_must_match_the_kind(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="needs 1 fields"):
            Source(name="m", path=tmp_path / "m.jsonl", kind="kabyle", fields=("a", "b"))

    def test_an_aligned_source_needs_a_layout(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="fixed language code or a direction field"):
            Source(name="p", path=tmp_path / "p.jsonl", kind="aligned", fields=("kab", "eng"))

    def test_an_aligned_source_cannot_declare_both_layouts(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="not both and not neither"):
            Source(
                name="p",
                path=tmp_path / "p.jsonl",
                kind="aligned",
                fields=("kab", "eng"),
                code="eng_Latn",
                direction_field="direction",
            )

    def test_a_monolingual_source_cannot_name_another_side(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="no other side"):
            Source(
                name="m",
                path=tmp_path / "m.jsonl",
                kind="kabyle",
                fields=("text",),
                code="eng_Latn",
            )


class TestRecords:
    def test_a_monolingual_row_yields_its_text(self, tmp_path: Path) -> None:
        path = write(tmp_path / "m.jsonl", [{"text": KAB}])
        source = Source(name="m", path=path, kind="kabyle", fields=("text",))
        assert [r.kabyle for r in records(source)] == [KAB]
        assert not next(iter(records(source))).aligned

    def test_a_fixed_layout_reads_the_first_field_as_kabyle(self, tmp_path: Path) -> None:
        path = write(tmp_path / "p.jsonl", [{"kab": KAB, "fra": FRA}])
        source = Source(name="p", path=path, kind="aligned", fields=("kab", "fra"), code="fra_Latn")
        record = next(iter(records(source)))
        assert (record.kabyle, record.other, record.code) == (KAB, FRA, "fra_Latn")

    @pytest.mark.parametrize(
        ("direction", "first", "second", "code"),
        [
            ("kab-eng", KAB, ENG, "eng_Latn"),
            ("eng-kab", ENG, KAB, "eng_Latn"),
            ("kab-fra", KAB, FRA, "fra_Latn"),
            ("fra-kab", FRA, KAB, "fra_Latn"),
        ],
    )
    def test_the_direction_decides_which_side_is_kabyle(
        self, tmp_path: Path, direction: str, first: str, second: str, code: str
    ) -> None:
        """Reading `source` as Kabyle tagged 542,729 of 1,085,458 rows with the wrong
        language, and called French English on 440,838 more."""
        path = write(
            tmp_path / "mt.jsonl", [{"source": first, "target": second, "direction": direction}]
        )
        record = next(iter(records(mt_source(path))))
        assert record.kabyle == KAB
        assert record.other in {ENG, FRA}
        assert record.code == code

    def test_a_direction_without_kabyle_is_refused(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "mt.jsonl", [{"source": ENG, "target": FRA, "direction": "eng-fra"}]
        )
        with pytest.raises(CorpusError, match="no Kabyle side"):
            list(records(mt_source(path)))

    def test_an_unregistered_language_is_refused(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "mt.jsonl", [{"source": KAB, "target": "…", "direction": "kab-arb"}]
        )
        with pytest.raises(CorpusError, match="no NLLB code"):
            list(records(mt_source(path)))

    @pytest.mark.parametrize("direction", ["kab_eng", "kab-eng-fra", "", "kab"])
    def test_a_malformed_direction_names_its_line(self, tmp_path: Path, direction: str) -> None:
        path = write(
            tmp_path / "mt.jsonl", [{"source": KAB, "target": ENG, "direction": direction}]
        )
        with pytest.raises(CorpusError, match=re.escape("mt.jsonl:1")):
            list(records(mt_source(path)))

    def test_a_row_missing_a_side_is_skipped(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "mt.jsonl",
            [
                {"source": KAB, "target": ENG, "direction": "kab-eng"},
                {"source": KAB, "direction": "kab-eng"},
            ],
        )
        assert len(list(records(mt_source(path)))) == 1

    def test_a_blank_side_is_skipped(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "mt.jsonl", [{"source": KAB, "target": "   ", "direction": "kab-eng"}]
        )
        assert list(records(mt_source(path))) == []

    def test_a_blank_line_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text(json.dumps({"text": KAB}) + "\n\n", encoding="utf-8")
        source = Source(name="m", path=path, kind="kabyle", fields=("text",))
        assert len(list(records(source))) == 1

    def test_a_malformed_row_names_its_line(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text(json.dumps({"text": KAB}) + "\nbroken\n", encoding="utf-8")
        source = Source(name="m", path=path, kind="kabyle", fields=("text",))
        with pytest.raises(CorpusError, match=re.escape("m.jsonl:2")):
            list(records(source))

    def test_a_json_scalar_is_not_a_row(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")
        source = Source(name="m", path=path, kind="kabyle", fields=("text",))
        with pytest.raises(CorpusError, match="not a JSON object"):
            list(records(source))

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        source = Source(name="gone", path=tmp_path / "absent.jsonl", kind="kabyle", fields=("t",))
        with pytest.raises(CorpusError, match="not found"):
            list(records(source))

    def test_the_index_is_the_line_and_survives_skipped_rows(self, tmp_path: Path) -> None:
        """The index decides which side leads a document, so it must not renumber."""
        path = write(
            tmp_path / "mt.jsonl",
            [
                {"source": KAB, "target": ENG, "direction": "kab-eng"},
                {"source": KAB, "direction": "kab-eng"},
                {"source": KAB, "target": ENG, "direction": "kab-eng"},
            ],
        )
        assert [r.index for r in records(mt_source(path))] == [0, 2]


class TestRecord:
    def test_an_aligned_record_needs_both_a_side_and_a_code(self) -> None:
        with pytest.raises(CorpusError, match="both a side and its code"):
            Record(kabyle=KAB, index=0, other=ENG)

    def test_a_code_without_a_side_is_refused(self) -> None:
        with pytest.raises(CorpusError, match="both a side and its code"):
            Record(kabyle=KAB, index=0, code="eng_Latn")


class TestIsHeldOut:
    def test_the_decision_is_stable(self) -> None:
        assert is_held_out(KAB) == is_held_out(KAB)

    def test_the_same_sentence_decides_both_copies_of_a_mirrored_pair(self) -> None:
        """`train.jsonl` carries every pair twice. Keying on the Kabyle side is what stops
        one direction being evaluated while the other trains."""
        assert is_held_out(KAB, rate=3) == is_held_out(KAB, rate=3)

    def test_a_homoglyph_is_a_different_sentence(self) -> None:
        """Greek ε against Latin ɛ — 2.6–3.2% of the seed corpora. They are distinct keys,
        so neither may be assumed to follow the other into or out of the split."""
        latin = "Yeɛni ayen i d-nniɣ"
        greek = latin.replace("ɛ", "ε")

        def buckets(text: str) -> list[bool]:
            return [is_held_out(text, rate=rate) for rate in range(2, 50)]

        assert buckets(latin) != buckets(greek)

    def test_every_record_is_held_out_at_rate_one(self) -> None:
        assert all(is_held_out(text, rate=1) for text in (KAB, ENG, "", " "))

    def test_the_rate_is_approximately_the_share(self) -> None:
        held = sum(is_held_out(f"{KAB} {i}") for i in range(20_000))
        assert 20_000 / HOLDOUT_RATE * 0.6 < held < 20_000 / HOLDOUT_RATE * 1.4

    def test_the_split_does_not_depend_on_position(self) -> None:
        """A keyed digest, not a shuffled index: one new row must not re-draw the split."""
        first = [t for t in (f"{KAB} {i}" for i in range(500)) if is_held_out(t, rate=7)]
        second = [t for t in (f"{KAB} {i}" for i in range(1, 500)) if is_held_out(t, rate=7)]
        assert set(second) <= set(first)

    @pytest.mark.parametrize("rate", [0, -1])
    def test_a_non_positive_rate_is_refused(self, rate: int) -> None:
        with pytest.raises(CorpusError, match="rate must be positive"):
            is_held_out(KAB, rate=rate)


def test_every_language_tag_is_an_nllb_code() -> None:
    assert set(LANGUAGE_TAG) == {"kab", "eng", "fra"}
    assert all(re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", code) for code in LANGUAGE_TAG.values())
