"""Pivot selection and the two-teacher agreement filter.

The filter is the whole quality argument for synthetic data: the Kabyle side is human, so
the only thing that can be wrong is the machine-made source. Two independent teachers
agreeing is the evidence; one teacher is a guess.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.mt.pivot import (
    PivotRecord,
    Policy,
    combine,
    filter_language,
    read,
    select,
    usable,
    write,
    write_pairs,
)


class FakeIdentifier:
    """Labels by lookup, defaulting to a wrong-but-plausible neighbour."""

    def __init__(self, labels: dict[str, str], default: str = "spa_Latn") -> None:
        self.labels = labels
        self.default = default
        self.calls = 0

    def identify(self, texts: Sequence[str]) -> list[str]:
        self.calls += 1
        return [self.labels.get(text, self.default) for text in texts]


KAB = "azul fell-awen ay imdukkal"
ENG = "hello to you my friends"
FRA = "bonjour a vous mes amis"


def _row(kab: str, other: str, language: str, source: str = "opus.tatoeba-kab") -> str:
    return json.dumps(
        {"kab": kab, language: other, "source": source, "defects": []}, ensure_ascii=False
    )


def _corpus(directory: Path, eng: list[str], fra: list[str]) -> Path:
    (directory / "agbalu-parallel-v1.kab-eng.jsonl").write_text("\n".join(eng), encoding="utf-8")
    (directory / "agbalu-parallel-v1.kab-fra.jsonl").write_text("\n".join(fra), encoding="utf-8")
    return directory


class TestFilterLanguage:
    def test_the_claimed_language_survives(self) -> None:
        texts = {"a": "the cat sat", "b": "le chat"}
        identifier = FakeIdentifier({"the cat sat": "eng_Latn", "le chat": "fra_Latn"})
        kept, dropped = filter_language(texts, "eng", identifier)
        assert kept == {"a": "the cat sat"}
        assert dropped == 1

    def test_a_latin_script_neighbour_is_caught(self) -> None:
        """The corpus's own detector misses Spanish sitting in the French field; this is
        the stage that exists to catch it."""
        texts = {"a": "Tom dijo que nunca comió suchi."}
        kept, dropped = filter_language(texts, "fra", FakeIdentifier({}))
        assert kept == {}
        assert dropped == 1

    def test_it_batches_rather_than_calling_per_string(self) -> None:
        texts = {str(i): "le chat" for i in range(25)}
        identifier = FakeIdentifier({"le chat": "fra_Latn"})
        kept, _ = filter_language(texts, "fra", identifier, batch=10)
        assert len(kept) == 25
        assert identifier.calls == 3

    def test_an_empty_input_makes_no_call(self) -> None:
        identifier = FakeIdentifier({})
        assert filter_language({}, "eng", identifier) == ({}, 0)
        assert identifier.calls == 0


class TestSelect:
    def test_a_sentence_with_both_sides_gets_two_teachers(self, tmp_path: Path) -> None:
        _corpus(tmp_path, [_row(KAB, ENG, "eng")], [_row(KAB, FRA, "fra")])
        identifier = FakeIdentifier({ENG: "eng_Latn", FRA: "fra_Latn"})
        records, stats = select(identifier, tmp_path)
        assert [r.teachers for r in records] == [2]
        assert stats["two_teacher"] == 1
        assert stats["kept"] == 1

    def test_a_sentence_with_one_side_is_kept_with_one_teacher(self, tmp_path: Path) -> None:
        _corpus(tmp_path, [_row(KAB, ENG, "eng")], [])
        records, stats = select(FakeIdentifier({ENG: "eng_Latn"}), tmp_path)
        assert [(r.teachers, r.fra) for r in records] == [(1, None)]
        assert stats["eng_only"] == 1

    def test_mined_rows_are_excluded(self, tmp_path: Path) -> None:
        """NLLB's own output would teach the teacher its own guesses back."""
        _corpus(tmp_path, [_row(KAB, ENG, "eng", source="opus.nllb-kab")], [])
        records, stats = select(FakeIdentifier({ENG: "eng_Latn"}), tmp_path)
        assert records == []
        assert stats["mined_excluded"] == 1

    def test_a_hard_defect_is_excluded(self, tmp_path: Path) -> None:
        row = json.dumps(
            {"kab": "azul", "eng": "", "source": "opus.tatoeba-kab", "defects": ["empty"]}
        )
        _corpus(tmp_path, [row], [])
        records, stats = select(FakeIdentifier({}), tmp_path)
        assert records == []
        assert stats["hard_defective"] == 1

    def test_a_side_that_fails_identification_is_dropped(self, tmp_path: Path) -> None:
        _corpus(tmp_path, [_row(KAB, "hola mi amigo querido", "eng")], [_row(KAB, FRA, "fra")])
        identifier = FakeIdentifier({FRA: "fra_Latn"})
        records, stats = select(identifier, tmp_path)
        assert [(r.teachers, r.eng) for r in records] == [(1, None)]
        assert stats["wrong_language"]["eng"] == 1


class TestUsable:
    """Localisation misalignment is invisible to both the corpus's defect labels and the
    agreement filter, so it has to be caught here or not at all."""

    def test_prose_survives(self) -> None:
        assert usable(PivotRecord(kab="azul fell-awen a yimdukkal", eng="hello my friends"))

    def test_a_printf_placeholder_is_refused(self) -> None:
        assert not usable(PivotRecord(kab="!! tuccḍa: %s", eng="an error occurred here"))

    def test_a_bare_identifier_is_refused(self) -> None:
        """The pair that exposed this: Kabyle prose against an SPDX licence id."""
        assert not usable(PivotRecord(kab="tuccḍa tella deg umahil", eng="GPL-2.0-or-later"))

    def test_markup_is_refused(self) -> None:
        assert not usable(PivotRecord(kab="ldi tawwurt tamaynut", eng="open <b>new</b> door"))

    def test_a_very_short_side_is_refused(self) -> None:
        assert not usable(PivotRecord(kab="azul fell-awen a yimdukkal", eng="ok then"))

    def test_every_present_side_is_checked(self) -> None:
        record = PivotRecord(kab="azul fell-awen a yimdukkal", eng="hello my friends", fra="%d")
        assert not usable(record)

    def test_an_absent_side_is_not_checked(self) -> None:
        assert usable(PivotRecord(kab="azul fell-awen a yimdukkal", eng="hello my friends"))


class TestCombine:
    def _records(self, n: int = 1) -> list[PivotRecord]:
        return [PivotRecord(kab=f"kab {i}", eng=f"en {i}", fra=f"fr {i}") for i in range(n)]

    def test_agreeing_teachers_are_kept_with_their_score(self) -> None:
        pairs, stats = combine(
            self._records(),
            {0: "مرحبا"},
            {0: "مرحبا"},
            Policy("arb_Arab", 50.0),
            lambda a, b: 100.0 if a == b else 0.0,
        )
        assert [p.teachers for p in pairs] == [2]
        assert pairs[0].agreement == 100.0
        assert stats["agreed"] == 1
        assert stats["disagreed"] == 0

    def test_disagreeing_teachers_are_dropped_not_demoted(self) -> None:
        """A disagreement means one teacher is wrong and nothing says which."""
        pairs, stats = combine(
            self._records(),
            {0: "one thing"},
            {0: "something else"},
            Policy("arb_Arab", 50.0),
            lambda _a, _b: 0.0,
        )
        assert pairs == []
        assert stats["disagreed"] == 1
        assert stats["kept"] == 0

    def test_a_single_teacher_is_kept_without_a_score(self) -> None:
        records = [PivotRecord(kab="kab", eng="en", fra=None)]
        pairs, stats = combine(
            records, {0: "solo"}, {}, Policy("spa_Latn", 50.0), lambda _a, _b: 0.0
        )
        assert [(p.teachers, p.agreement, p.source) for p in pairs] == [(1, None, "solo")]
        assert stats["single_teacher"] == 1

    def test_single_teachers_can_be_refused(self) -> None:
        records = [PivotRecord(kab="kab", eng="en", fra=None)]
        pairs, stats = combine(
            records,
            {0: "solo"},
            {},
            Policy("spa_Latn", 50.0, keep_single=False),
            lambda _a, _b: 0.0,
        )
        assert pairs == []
        assert stats["single_teacher"] == 0

    def test_the_french_teacher_wins_a_tie(self) -> None:
        pairs, _ = combine(
            self._records(),
            {0: "from english"},
            {0: "from french"},
            Policy("arb_Arab", 0.0),
            lambda _a, _b: 100.0,
        )
        assert pairs[0].source == "from french"

    def test_a_record_neither_teacher_translated_is_skipped(self) -> None:
        pairs, stats = combine(
            self._records(), {}, {}, Policy("arb_Arab", 50.0), lambda _a, _b: 0.0
        )
        assert pairs == []
        assert stats["two_teacher"] == 0
        assert stats["mean_agreement"] is None

    def test_the_target_is_always_the_human_kabyle(self) -> None:
        """Synthetic source, authentic target — the whole point of the arrangement."""
        pairs, _ = combine(
            self._records(),
            {0: "machine made"},
            {0: "machine made"},
            Policy("arb_Arab", 50.0),
            lambda _a, _b: 100.0,
        )
        assert pairs[0].kab == "kab 0"

    @pytest.mark.parametrize("threshold", [0.0, 49.9, 50.0])
    def test_the_threshold_is_inclusive(self, threshold: float) -> None:
        pairs, _ = combine(
            self._records(),
            {0: "a"},
            {0: "b"},
            Policy("arb_Arab", threshold),
            lambda _a, _b: 50.0,
        )
        assert len(pairs) == 1


class TestRoundTrip:
    def test_records_survive_write_and_read(self, tmp_path: Path) -> None:
        records = [
            PivotRecord(kab="azul", eng="hello", fra="bonjour"),
            PivotRecord(kab="tanemmirt", eng=None, fra="merci"),
        ]
        path = tmp_path / "pivot.jsonl"
        write(records, path)
        assert read(path) == records

    def test_pairs_are_written_with_their_provenance(self, tmp_path: Path) -> None:
        pairs, _ = combine(
            [PivotRecord(kab="azul", eng="hello", fra="bonjour")],
            {0: "مرحبا"},
            {0: "مرحبا"},
            Policy("arb_Arab", 50.0),
            lambda _a, _b: 87.5,
        )
        path = tmp_path / "kab-arb_Arab.jsonl"
        write_pairs(pairs, path)
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row == {
            "source": "مرحبا",
            "target": "azul",
            "language": "arb_Arab",
            "teachers": 2,
            "agreement": 87.5,
        }
