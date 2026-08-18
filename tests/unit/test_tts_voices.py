"""Which speakers become candidate voices, and the two joins that can quietly go wrong.

Demographics are written per row and mostly left blank, so a first-row read reports the
male candidate as unlabelled. And the held-out check is a join: it is asserted here on a
corpus where it *must* fire, because a join that cannot match returns zero and zero reads
as a clean result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from agbalu.speech.corpus import Clip
from agbalu.tts.voices import LONG_MS, VoiceError, demographics, identify

COLUMNS: Final = ("client_id", "path", "sentence", "age", "gender")


def clip(speaker: str, name: str, duration_ms: int = 3_000, split: str = "train") -> Clip:
    return Clip(
        clip=name,
        speaker=speaker,
        split=split,
        duration_ms=duration_ms,
        text="Azul fell-ak",
        target="azul fell-ak",
        repaired=False,
    )


def write_table(root: Path, name: str, entries: list[tuple[str, str, str]]) -> None:
    """`entries` is (client_id, age, gender), one per row, written as Common Voice does."""
    root.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(COLUMNS)]
    lines += [
        "\t".join((client, f"{client}_{index}.mp3", "Azul", age, gender))
        for index, (client, age, gender) in enumerate(entries)
    ]
    (root / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestRanking:
    def test_speakers_rank_by_total_duration_not_clip_count(self, tmp_path: Path) -> None:
        train = [clip("many", f"m{i}", 1_000) for i in range(10)] + [
            clip("long", f"l{i}", 5_000) for i in range(5)
        ]
        write_table(tmp_path, "train", [])
        voices = identify(train, [], tmp_path)
        assert [voice.speaker for voice in voices] == ["long", "many"]
        assert voices[0].duration_ms == 25_000
        assert voices[0].rank == 1

    def test_a_tie_breaks_on_the_speaker_id_so_the_order_is_stable(self, tmp_path: Path) -> None:
        train = [clip("b", "b1", 4_000), clip("a", "a1", 4_000)]
        write_table(tmp_path, "train", [])
        assert [voice.speaker for voice in identify(train, [], tmp_path)] == ["a", "b"]

    def test_hours_and_the_clip_statistics_come_from_the_clips(self, tmp_path: Path) -> None:
        train = [clip("s", "c1", 2_000), clip("s", "c2", 4_000), clip("s", "c3", 6_000)]
        write_table(tmp_path, "train", [])
        voice = identify(train, [], tmp_path, top=1)[0]
        assert voice.clips == 3
        assert voice.mean_ms == 4_000
        assert voice.median_ms == 4_000
        assert voice.max_ms == 6_000
        assert voice.hours == pytest.approx(12_000 / 3_600_000)

    def test_a_clip_exactly_at_the_long_threshold_counts(self, tmp_path: Path) -> None:
        train = [clip("s", "c1", LONG_MS), clip("s", "c2", LONG_MS - 1)]
        write_table(tmp_path, "train", [])
        voice = identify(train, [], tmp_path, top=1)[0]
        assert voice.long_clips == 1

    def test_top_bounds_the_result(self, tmp_path: Path) -> None:
        train = [clip(name, f"{name}1", 1_000) for name in ("a", "b", "c")]
        write_table(tmp_path, "train", [])
        assert len(identify(train, [], tmp_path, top=2)) == 2
        assert len(identify(train, [], tmp_path, top=9)) == 3

    def test_an_empty_train_split_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(VoiceError, match="no train clips"):
            identify([], [], tmp_path)

    def test_a_top_below_one_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(VoiceError, match="at least 1"):
            identify([clip("s", "c1")], [], tmp_path, top=0)


class TestHeldOutCheck:
    def test_it_fires_on_a_speaker_that_leaked(self, tmp_path: Path) -> None:
        train = [clip("s", "c1", 9_000)]
        held_out = [clip("s", "d1", split="dev"), clip("s", "t1", split="test")]
        write_table(tmp_path, "train", [])
        assert identify(train, held_out, tmp_path, top=1)[0].other_split_clips == 2

    def test_it_is_zero_when_the_splits_are_disjoint(self, tmp_path: Path) -> None:
        train = [clip("s", "c1", 9_000)]
        write_table(tmp_path, "train", [])
        voice = identify(train, [clip("other", "d1", split="dev")], tmp_path, top=1)[0]
        assert voice.other_split_clips == 0


class TestDemographics:
    def test_a_label_on_a_minority_of_rows_still_resolves(self, tmp_path: Path) -> None:
        """The corpus case: the second candidate carries `male_masculine` on 5,428 of its
        16,002 train rows and nothing on the other 10,574."""
        write_table(
            tmp_path,
            "train",
            [("s", "", "")] * 11 + [("s", "fifties", "male_masculine")] * 5,
        )
        resolved = demographics(tmp_path, frozenset({"s"}))["s"]
        assert resolved.gender == "male_masculine"
        assert resolved.age == "fifties"
        assert resolved.labelled_rows == 5
        assert resolved.total_rows == 16

    def test_the_majority_wins_when_the_labels_disagree(self, tmp_path: Path) -> None:
        write_table(
            tmp_path,
            "train",
            [("s", "", "female_feminine")] * 3 + [("s", "", "male_masculine")],
        )
        assert demographics(tmp_path, frozenset({"s"}))["s"].gender == "female_feminine"

    def test_a_tie_resolves_the_same_way_every_run(self, tmp_path: Path) -> None:
        write_table(tmp_path, "train", [("s", "", "male_masculine"), ("s", "", "female_feminine")])
        first = demographics(tmp_path, frozenset({"s"}))["s"].gender
        assert first == "female_feminine"
        assert demographics(tmp_path, frozenset({"s"}))["s"].gender == first

    def test_labels_are_gathered_across_every_table(self, tmp_path: Path) -> None:
        write_table(tmp_path, "train", [("s", "", "")])
        write_table(tmp_path, "validated", [("s", "thirties", "female_feminine")])
        resolved = demographics(tmp_path, frozenset({"s"}))["s"]
        assert resolved.gender == "female_feminine"
        assert resolved.total_rows == 2

    def test_a_speaker_labelled_nowhere_reports_absence_not_a_guess(self, tmp_path: Path) -> None:
        write_table(tmp_path, "train", [("s", "", "")] * 4)
        resolved = demographics(tmp_path, frozenset({"s"}))["s"]
        assert resolved.gender == ""
        assert resolved.age == ""
        assert resolved.labelled_rows == 0
        assert resolved.total_rows == 4

    def test_other_speakers_rows_are_not_counted(self, tmp_path: Path) -> None:
        write_table(tmp_path, "train", [("s", "", "male_masculine"), ("t", "", "female_feminine")])
        resolved = demographics(tmp_path, frozenset({"s"}))["s"]
        assert resolved.gender == "male_masculine"
        assert resolved.total_rows == 1

    def test_missing_tables_are_not_fatal(self, tmp_path: Path) -> None:
        resolved = demographics(tmp_path, frozenset({"s"}))["s"]
        assert resolved.total_rows == 0

    def test_the_payload_carries_the_label_counts(self, tmp_path: Path) -> None:
        write_table(tmp_path, "train", [("s", "", "")] * 3 + [("s", "", "male_masculine")])
        payload = identify([clip("s", "c1")], [], tmp_path, top=1)[0].as_dict()
        assert payload["gender"] == "male_masculine"
        assert payload["labelled_rows"] == 1
        assert payload["total_rows"] == 4
