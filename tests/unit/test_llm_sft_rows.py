"""SFT rows: whole segments, first-fit packing, the mask carried into the packed row."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agbalu.llm import sft_rows
from agbalu.llm.chat import IGNORE, Sample
from agbalu.llm.packing import IGNORE_INDEX
from agbalu.llm.sft_rows import Segment

EOS = 9


def turn(name: str, prompt: int, answer: int, *, think: bool = False) -> Segment:
    ids = tuple(range(100, 100 + prompt + answer))
    labels = (*([IGNORE] * prompt), *ids[prompt:])
    return sft_rows.from_sample(Sample(name, 1, ids, labels, think))


class TestSegments:
    def test_a_sample_keeps_its_mask_and_kind(self) -> None:
        segment = turn("c", 3, 2, think=True)
        assert segment.supervised == (False, False, False, True, True)
        assert segment.kind == sft_rows.THINK
        assert segment.source == "c#1"

    def test_a_replay_document_is_eos_first_and_scored_after_it(self) -> None:
        segment = sft_rows.from_document([5, 6, 7], EOS, "fineweb")
        assert segment.ids == (EOS, 5, 6, 7)
        assert segment.supervised == (False, True, True, True)

    @pytest.mark.parametrize(("ids", "mask"), [((1,), (True,)), ((1, 2), (True,))])
    def test_a_malformed_segment_is_refused(
        self, ids: tuple[int, ...], mask: tuple[bool, ...]
    ) -> None:
        with pytest.raises(sft_rows.RowsError):
            Segment(ids, mask, sft_rows.NO_THINK, "x")

    def test_a_segment_with_nothing_to_learn_is_refused(self) -> None:
        with pytest.raises(sft_rows.RowsError, match="nothing is supervised"):
            Segment((1, 2, 3), (True, False, False), sft_rows.NO_THINK, "x")


class TestFirstFit:
    def test_every_segment_lands_once_and_no_row_overflows(self) -> None:
        lengths = [5, 3, 8, 2, 7, 1 + 1, 4]
        plan = sft_rows.first_fit(lengths, 10, seed=1)
        assert sorted(i for row in plan for i in row) == list(range(len(lengths)))
        assert all(sum(lengths[i] for i in row) <= 10 for row in plan)

    def test_the_plan_is_seeded(self) -> None:
        lengths = [3] * 20
        assert sft_rows.first_fit(lengths, 10, seed=4) == sft_rows.first_fit(lengths, 10, seed=4)
        assert sft_rows.first_fit(lengths, 10, seed=4) != sft_rows.first_fit(lengths, 10, seed=5)

    def test_a_segment_longer_than_a_row_is_refused(self) -> None:
        with pytest.raises(sft_rows.RowsError, match="outside"):
            sft_rows.first_fit([11], 10)


class TestRows:
    def segments(self) -> list[Segment]:
        return [
            turn("a", 4, 3),
            turn("b", 2, 5, think=True),
            sft_rows.from_document([1, 2, 3, 4], EOS, "replay"),
            turn("c", 6, 2),
        ]

    def test_packing_conserves_tokens_and_supervision(self) -> None:
        segments = self.segments()
        rows = sft_rows.pack(segments, capacity=12, seed=0)
        assert len(rows.tokens) == sum(len(s.ids) for s in segments)
        assert int(rows.supervised.sum()) == sum(sum(s.supervised) for s in segments)

    def test_a_packed_row_masks_prompts_and_segment_starts(self) -> None:
        rows = sft_rows.pack(self.segments(), capacity=12, seed=0)
        for index in range(len(rows)):
            packed = rows.packed(index)
            begin, end = int(rows.row_starts[index]), int(rows.row_starts[index + 1])
            labels = packed.labels[0]
            assert ((labels != IGNORE_INDEX) <= rows.supervised[begin:end]).all()
            starts = packed.cu_seqlens[:-1]
            assert (labels[starts] == IGNORE_INDEX).all()
            assert packed.cu_seqlens[-1] == end - begin

    def test_the_trainer_target_count_matches_the_summary(self) -> None:
        rows = sft_rows.pack(self.segments(), capacity=12, seed=0)
        targets = sum(rows.packed(i).targets for i in range(len(rows)))
        assert targets == sft_rows.summary(rows, 12)["supervised_tokens"]

    def test_the_summary_reports_shares_by_kind(self) -> None:
        stats = sft_rows.summary(sft_rows.pack(self.segments(), capacity=12, seed=0), 12)
        shares = stats["supervised_share"]
        assert isinstance(shares, dict)
        assert set(shares) == {"no_think", "think", "replay"}
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)

    def test_a_round_trip_through_disk(self, tmp_path: Path) -> None:
        rows = sft_rows.pack(self.segments(), capacity=12, seed=0)
        sft_rows.write(rows, tmp_path / "sft.npz")
        again = sft_rows.Rows.read(tmp_path / "sft.npz")
        for name in sft_rows.ARRAYS:
            assert np.array_equal(getattr(rows, name), getattr(again, name))
        assert not (tmp_path / ".sft.npz.partial").exists()

    def test_a_missing_file_names_the_command_that_builds_it(self, tmp_path: Path) -> None:
        with pytest.raises(sft_rows.RowsError, match="make data TASK=sft"):
            sft_rows.Rows.read(tmp_path / "absent.npz")

    def test_a_row_starting_inside_a_segment_is_refused(self) -> None:
        rows = sft_rows.pack(self.segments(), capacity=12, seed=0)
        broken = sft_rows.Rows(
            rows.tokens, rows.supervised, rows.segment_starts[1:], rows.row_starts, rows.kinds[1:]
        )
        with pytest.raises(sft_rows.RowsError, match="inside a segment"):
            broken.check("test")

    def test_nothing_to_pack(self) -> None:
        with pytest.raises(sft_rows.RowsError, match="no segment"):
            sft_rows.pack([])
