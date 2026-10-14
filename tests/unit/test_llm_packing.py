"""Packed rows: segment boundaries, restarting positions and the masked first labels."""

from __future__ import annotations

import numpy as np
import pytest

from agbalu.llm.packing import (
    IGNORE_INDEX,
    PackingError,
    concatenate,
    pack_documents,
    pack_row,
    plan_rows,
)

EOS = 0


class TestPackRow:
    def test_segments_start_at_each_eos_and_each_block_edge(self) -> None:
        blocks = np.array([[0, 5, 6, 0, 7, 8], [9, 9, 0, 4, 4, 4]], dtype=np.uint32)
        row = pack_row(blocks, EOS)
        assert row.cu_seqlens.tolist() == [0, 3, 6, 8, 12]
        assert row.position_ids[0].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 3]

    def test_the_label_at_every_segment_start_is_masked(self) -> None:
        blocks = np.array([[0, 5, 6, 0, 7, 8]], dtype=np.uint32)
        row = pack_row(blocks, EOS)
        assert row.labels[0].tolist() == [IGNORE_INDEX, 5, 6, IGNORE_INDEX, 7, 8]
        assert row.targets == 4

    def test_a_block_cut_mid_document_is_its_own_segment(self) -> None:
        row = pack_row(np.array([[3, 4, 5], [6, 7, 8]], dtype=np.uint32), EOS)
        assert row.cu_seqlens.tolist() == [0, 3, 6]
        assert row.labels[0].tolist()[3] == IGNORE_INDEX

    def test_max_length_is_the_longest_segment(self) -> None:
        row = pack_row(np.array([[0, 1, 2, 3, 0, 1]], dtype=np.uint32), EOS)
        assert row.max_length == 4

    @pytest.mark.parametrize("shape", [(0, 4), (2, 1), (8,)])
    def test_bad_shapes_are_refused(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(PackingError):
            pack_row(np.zeros(shape, dtype=np.uint32), EOS)


class TestPlanRows:
    def test_fills_greedily_in_order(self) -> None:
        assert plan_rows([3, 3, 3, 1], 6) == [[0, 1], [2, 3]]

    def test_an_exact_fit_shares_a_row(self) -> None:
        assert plan_rows([2, 4], 6) == [[0, 1]]

    def test_every_document_is_planned_once(self) -> None:
        lengths = [5, 1, 7, 3, 3, 8, 2]
        planned = [i for group in plan_rows(lengths, 8) for i in group]
        assert planned == list(range(len(lengths)))

    @pytest.mark.parametrize("lengths", [[0], [9]])
    def test_an_empty_or_oversized_document_is_refused(self, lengths: list[int]) -> None:
        with pytest.raises(PackingError):
            plan_rows(lengths, 8)

    def test_capacity_below_two_is_refused(self) -> None:
        with pytest.raises(PackingError):
            plan_rows([1], 1)


class TestPackDocuments:
    def test_each_document_is_prefixed_and_scores_all_its_tokens(self) -> None:
        rows = pack_documents([[5, 6], [7], [8, 9, 10]], EOS, 5)
        assert [r.input_ids[0].tolist() for r in rows] == [[0, 5, 6, 0, 7], [0, 8, 9, 10]]
        assert sum(r.targets for r in rows) == 6

    def test_a_long_document_is_truncated_to_capacity(self) -> None:
        rows = pack_documents([[1] * 10], EOS, 4)
        assert rows[0].input_ids[0].tolist() == [0, 1, 1, 1]

    def test_an_empty_document_is_refused(self) -> None:
        with pytest.raises(PackingError):
            pack_documents([[1], []], EOS, 4)

    def test_concatenate_keeps_every_token(self) -> None:
        row = concatenate([[0, 1], [0, 2, 3]])
        assert row.input_ids[0].tolist() == [0, 1, 0, 2, 3]
        assert row.cu_seqlens.tolist() == [0, 2, 5]
