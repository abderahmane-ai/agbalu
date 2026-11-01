"""Verify exact tensor alignment and padding masks."""

import numpy as np
import torch

from agbalu.restoration.dataset import Example, collate, collate_update


def test_collation_preserves_spans_and_masks_padding(example: Example) -> None:
    batch = collate([example], 64)
    assert batch.ids[0, :6].tolist() == [2, 6, 7, 8, 9, 3]
    assert batch.spans[0, :3].tolist() == [[1, 3], [3, 4], [4, 5]]
    assert int(batch.attention.sum()) == 6
    assert batch.marks[0, :4].tolist() == [0, 1, 0, 2]
    assert not bool(batch.words[0, 3:].any())
    assert bool((batch.casing[0, 3:] == -100).all())


def test_unequal_rows_do_not_gain_supervision_from_padding(example: Example) -> None:
    short = Example(
        np.asarray([2, 8, 3], np.int64),
        np.asarray([[1, 2]], np.int64),
        np.asarray([0, 3], np.int64),
        np.asarray([1], np.int64),
        np.asarray([0, 0], np.int64),
        (True, True),
    )
    batch = collate([example, short], 64).to(torch.device("cpu"))
    assert batch.marks[1, :2].tolist() == [0, 3]
    assert bool((batch.marks[1, 2:] == -100).all())
    assert batch.ids.size(0) == 1
    assert batch.ids[0, :9].tolist() == [2, 6, 7, 8, 9, 3, 2, 8, 3]
    assert batch.documents[0, :9].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert batch.positions[0, :9].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2]
    assert batch.spans[1, 0].tolist() == [7, 8]


def test_packing_accounts_for_every_token_across_blocks(example: Example) -> None:
    batch = collate([example] * 100, 64)
    assert batch.ids.shape == (2, 512)
    assert int(batch.attention.sum()) == 600
    for index in range(100):
        valid = batch.documents == index
        assert batch.ids[valid].tolist() == example.ids.tolist()
        assert batch.positions[valid].tolist() == list(range(6))


def test_long_updates_keep_a_fixed_gpu_token_bound() -> None:
    row = Example(
        np.asarray([2, *([6] * 62), 3], np.int64),
        np.asarray([[1, 63]], np.int64),
        np.asarray([0, 2], np.int64),
        np.asarray([0], np.int64),
        np.asarray([0, 0], np.int64),
        (True, True),
    )
    batches = collate_update([row] * 32, 64, 32)
    assert len(batches) == 4
    assert all(batch.ids.shape == (1, 512) for batch in batches)
    assert sum(batch.words.size(0) for batch in batches) == 32
    assert sum(int(batch.attention.sum()) for batch in batches) == 2048
