"""A packed row reaches the model with its boundaries intact."""

from __future__ import annotations

import numpy as np
import torch

from agbalu.llm.forward import hidden_states, to_device
from agbalu.llm.packing import pack_row


class Recorder:
    """A backbone that records its keyword arguments and returns one-hot hidden states."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def __call__(self, **kwargs: object) -> Recorder:
        self.seen = kwargs
        ids = kwargs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        self.last_hidden_state = torch.nn.functional.one_hot(ids, 8).float()
        return self


def test_to_device_keeps_dtypes_and_values() -> None:
    row = pack_row(np.array([[0, 1, 2, 0, 3]], dtype=np.uint32), 0)
    moved = to_device(row, torch.device("cpu"))
    assert moved.input_ids.dtype == torch.long
    assert moved.cu_seqlens.dtype == torch.int32
    assert moved.cu_seqlens.tolist() == [0, 3, 5]
    assert moved.labels.tolist() == row.labels.tolist()
    assert moved.targets == row.targets


def test_hidden_states_passes_the_same_boundaries_for_queries_and_keys() -> None:
    row = to_device(pack_row(np.array([[0, 1, 2, 0, 3]], dtype=np.uint32), 0), torch.device("cpu"))
    recorder = Recorder()
    hidden = hidden_states(recorder, row)
    assert hidden.shape == (1, 5, 8)
    assert recorder.seen["cu_seq_lens_q"] is recorder.seen["cu_seq_lens_k"]
    assert recorder.seen["max_length_q"] == recorder.seen["max_length_k"] == 3
