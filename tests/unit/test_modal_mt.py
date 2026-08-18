"""The MT collator.

`Seq2SeqTrainer` was left to build `decoder_input_ids` and stopped doing so: transformers
5 dropped `prepare_decoder_input_ids_from_labels` from the M2M100 classes, and label
smoothing pops `labels` before the model can shift them itself. The run died at step 0 on
a message naming the opposite defect, so what the collator emits is pinned here.
"""

from __future__ import annotations

import torch
from modal_app.mt import Collator, shift_right

PAD = 1
START = 2


class FakeCollate:
    """The base collator's output: padded ids and -100-padded labels, nothing else."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __call__(self, _features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return dict(self.batch)


class TestShiftRight:
    def test_labels_move_one_place_right_behind_the_start_token(self) -> None:
        labels = torch.tensor([[10, 11, 12]])
        assert shift_right(labels, PAD, START).tolist() == [[START, 10, 11]]

    def test_label_padding_becomes_pad(self) -> None:
        """-100 is the loss's ignore index; the decoder embedding cannot index it."""
        labels = torch.tensor([[10, -100, -100]])
        assert shift_right(labels, PAD, START).tolist() == [[START, 10, PAD]]

    def test_a_single_column_is_only_the_start_token(self) -> None:
        assert shift_right(torch.tensor([[10]]), PAD, START).tolist() == [[START]]

    def test_rows_are_shifted_independently(self) -> None:
        labels = torch.tensor([[10, 11], [-100, 12]])
        assert shift_right(labels, PAD, START).tolist() == [[START, 10], [START, PAD]]

    def test_the_labels_are_not_modified(self) -> None:
        labels = torch.tensor([[10, 11, -100]])
        shift_right(labels, PAD, START)
        assert labels.tolist() == [[10, 11, -100]]


class TestCollator:
    def test_decoder_inputs_are_added(self) -> None:
        inner = FakeCollate(
            {
                "input_ids": torch.tensor([[5, 6]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "labels": torch.tensor([[10, 11, -100]]),
            }
        )
        batch = Collator(inner, PAD, START)([])
        assert batch["decoder_input_ids"].tolist() == [[START, 10, 11]]

    def test_everything_else_is_passed_through(self) -> None:
        inner = FakeCollate({"input_ids": torch.tensor([[5]]), "labels": torch.tensor([[10]])})
        batch = Collator(inner, PAD, START)([])
        assert batch["input_ids"].tolist() == [[5]]
        assert set(batch) == {"input_ids", "labels", "decoder_input_ids"}

    def test_a_collator_that_already_built_them_is_left_alone(self) -> None:
        """A future transformers may restore them; overwriting would hide the change."""
        supplied = torch.tensor([[9, 9]])
        inner = FakeCollate({"labels": torch.tensor([[10, 11]]), "decoder_input_ids": supplied})
        batch = Collator(inner, PAD, START)([])
        assert batch["decoder_input_ids"].tolist() == supplied.tolist()

    def test_the_caller_s_batch_is_not_mutated(self) -> None:
        inner = FakeCollate({"labels": torch.tensor([[10]])})
        Collator(inner, PAD, START)([])
        assert "decoder_input_ids" not in inner.batch
