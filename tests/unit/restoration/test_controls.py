"""Exercise conservation and numerical metric sensitivity controls."""

import pytest
import torch

from agbalu.restoration.dataset import Example, collate
from agbalu.restoration.decoding import render
from agbalu.restoration.evaluation import confusion, macro
from agbalu.restoration.model import Restorer


def test_changing_a_neighboring_batch_row_cannot_change_a_prediction(
    model: Restorer, example: Example
) -> None:
    batch = collate([example, example], 64)
    with torch.inference_mode():
        original = model.forward(batch).punctuation[1].clone()
        batch.ids[0, 1:5] = 32
        changed = model.forward(batch).punctuation[1]
    torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-6)


def test_corrupt_word_alignment_does_not_pass_as_a_valid_output() -> None:
    with pytest.raises(ValueError, match="cover"):
        render(("azul", "fell-awen"), (0, 2), (0, 0), ("@", "@"))


def test_no_insertion_control_cannot_claim_punctuation_quality() -> None:
    gold = torch.tensor([0, 0, 1, 2, 3])
    matrix = confusion(gold, torch.zeros_like(gold), 4)
    assert macro(matrix, start=1) == 0
