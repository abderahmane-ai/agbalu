"""Verify padded batches retain exact supervision counts."""

import pytest
import torch

from agbalu.codec_tts.dataset import collate_cpu
from agbalu.codec_tts.layout import IGNORE, SPECIALS, prompt_layout, target_layout


def test_padding_and_reference_positions_are_masked() -> None:
    tokens = dict(zip(SPECIALS, range(1, 6), strict=True))
    first = target_layout(prompt_layout(tokens, [7]), torch.zeros((2, 8), dtype=torch.long))
    second = target_layout(prompt_layout(tokens, [8, 9]), torch.ones((4, 8), dtype=torch.long))
    text, _, _, valid, labels = collate_cpu([first, second])
    assert text.shape == (2, second.positions)
    assert labels.ne(IGNORE).sum().item() == first.targets + second.targets
    assert not valid[0, first.positions :].any()
    assert labels[0, first.positions :].eq(IGNORE).all()


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty codec microbatch"):
        collate_cpu([])
