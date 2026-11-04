"""Check codec loss against an analytic uniform distribution."""

import math

import pytest
import torch

from agbalu.codec_tts.layout import CODEBOOKS, IGNORE, VOCABULARY
from agbalu.codec_tts.objective import codec_loss


def test_uniform_loss_and_mask_counts() -> None:
    labels = torch.zeros((1, 3, CODEBOOKS), dtype=torch.long)
    labels[:, 0, :4] = IGNORE
    result = codec_loss(torch.zeros((1, 3, CODEBOOKS, VOCABULARY)), labels)
    assert result.count.item() == 20
    assert result.stream_counts.tolist() == [2, 2, 2, 2, 3, 3, 3, 3]
    assert result.numerator.item() == pytest.approx(20 * math.log(VOCABULARY))


def test_empty_supervision_fails() -> None:
    with pytest.raises(ValueError, match="zero supervised"):
        codec_loss(torch.zeros((1, CODEBOOKS, VOCABULARY)), torch.full((1, CODEBOOKS), IGNORE))
