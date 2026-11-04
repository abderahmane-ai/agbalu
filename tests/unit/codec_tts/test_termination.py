"""Account for the first terminal target of every codec stream."""

import pytest
import torch

from agbalu.codec_tts.layout import BOC, CODEBOOKS, EOC, IGNORE, delay_codes, validate_codes


@pytest.mark.parametrize("frames", [1, 2, 20])
def test_terminal_supervision_is_not_structural_padding(frames: int) -> None:
    delayed, labels = delay_codes(torch.ones((frames, CODEBOOKS), dtype=torch.long), terminal=True)
    assert delayed.shape == (frames + CODEBOOKS, CODEBOOKS)
    for stream in range(CODEBOOKS):
        assert delayed[:stream, stream].eq(BOC).all()
        assert labels[:, stream].eq(EOC).sum().item() == 1
        assert labels[frames + stream, stream].item() == EOC
        assert labels[:stream, stream].eq(IGNORE).all()
        assert labels[:, stream].ne(IGNORE).sum().item() == frames + 1


@pytest.mark.parametrize("value", [-1, 1024, 1025])
def test_structural_ids_are_not_raw_codec_values(value: int) -> None:
    with pytest.raises(ValueError, match="raw codec value"):
        validate_codes(torch.full((2, CODEBOOKS), value))
