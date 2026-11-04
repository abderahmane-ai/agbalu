"""Preserve the native final-row termination boundary."""

import torch

from agbalu.codec_tts.inference import SamplerState, Sampling, advance_sampler, sample_row
from agbalu.codec_tts.layout import BOC, EOC, reverse_delay


def test_native_wind_down_retains_all_real_frames() -> None:
    state = SamplerState()
    rows: list[torch.Tensor] = []
    frames = 12
    for position in range(frames + 8):
        codes = torch.tensor([7 if position - stream < frames else EOC for stream in range(8)])
        row = advance_sampler(codes, state)
        rows.append(row)
        if state.is_done:
            break
    assert len(rows) == frames + 7
    delayed = torch.stack(rows)
    assert delayed[0, 1:].eq(BOC).all()
    decoded = reverse_delay(delayed, frames)
    assert decoded.shape == (frames, 8)
    assert decoded.eq(7).all()


def test_greedy_sampler_does_not_consume_rng() -> None:
    generator = torch.Generator().manual_seed(3)
    before = generator.get_state().clone()
    logits = torch.zeros((8, 1026))
    logits[:, 42] = 1
    result = sample_row(logits, Sampling(temperature=0), generator)
    assert result.eq(42).all()
    assert torch.equal(before, generator.get_state())
