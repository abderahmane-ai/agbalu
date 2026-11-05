"""Preserve the native final-row termination boundary."""

import pytest
import torch

from agbalu.codec_tts.inference import (
    Decoding,
    SamplerState,
    Sampling,
    advance_sampler,
    alignment_logits,
    recover_repetition,
    sample_row,
)
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


def test_eccd_preserves_expert_when_disabled_and_matching_distributions() -> None:
    expert = torch.tensor([[1.0, 0.0, -1.0]])
    assert alignment_logits(expert, expert, Decoding()) is expert
    result = alignment_logits(expert, expert + 3, Decoding(alignment_strength=1))
    torch.testing.assert_close(result.softmax(-1), expert.softmax(-1))


def test_eccd_enhances_only_plausible_positive_text_evidence() -> None:
    expert = torch.tensor([[2.0, 1.0, 0.0]])
    amateur = torch.tensor([[0.0, 1.0, 2.0]])
    baseline = expert.log_softmax(-1)
    result = alignment_logits(expert, amateur, Decoding(alignment_strength=1, plausible_tokens=1))
    assert result[0, 0] > baseline[0, 0]
    assert torch.equal(result[0, 1:], baseline[0, 1:])
    with pytest.raises(ValueError, match="shapes differ"):
        alignment_logits(expert, amateur[:, :2], Decoding(alignment_strength=1))


def test_loop_recovery_preserves_rng_until_collapse_and_other_streams() -> None:
    generator = torch.Generator().manual_seed(5)
    logits = torch.zeros(8, 1026)
    row = torch.full((8,), 42)
    before = generator.get_state().clone()
    normal, changed = recover_repetition(logits, row, torch.full((24,), 42), generator)
    assert normal is row
    assert not changed
    assert torch.equal(generator.get_state(), before)
    recovered, changed = recover_repetition(logits, row, torch.full((25,), 42), generator)
    assert changed
    assert recovered[0] != 42
    assert torch.equal(recovered[1:], row[1:])
    end, changed = recover_repetition(
        logits, torch.full((8,), EOC), torch.full((25,), EOC), generator
    )
    assert not changed
    assert end.eq(EOC).all()
