"""Verify native prompt order and exactly one teacher-forcing shift."""

import torch

from agbalu.codec_tts.layout import (
    CODEBOOKS,
    IGNORE,
    SPECIALS,
    delay_codes,
    prompt_layout,
    reverse_delay,
    target_layout,
)


def test_reference_prompt_order() -> None:
    codes = torch.arange(16).reshape(2, CODEBOOKS)
    tokens = dict(zip(SPECIALS, range(1, 6), strict=True))
    prompt = prompt_layout(tokens, [9, 10], codes, [8])
    assert prompt.text_ids[:4].tolist() == [1, 2, 8, 3]
    assert prompt.text_ids[-4:].tolist() == [4, 9, 10, 5]
    assert prompt.audio_mask.sum().item() == 9
    assert prompt.labels.eq(IGNORE).all()
    assert torch.equal(reverse_delay(delay_codes(codes)[0], 2), codes)


def test_final_prompt_state_predicts_first_audio_target() -> None:
    prompt = prompt_layout(dict(zip(SPECIALS, range(1, 6), strict=True)), [7])
    codes = torch.full((3, CODEBOOKS), 17)
    layout = target_layout(prompt, codes)
    assert layout.labels[prompt.positions - 1, 0].item() == 17
    assert layout.labels[: prompt.positions - 1].eq(IGNORE).all()
    assert layout.audio_codes[prompt.positions, 0].item() == 17
    assert not layout.audio_mask[prompt.positions - 1]
    assert layout.targets == CODEBOOKS * 4
