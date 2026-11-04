"""Construct native codec prompts and shifted target masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

CODEBOOKS: Final = 8
CODE_AXES: Final = 2
CODE_VALUES: Final = 1024
VOCABULARY: Final = 1026
BOC: Final = 1024
EOC: Final = 1025
IGNORE: Final = -100
SPECIALS: Final = ("<|tts|>", "<|ref_text|>", "<|ref_audio|>", "<|text|>", "<|audio|>")


@dataclass(frozen=True)
class SequenceLayout:
    text_ids: Tensor
    audio_codes: Tensor
    audio_mask: Tensor
    labels: Tensor

    @property
    def positions(self) -> int:
        return self.text_ids.numel()

    @property
    def targets(self) -> int:
        return int(self.labels.ne(IGNORE).sum().item())


def validate_codes(codes: Tensor) -> None:
    if codes.ndim != CODE_AXES or codes.shape[1] != CODEBOOKS or codes.shape[0] == 0:
        message = f"codec must have nonempty [frames,{CODEBOOKS}] shape, got {tuple(codes.shape)}"
        raise ValueError(message)
    if codes.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint16):
        message = f"codec must be integer, got {codes.dtype}"
        raise TypeError(message)
    values = codes.to(torch.long)
    if bool(((values < 0) | (values >= CODE_VALUES)).any().item()):
        bad = values[(values < 0) | (values >= CODE_VALUES)][0].item()
        message = f"raw codec value={bad} outside [0,{CODE_VALUES})"
        raise ValueError(message)


def delay_codes(codes: Tensor, *, terminal: bool = False) -> tuple[Tensor, Tensor]:
    validate_codes(codes)
    frames = codes.shape[0]
    rows = frames + CODEBOOKS - 1 + int(terminal)
    delayed = torch.full((rows, CODEBOOKS), EOC, dtype=torch.long, device=codes.device)
    labels = torch.full_like(delayed, IGNORE)
    for stream in range(CODEBOOKS):
        delayed[:stream, stream] = BOC
        delayed[stream : stream + frames, stream] = codes[:, stream].long()
        labels[stream : stream + frames, stream] = codes[:, stream].long()
        if terminal:
            labels[stream + frames, stream] = EOC
    return delayed, labels


def reverse_delay(delayed: Tensor, frames: int) -> Tensor:
    if frames <= 0 or delayed.shape != (frames + CODEBOOKS - 1, CODEBOOKS):
        message = f"invalid delayed shape={tuple(delayed.shape)} for frames={frames}"
        raise ValueError(message)
    result = torch.stack([delayed[c : c + frames, c] for c in range(CODEBOOKS)], dim=1)
    validate_codes(result)
    return result


def prompt_layout(
    tokens: Mapping[str, int],
    text_ids: Sequence[int],
    reference_codes: Tensor | None = None,
    reference_text_ids: Sequence[int] = (),
) -> SequenceLayout:
    missing = set(SPECIALS) - tokens.keys()
    if missing or not text_ids:
        message = f"invalid prompt: missing specials={sorted(missing)}, text length={len(text_ids)}"
        raise ValueError(message)
    if any(token < 0 for token in (*tokens.values(), *text_ids, *reference_text_ids)):
        message = "prompt token IDs must be nonnegative"
        raise ValueError(message)
    prefix = [tokens["<|tts|>"]]
    if reference_codes is not None and reference_text_ids:
        prefix += [tokens["<|ref_text|>"], *reference_text_ids]
    reference = delay_codes(reference_codes)[0] if reference_codes is not None else None
    start = 0
    if reference is not None:
        prefix.append(tokens["<|ref_audio|>"])
        start = len(prefix)
        prefix.extend([0] * reference.shape[0])
    prefix += [tokens["<|text|>"], *text_ids, tokens["<|audio|>"]]
    codes = torch.zeros((len(prefix), CODEBOOKS), dtype=torch.long)
    mask = torch.zeros(len(prefix), dtype=torch.bool)
    if reference is not None:
        codes[start : start + reference.shape[0]] = reference.cpu()
        mask[start : start + reference.shape[0]] = True
    return SequenceLayout(
        torch.tensor(prefix, dtype=torch.long), codes, mask, torch.full_like(codes, IGNORE)
    )


def target_layout(
    prompt: SequenceLayout, target: Tensor, max_positions: int = 8192
) -> SequenceLayout:
    delayed, supervision = delay_codes(target, terminal=True)
    rows = delayed.shape[0] - 1
    labels = torch.full((prompt.positions + rows, CODEBOOKS), IGNORE, dtype=torch.long)
    labels[prompt.positions - 1 :] = supervision.cpu()
    positions = prompt.positions + rows
    if positions > max_positions:
        message = f"mixed positions={positions} exceeds max_positions={max_positions}"
        raise ValueError(message)
    return SequenceLayout(
        torch.cat((prompt.text_ids, torch.zeros(rows, dtype=torch.long))),
        torch.cat((prompt.audio_codes, delayed[:-1].cpu())),
        torch.cat((prompt.audio_mask, torch.ones(rows, dtype=torch.bool))),
        labels,
    )
