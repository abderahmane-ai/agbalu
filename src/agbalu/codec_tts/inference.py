"""Generate native delayed codec streams with isolated request state."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from pydantic import Field
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.configuration import Specification
from agbalu.codec_tts.layout import BOC, CODEBOOKS, EOC, SequenceLayout, reverse_delay

MAX_CONTEXT_POSITIONS = 8192
COLLAPSE_FRAMES = 25


class GenerationError(RuntimeError):
    def __init__(self, message: str, trace: dict[str, object]) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass
class LoopState:
    is_recovering: bool = False


class Sampling(Specification):
    temperature: float = Field(default=0.8, ge=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=30, ge=0)
    max_new_positions: int = Field(default=1500, ge=8)
    seed: int = Field(default=20261102, ge=0)


class Decoding(Specification):
    alignment_strength: float = Field(default=0, ge=0, le=3)
    plausible_tokens: int = Field(default=25, ge=1, le=1026)
    recover_loops: bool = False


def alignment_logits(expert: Tensor, amateur: Tensor, settings: Decoding) -> Tensor:
    if expert.shape != amateur.shape:
        message = f"ECCD logit shapes differ: {tuple(expert.shape)}, {tuple(amateur.shape)}"
        raise ValueError(message)
    if settings.alignment_strength == 0:
        return expert
    positive = expert.float().log_softmax(-1)
    negative = amateur.float().log_softmax(-1)
    indices = positive.topk(min(settings.plausible_tokens, positive.shape[-1]), dim=-1).indices
    head = torch.zeros_like(positive, dtype=torch.bool).scatter_(-1, indices, True)
    compatibility = negative.exp().masked_fill(~head, 0).sum(-1, keepdim=True)
    enhancement = (positive - negative).clamp_min(0).masked_fill(~head, 0)
    return positive + settings.alignment_strength * (1 - compatibility) * enhancement


def recover_repetition(
    logits: Tensor,
    row: Tensor,
    history: Tensor,
    generator: torch.Generator,
    state: LoopState | None = None,
) -> tuple[Tensor, bool]:
    # Repeating codec values is normal. Recover only an observed one-second collapse.
    collapsed = history.numel() >= COLLAPSE_FRAMES and bool(
        history[-COLLAPSE_FRAMES:].eq(row[0]).all()
    )
    if state is not None:
        state.is_recovering |= collapsed
        collapsed = state.is_recovering and bool(history[-10:].eq(row[0]).any())
        if not collapsed:
            state.is_recovering = False
    if not collapsed or int(row[0]) >= BOC:
        return row, False
    alternative = logits[0].float().clone()
    alternative[row[0]] = float("-inf")
    result = row.clone()
    result[0] = torch.multinomial(alternative.softmax(-1), 1, generator=generator)[0]
    return result, True


@dataclass
class SamplerState:
    delay_count: int = 0
    eoc_countdown: int | None = None
    is_done: bool = False


def sample_row(logits: Tensor, settings: Sampling, generator: torch.Generator) -> Tensor:
    if settings.temperature == 0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / settings.temperature
    if settings.top_k:
        threshold = scaled.topk(min(settings.top_k, scaled.shape[-1]), dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    ordered, indices = scaled.sort(descending=True, dim=-1)
    remove = ordered.softmax(dim=-1).cumsum(dim=-1) > settings.top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    mask = torch.zeros_like(remove).scatter(-1, indices, remove)
    probabilities = scaled.masked_fill(mask, float("-inf")).softmax(dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def advance_sampler(codes: Tensor, state: SamplerState) -> Tensor:
    result = codes.clone()
    if state.delay_count < CODEBOOKS:
        state.delay_count += 1
        result[state.delay_count :] = BOC
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        state.is_done = state.eoc_countdown <= 0
    elif int(result[0].item()) == EOC:
        state.eoc_countdown = CODEBOOKS - 2
    return result


def unpack_output(output: object) -> tuple[Tensor, Cache]:
    if not isinstance(output, BaseModelOutputWithPast):
        message = f"invalid decoder output: {type(output).__name__}"
        raise TypeError(message)
    hidden, cache = output.last_hidden_state, output.past_key_values
    if not isinstance(hidden, Tensor) or not isinstance(cache, Cache):
        message = "cached generation requires hidden states and an explicit KV cache"
        raise TypeError(message)
    return hidden[:, -1], cache


@torch.inference_mode()
def generate_codes(
    model: Generator,
    prompt: SequenceLayout,
    settings: Sampling,
    decoding: Decoding | None = None,
    contrast_prompt: SequenceLayout | None = None,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[Tensor, dict[str, object]]:
    decoding = decoding or Decoding()
    if prompt.positions + settings.max_new_positions > MAX_CONTEXT_POSITIONS:
        message = f"generation context exceeds {MAX_CONTEXT_POSITIONS}: prompt={prompt.positions}"
        message += f", budget={settings.max_new_positions}"
        raise ValueError(message)
    device = model.audio_embedding.weight.device
    if decoding.alignment_strength and contrast_prompt is None:
        message = "ECCD requires a text-ablated contrast prompt"
        raise ValueError(message)
    previous_mode = model.training
    model.train(False)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return decode_stream(
                model, prompt, settings, decoding, contrast_prompt, generator, deadline_monotonic
            )
    finally:
        model.train(previous_mode)


def prefill(model: Generator, prompt: SequenceLayout) -> tuple[Tensor, Cache]:
    device = model.audio_embedding.weight.device
    ids = prompt.text_ids.to(device).unsqueeze(0)
    codes = prompt.audio_codes.to(device).unsqueeze(0)
    mask = prompt.audio_mask.to(device).reshape(1, -1, 1)
    text: Tensor = model.body.embed_tokens(ids)
    embeddings = torch.where(mask, model.embed_audio(codes), text)
    return unpack_output(model.body(inputs_embeds=embeddings, use_cache=True))


def decode_stream(
    model: Generator,
    prompt: SequenceLayout,
    settings: Sampling,
    decoding: Decoding,
    contrast_prompt: SequenceLayout | None,
    generator: torch.Generator,
    deadline_monotonic: float | None,
) -> tuple[Tensor, dict[str, object]]:
    device = model.audio_embedding.weight.device
    hidden, cache = prefill(model, prompt)
    contrast = (
        prefill(model, contrast_prompt) if decoding.alignment_strength and contrast_prompt else None
    )
    state = SamplerState()
    rows = torch.empty((settings.max_new_positions, CODEBOOKS), device=device, dtype=torch.long)
    replacements = 0
    loop_state = LoopState()
    count = 0
    for position in range(settings.max_new_positions):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            message = f"speech deadline exceeded at generated position={position}"
            raise TimeoutError(message)
        logits = model.classify(hidden)[0]
        if contrast is not None:
            # Guide the coarse stream; preserve the native residual-codebook distributions.
            guided = alignment_logits(logits[:1], model.classify(contrast[0])[0, :1], decoding)
            logits = torch.cat((guided, logits[1:]), dim=0)
        sampled = sample_row(logits, settings, generator)
        if decoding.recover_loops:
            sampled, recovered = recover_repetition(
                logits, sampled, rows[:position, 0], generator, loop_state
            )
            replacements += int(recovered)
        row = advance_sampler(sampled, state)
        rows[position] = row
        count = position + 1
        if state.is_done:
            break
        embedded = model.embed_audio(row.reshape(1, 1, -1))
        hidden, cache = unpack_output(
            model.body(inputs_embeds=embedded, past_key_values=cache, use_cache=True)
        )
        if contrast is not None:
            contrast = unpack_output(
                model.body(inputs_embeds=embedded, past_key_values=contrast[1], use_cache=True)
            )
    if not state.is_done:
        message = f"generation hit hard cap={settings.max_new_positions} without native termination"
        raise GenerationError(
            message,
            {
                "terminated": False,
                "trace": rows[:count].cpu().tolist(),
                "generated_positions": count,
                "sampling": settings.model_dump(),
                "decoding": decoding.model_dump(),
                "repetition_resamples": replacements,
            },
        )
    delayed = rows[:count].cpu()
    first_eoc = delayed[:, 0].eq(EOC).nonzero()
    if first_eoc.numel() == 0:
        message = "native termination has no first-stream EOC"
        raise RuntimeError(message)
    frames = int(first_eoc[0, 0].item())
    needed = frames + CODEBOOKS - 1
    if delayed.shape[0] < needed:
        message = f"incomplete delayed wind-down: rows={delayed.shape[0]}, needed={needed}"
        raise RuntimeError(message)
    result = reverse_delay(delayed[:needed], frames)
    return result, {
        "terminated": True,
        "frames": frames,
        "trace": delayed.tolist(),
        "sampling": settings.model_dump(),
        "decoding": decoding.model_dump(),
        "repetition_resamples": replacements,
        "generated_positions": count,
    }
