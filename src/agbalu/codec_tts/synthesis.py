"""Produce one reference-conditioned utterance for any reference language."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from pydantic import Field, model_validator
from torch import Tensor

from agbalu.codec_tts.adaptation import scaled_adapter
from agbalu.codec_tts.artifacts import atomic_binary, digest_file, provenance, run_lock, write_json
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.codec import Codec
from agbalu.codec_tts.configuration import Specification
from agbalu.codec_tts.dataset import TextTokenizer
from agbalu.codec_tts.inference import (
    MAX_CONTEXT_POSITIONS,
    Decoding,
    GenerationError,
    Sampling,
    generate_codes,
)
from agbalu.codec_tts.layout import SPECIALS, SequenceLayout, prompt_layout
from agbalu.codec_tts.verification import Quality, spoken_text
from agbalu.codec_tts.waveforms import Waveform, measure_signal, resample_waveform

DEFAULT_INFERENCE = Path("resources/codec_tts/inference.json")
STEREO_AXES = 2


class InferenceSettings(Specification):
    version: int = Field(default=1, ge=1, le=1)
    sampling: Sampling = Sampling()
    decoding: Decoding = Decoding()
    adapter_multiplier: float = Field(default=1, ge=0, le=2)
    candidates: int = Field(default=4, ge=1, le=16)
    max_segment_characters: int = Field(default=160, ge=32, le=1000)
    max_cer: float = Field(default=0.2, ge=0, le=1)
    content_tolerance: float = Field(default=0.08, ge=0, le=1)
    min_voice_cosine: float = Field(default=0.21693290024995804, ge=-1, le=1)
    min_predicted_mos: float = Field(default=3.5, ge=1, le=5)
    quality_tolerance: float = Field(default=0.25, ge=0, le=4)
    timeout_s: float = Field(default=840, gt=0)
    strategies: tuple[GenerationStrategy, ...] = ()
    max_duration_ratio: float = Field(default=3, ge=1)
    min_generation_s: float = Field(default=10, gt=0)
    fallback_candidates: int = Field(default=2, ge=0, le=4)

    @model_validator(mode="after")
    def validate_strategy_budget(self) -> InferenceSettings:
        if len(self.strategies) > self.candidates:
            message = "candidate budget must cover every configured decoding strategy"
            raise ValueError(message)
        return self


class GenerationStrategy(Specification):
    adapter_multiplier: float = Field(ge=0, le=2)
    decoding: Decoding


class SpeechRequest(Specification):
    text: str = Field(min_length=1)
    reference: Path
    reference_text: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Candidate(Specification):
    request_sha256: str
    seed: int
    waveform_sha256: str | None = None
    quality: Quality | None = None
    error: str | None = None
    trace: dict[str, object] = Field(default_factory=dict)
    adapter_multiplier: float = Field(default=1, ge=0, le=2)

    @model_validator(mode="after")
    def validate_result(self) -> Candidate:
        if self.error is None and (
            self.quality is None
            or self.waveform_sha256 is None
            or self.trace.get("terminated") is not True
        ):
            message = "a successful candidate requires verified, natively terminated audio"
            raise ValueError(message)
        return self


class Verifier(Protocol):
    def assess(self, waveform: Waveform, text: str, reference: Waveform) -> Quality: ...


def rejection_reasons(quality: Quality, settings: InferenceSettings) -> tuple[str, ...]:
    reasons: list[str] = []
    if quality.cer > settings.max_cer:
        reasons.append(f"CER={quality.cer:.6f} exceeds {settings.max_cer:.6f}")
    if quality.voice_cosine < settings.min_voice_cosine:
        reasons.append(
            f"voice_cosine={quality.voice_cosine:.6f} below {settings.min_voice_cosine:.6f}"
        )
    if quality.predicted_mos < settings.min_predicted_mos:
        reasons.append(
            f"predicted_MOS={quality.predicted_mos:.6f} below {settings.min_predicted_mos:.6f}"
        )
    if not quality.hypothesis:
        reasons.append("empty ASR transcript")
    return tuple(reasons)


def choose_candidate(values: Sequence[Candidate], settings: InferenceSettings) -> int | None:
    admitted = [
        (index, value.quality)
        for index, value in enumerate(values)
        if value.error is None
        and value.quality is not None
        and not rejection_reasons(value.quality, settings)
    ]
    if not admitted:
        return None
    best_cer = min(quality.cer for _, quality in admitted)
    accurate = [
        (index, quality)
        for index, quality in admitted
        if quality.cer <= best_cer + settings.content_tolerance
    ]
    best_mos = max(quality.predicted_mos for _, quality in accurate)
    natural = [
        (index, quality)
        for index, quality in accurate
        if quality.predicted_mos >= best_mos - settings.quality_tolerance
    ]
    return max(
        natural, key=lambda row: (row[1].voice_cosine, -row[1].cer, row[1].predicted_mos, -row[0])
    )[0]


def split_text(text: str, limit: int) -> tuple[str, ...]:
    spoken_text(text)
    if limit < 1:
        message = f"invalid segment limit={limit}"
        raise ValueError(message)
    tags = tuple(re.finditer(r"<\|[^<>]*\|>", text))
    result: list[str] = []
    start = 0
    while start < len(text):
        stop = min(start + limit, len(text))
        if stop < len(text):
            containing = next((tag for tag in tags if tag.start() < stop < tag.end()), None)
            if containing is not None:
                stop = containing.start()
            fragment = text[start:stop]
            boundaries = [match.end() for match in re.finditer(r"[.!?;:]\s+", fragment)]
            if not boundaries:
                boundaries = [
                    match.end()
                    for match in re.finditer(r"\s+", fragment)
                    if not any(tag.start() < start + match.start() < tag.end() for tag in tags)
                ]
            if not boundaries:
                message = f"unsplittable speech token at character={start}, limit={limit}"
                raise ValueError(message)
            stop = start + boundaries[-1]
        part = text[start:stop]
        if not part.strip() and result:
            result[-1] += part
        else:
            result.append(part)
        start = stop
    if "".join(result) != text:
        message = "speech text conservation failed"
        raise AssertionError(message)
    return tuple(result)


def read_waveform(path: Path) -> Waveform:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim == STEREO_AXES:
        audio = audio.mean(axis=1)
    return resample_waveform(np.asarray(audio, dtype=np.float32), int(rate))


def save_waveform(path: Path, waveform: Waveform) -> None:
    measure_signal(waveform, 24000)
    with atomic_binary(path) as stream:
        sf.write(stream, waveform, 24000, format="WAV", subtype="FLOAT")


def trim_edges(audio: Waveform) -> Waveform:
    """Drop only the silence before the first sound and after the last."""
    rate = 24000
    frame = int(0.02 * rate)
    hop = frame // 2
    if audio.size <= frame:
        return audio
    count = 1 + (audio.size - frame) // hop
    rms = np.empty(count, dtype=np.float32)
    for index, start in enumerate(range(0, audio.size - frame + 1, hop)):
        chunk = audio[start : start + frame]
        rms[index] = float(np.sqrt(np.mean(np.square(chunk))))
    threshold = max(0.005, float(np.percentile(rms, 90)) * 0.08)
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return audio
    margin = int(0.04 * rate)
    first = max(0, int(active[0] * hop) - margin)
    last = min(int(audio.size), int((active[-1] + 1) * hop + frame) + margin)
    if last - first < frame:
        return audio
    return np.asarray(audio[first:last], dtype=np.float32)


def close_gaps(audio: Waveform) -> Waveform:
    """Shorten an internal silence longer than 320 ms to 140 ms."""
    rate = 24000
    frame = int(0.02 * rate)
    hop = frame // 2
    if audio.size <= frame:
        return audio
    count = 1 + (audio.size - frame) // hop
    rms = np.empty(count, dtype=np.float32)
    for index, start in enumerate(range(0, audio.size - frame + 1, hop)):
        chunk = audio[start : start + frame]
        rms[index] = float(np.sqrt(np.mean(np.square(chunk))))
    threshold = max(0.005, float(np.percentile(rms, 90)) * 0.08)
    silent = rms <= threshold
    keep = int(0.14 * rate)
    minimum = int(0.32 * rate)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < silent.size:
        if not bool(silent[index]):
            index += 1
            continue
        start_index = index
        while index < silent.size and bool(silent[index]):
            index += 1
        start = int(start_index * hop)
        end = min(int(audio.size), int((index - 1) * hop) + frame)
        if start > hop and end < audio.size - hop and end - start >= minimum:
            spans.append((start, end))
    if not spans:
        return np.asarray(audio, dtype=np.float32)
    pieces: list[Waveform] = []
    cursor = 0
    for start, end in spans:
        pieces.append(np.asarray(audio[cursor:start], dtype=np.float32))
        pieces.append(np.asarray(audio[start : start + keep], dtype=np.float32))
        cursor = end
    pieces.append(np.asarray(audio[cursor:], dtype=np.float32))
    return join_segments(pieces)


def join_segments(waves: Sequence[Waveform]) -> Waveform:
    if not waves:
        message = "speech assembly received no audio"
        raise ValueError(message)
    fade = int(0.01 * 24000)
    joined = np.asarray(waves[0], dtype=np.float32)
    for nxt in waves[1:]:
        following = np.asarray(nxt, dtype=np.float32)
        width = min(fade, joined.size // 2, following.size // 2)
        if width < 2:
            joined = np.concatenate((joined, following))
            continue
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
        mixed = joined[-width:] * (1.0 - ramp) + following[:width] * ramp
        joined = np.concatenate((joined[:-width], mixed, following[width:]))
    return np.asarray(joined, dtype=np.float32)


def prompt_fits(
    tokenizer: TextTokenizer,
    specials: Mapping[str, int],
    text: str,
    reference_codes: Tensor,
    reference_ids: Sequence[int],
    max_new_positions: int,
) -> bool:
    prompt = prompt_layout(
        specials, tokenizer.encode(text, add_special_tokens=False), reference_codes, reference_ids
    )
    remaining = MAX_CONTEXT_POSITIONS - prompt.positions
    return remaining >= 8 and min(max_new_positions, remaining) >= 8


def partition_text(
    tokenizer: TextTokenizer,
    specials: Mapping[str, int],
    text: str,
    reference_codes: Tensor,
    reference_ids: Sequence[int],
    max_new_positions: int,
) -> tuple[str, ...]:
    if prompt_fits(tokenizer, specials, text, reference_codes, reference_ids, max_new_positions):
        return (text,)
    parts = split_text(text, max(32, len(text) // 2))
    if len(parts) == 1:
        message = "speech text does not fit the context and has no split point"
        raise ValueError(message)
    return tuple(
        piece
        for part in parts
        for piece in partition_text(
            tokenizer, specials, part, reference_codes, reference_ids, max_new_positions
        )
    )


def speech_identity(
    request: SpeechRequest, settings: InferenceSettings, reference_sha256: str
) -> str:
    payload = {
        "request": request.model_dump(mode="json"),
        "sampling": settings.sampling.model_dump(),
        "decoding": settings.decoding.model_dump(),
        "adapter_multiplier": settings.adapter_multiplier,
        "reference_sha256": reference_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def render_speech(
    model: Generator,
    codec: Codec,
    tokenizer: TextTokenizer,
    request: SpeechRequest,
    settings: InferenceSettings,
    output: Path,
    provenance_stamp: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Sample one utterance. The reference language does not change the decode."""
    started = time.monotonic()
    spoken_text(request.reference_text)
    specials = tokenizer.get_added_vocab()
    for tag in re.findall(r"<\|[^<>]*\|>", request.text):
        if tag not in specials or re.fullmatch(
            r"<\|(?:emotion|style|prosody|sfx):[a-z_]+\|>", tag
        ) is None:
            message = f"unsupported delivery tag {tag!r}"
            raise ValueError(message)
    reference_sha = digest_file(request.reference)
    identity = speech_identity(request, settings, reference_sha)
    audit = output.with_suffix(".stats.json")
    with run_lock(output.parent):
        if audit.is_file():
            previous = json.loads(audit.read_bytes())
            if previous.get("identity") == identity and previous.get("complete") is True:
                if not output.is_file() or digest_file(output) != previous.get("output_sha256"):
                    message = f"checksum mismatch at {output}"
                    raise ValueError(message)
                return dict(previous)
            if previous.get("identity") not in (None, identity):
                message = f"different speech request already occupies {output}"
                raise FileExistsError(message)
        reference = read_waveform(request.reference)
        if measure_signal(reference, 24000).rms_dbfs is None:
            message = f"reference audio is silent: {request.reference}"
            raise ValueError(message)
        reference_codes = codec.encode(reference)
        reference_ids = tokenizer.encode(request.reference_text, add_special_tokens=False)
        parts = partition_text(
            tokenizer,
            specials,
            request.text,
            reference_codes,
            reference_ids,
            settings.sampling.max_new_positions,
        )
        prefix = re.match(
            r"(?:\s*<\|(?:emotion|style|prosody):(?!pause\|>|long_pause\|>)[^<>]+\|>)*",
            request.text,
        )
        delivery = prefix.group() if prefix else ""
        decoding = settings.decoding.model_copy(
            update={"alignment_strength": 0, "recover_loops": True}
        )
        waves: list[Waveform] = []
        segments: list[dict[str, object]] = []
        for number, text in enumerate(parts):
            conditioning = (delivery if number else "") + text
            prompt = prompt_layout(
                specials,
                tokenizer.encode(conditioning, add_special_tokens=False),
                reference_codes,
                reference_ids,
            )
            remaining = MAX_CONTEXT_POSITIONS - prompt.positions
            positions = min(settings.sampling.max_new_positions, remaining)
            sampling = settings.sampling.model_copy(
                update={"max_new_positions": positions, "seed": settings.sampling.seed + number}
            )
            with scaled_adapter(model, settings.adapter_multiplier):
                codes, trace = generate_codes(model, prompt, sampling, decoding, None)
            waves.append(np.asarray(codec.decode(codes).numpy(), dtype=np.float32))
            segments.append({"text": text, "seed": sampling.seed, **trace})
        joined = close_gaps(trim_edges(join_segments(waves)))
        peak = measure_signal(joined, 24000).peak
        gain = min(1.0, 0.99 / peak) if peak else 1.0
        save_waveform(output, np.asarray(joined * gain, dtype=np.float32))
        report: dict[str, object] = {
            **(
                {**provenance_stamp, "config_hash": identity}
                if provenance_stamp is not None
                else provenance(identity)
            ),
            "identity": identity,
            "complete": True,
            "segments": segments,
            "samples": joined.size,
            "sample_rate_hz": 24000,
            "duration_s": joined.size / 24000,
            "peak_gain": gain,
            "output": str(output),
            "output_sha256": digest_file(output),
            "elapsed_s": time.monotonic() - started,
            "adapter_multiplier": settings.adapter_multiplier,
            "sampling": settings.sampling.model_dump(),
            "decoding": decoding.model_dump(),
        }
        write_json(audit, report)
    return report


def contrast_layout(prompt: SequenceLayout, specials: Mapping[str, int]) -> SequenceLayout:
    keep = prompt.audio_mask.clone()
    for token in SPECIALS:
        keep |= prompt.text_ids.eq(specials[token])
    return SequenceLayout(
        prompt.text_ids[keep],
        prompt.audio_codes[keep],
        prompt.audio_mask[keep],
        prompt.labels[keep],
    )


def candidate_settings(
    settings: InferenceSettings,
    text: str,
    reference_s: float,
    reference_characters: int,
    index: int,
) -> InferenceSettings:
    if reference_characters < 1:
        message = f"reference_characters must be positive, got {reference_characters}"
        raise ValueError(message)
    estimate_s = reference_s * len(spoken_text(text)) / reference_characters
    limit_s = max(settings.min_generation_s, settings.max_duration_ratio * estimate_s)
    positions = min(settings.sampling.max_new_positions, math.ceil(limit_s * 25) + 7)
    if "<|prosody:" in text or "<|style:" in text:
        positions = settings.sampling.max_new_positions
    active = settings.model_copy(
        update={"sampling": settings.sampling.model_copy(update={"max_new_positions": positions})}
    )
    if settings.strategies:
        strategy = settings.strategies[index % len(settings.strategies)]
        active = active.model_copy(
            update={
                "adapter_multiplier": strategy.adapter_multiplier,
                "decoding": strategy.decoding,
            }
        )
    if index >= settings.candidates:
        strength = (
            settings.strategies[
                (index - settings.candidates) % len(settings.strategies)
            ].adapter_multiplier
            if settings.strategies
            else settings.adapter_multiplier
        )
        active = active.model_copy(
            update={"adapter_multiplier": strength, "decoding": Decoding(recover_loops=True)}
        )
    return active


class SpeechSynthesis:
    def __init__(
        self,
        model: Generator,
        codec: Codec,
        tokenizer: TextTokenizer,
        verifier: Verifier,
        provenance_stamp: Mapping[str, object] | None = None,
    ) -> None:
        self.model, self.codec, self.tokenizer, self.verifier = model, codec, tokenizer, verifier
        self.provenance_stamp = provenance_stamp

    def candidate(
        self,
        text: str,
        prompt: SequenceLayout,
        reference: Waveform,
        settings: InferenceSettings,
        seed: int,
        path: Path,
        identity: str,
        deadline: float,
    ) -> tuple[Candidate, bool]:
        metadata = path.with_suffix(".stats.json")
        if metadata.is_file():
            previous = Candidate.model_validate_json(metadata.read_bytes())
            if previous.request_sha256 != identity or previous.seed != seed:
                message = f"candidate identity mismatch at {metadata}"
                raise ValueError(message)
            if previous.waveform_sha256 and (
                not path.is_file() or digest_file(path) != previous.waveform_sha256
            ):
                message = f"candidate checksum mismatch at {path}"
                raise ValueError(message)
            if previous.error is None or not previous.error.startswith("TimeoutError:"):
                return previous, True
        sampling = settings.sampling.model_copy(update={"seed": seed})
        try:
            if time.monotonic() >= deadline:
                message = f"speech request deadline exceeded before {path}"
                raise TimeoutError(message)
            with scaled_adapter(self.model, settings.adapter_multiplier):
                codes, trace = generate_codes(
                    self.model,
                    prompt,
                    sampling,
                    settings.decoding,
                    contrast_layout(prompt, self.tokenizer.get_added_vocab()),
                    deadline_monotonic=deadline,
                )
            wave = np.asarray(self.codec.decode(codes).numpy(), dtype=np.float32)
            save_waveform(path, wave)
            quality = self.verifier.assess(wave, text, reference)
            value = Candidate(
                request_sha256=identity,
                seed=seed,
                waveform_sha256=digest_file(path),
                quality=quality,
                trace=trace,
                adapter_multiplier=settings.adapter_multiplier,
            )
        except (RuntimeError, ValueError, TimeoutError) as error:
            value = Candidate(
                request_sha256=identity,
                seed=seed,
                error=f"{type(error).__name__}: {error}",
                adapter_multiplier=settings.adapter_multiplier,
                trace=error.trace
                if isinstance(error, GenerationError)
                else {"decoding": settings.decoding.model_dump()},
            )
        write_json(metadata, value.model_dump(mode="json"))
        return value, False

    def synthesize(
        self, request: SpeechRequest, settings: InferenceSettings, output: Path
    ) -> dict[str, object]:
        return render_speech(
            self.model,
            self.codec,
            self.tokenizer,
            request,
            settings,
            output,
            provenance_stamp=self.provenance_stamp,
        )

