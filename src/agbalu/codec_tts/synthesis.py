"""Produce verified reference-conditioned speech with resumable candidate selection."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from pydantic import Field, model_validator

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
from agbalu.codec_tts.verification import INSTRUMENTS, Quality, spoken_text
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
        started = time.monotonic()
        spoken_text(request.reference_text)
        specials = self.tokenizer.get_added_vocab()
        for tag in re.findall(r"<\|[^<>]*\|>", request.text):
            if (
                tag not in specials
                or re.fullmatch(r"<\|(?:emotion|style|prosody|sfx):[a-z_]+\|>", tag) is None
            ):
                message = f"unsupported delivery tag {tag!r}"
                raise ValueError(message)
        parts = split_text(request.text, settings.max_segment_characters)
        source = {
            name: digest_file(Path(__file__).with_name(name))
            for name in (
                "synthesis.py",
                "inference.py",
                "adaptation.py",
                "verification.py",
                "layout.py",
            )
        }
        identity_data = {
            "request": request.model_dump(mode="json"),
            "settings": settings.model_dump(),
            "reference_sha256": digest_file(request.reference),
            "source": source,
        }
        identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
        root, audit = output.with_suffix(".candidates"), output.with_suffix(".stats.json")
        with run_lock(root):
            marker = root / "request.json"
            if marker.exists() and json.loads(marker.read_bytes()).get("sha256") != identity:
                message = f"different speech request already occupies {root}"
                raise FileExistsError(message)
            if not marker.exists() and (output.exists() or audit.exists()):
                message = f"refusing to replace existing speech output {output}"
                raise FileExistsError(message)
            write_json(marker, {"sha256": identity})
            reference = read_waveform(request.reference)
            if measure_signal(reference, 24000).rms_dbfs is None:
                message = f"reference audio is silent: {request.reference}"
                raise ValueError(message)
            ref_codes = self.codec.encode(reference)
            ref_ids = self.tokenizer.encode(request.reference_text, add_special_tokens=False)
            report: dict[str, object] = {
                **(
                    {**self.provenance_stamp, "config_hash": identity}
                    if self.provenance_stamp is not None
                    else provenance(identity)
                ),
                **identity_data,
                "complete": False,
                "attempts_this_run": 0,
                "cached_attempts_this_run": 0,
                "segments": [],
                "instruments": INSTRUMENTS,
                "output": str(output),
                "audit": str(audit),
            }
            segment_reports: list[dict[str, object]] = []
            report["segments"] = segment_reports
            write_json(audit, report)
            waves: list[Waveform] = []
            generated, cached, offset = 0, 0, 0
            prefix = re.match(
                r"(?:\s*<\|(?:emotion|style|prosody):(?!pause\|>|long_pause\|>)[^<>]+\|>)*",
                request.text,
            )
            delivery = prefix.group() if prefix else ""
            for number, text in enumerate(parts):
                conditioning = (delivery if number else "") + text
                prompt = prompt_layout(
                    specials,
                    self.tokenizer.encode(conditioning, add_special_tokens=False),
                    ref_codes,
                    ref_ids,
                )
                budget = MAX_CONTEXT_POSITIONS - prompt.positions
                if settings.sampling.max_new_positions > budget:
                    message = f"reference/text exceeds context budget at character={offset}"
                    message += f": remaining={budget}"
                    raise ValueError(message)
                values: list[Candidate] = []
                paths: list[Path] = []
                segment: dict[str, object] = {
                    "text": text,
                    "char_start": offset,
                    "char_end": offset + len(text),
                    "candidates": [],
                    "selected": None,
                }
                segment_reports.append(segment)
                for index in range(settings.candidates + settings.fallback_candidates):
                    if (
                        index >= settings.candidates
                        and choose_candidate(values, settings) is not None
                    ):
                        break
                    active = candidate_settings(
                        settings,
                        text,
                        reference.size / 24000,
                        len(spoken_text(request.reference_text)),
                        index,
                    )
                    initial = values[: len(settings.strategies)]
                    if (
                        settings.strategies
                        and len(initial) == len(settings.strategies)
                        and all(value.trace.get("terminated") is False for value in initial)
                    ):
                        active = active.model_copy(
                            update={"decoding": Decoding(recover_loops=True)}
                        )
                    seed_index = index // max(1, len(settings.strategies))
                    seed = (
                        settings.sampling.seed
                        + number * (settings.candidates + settings.fallback_candidates)
                        + seed_index
                    )
                    path = root / f"segment-{number:04d}-candidate-{index:02d}.wav"
                    report.update(
                        active_candidate=str(path),
                        attempts_this_run=generated,
                        cached_attempts_this_run=cached,
                    )
                    write_json(audit, report)
                    value, reused = self.candidate(
                        text,
                        prompt,
                        reference,
                        active,
                        seed,
                        path,
                        identity,
                        started + settings.timeout_s,
                    )
                    cached += int(reused)
                    generated += int(not reused)
                    logging.getLogger(__name__).info(
                        json.dumps(
                            {
                                "event": "speech_candidate",
                                "segment": number,
                                "candidate": index,
                                "cached": reused,
                                "adapter_multiplier": active.adapter_multiplier,
                                "alignment_strength": active.decoding.alignment_strength,
                                "recover_loops": active.decoding.recover_loops,
                                "quality": value.quality.model_dump() if value.quality else None,
                                "error": value.error,
                                "elapsed_s": time.monotonic() - started,
                            }
                        )
                    )
                    values.append(value)
                    paths.append(path)
                    segment["candidates"] = [
                        {
                            **item.model_dump(mode="json"),
                            "rejected": rejection_reasons(item.quality, settings)
                            if item.quality
                            else (),
                        }
                        for item in values
                    ]
                    report.update(attempts_this_run=generated, cached_attempts_this_run=cached)
                    write_json(audit, report)
                chosen = choose_candidate(values, settings)
                segment["selected"] = chosen
                write_json(audit, report)
                if chosen is None:
                    message = (
                        f"no candidate passed content, voice and quality limits at segment={number}"
                    )
                    message += f"; see {audit}"
                    raise RuntimeError(message)
                waves.append(read_waveform(paths[chosen]))
                offset += len(text)
            joined = np.concatenate(waves).astype(np.float32)
            if offset != len(request.text) or joined.size != sum(wave.size for wave in waves):
                message = "speech text/sample conservation failed during assembly"
                raise AssertionError(message)
            peak = measure_signal(joined, 24000).peak
            gain = min(1.0, 0.99 / peak) if peak else 1.0
            save_waveform(output, np.asarray(joined * gain, dtype=np.float32))
            report.update(
                complete=True,
                active_candidate=None,
                samples=joined.size,
                sample_rate_hz=24000,
                duration_s=joined.size / 24000,
                peak_gain=gain,
                output_sha256=digest_file(output),
                elapsed_s=time.monotonic() - started,
                naturalness="predicted quality only; no human rating",
            )
            write_json(audit, report)
        return report
