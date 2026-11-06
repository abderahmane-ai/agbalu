"""Produce verified speech demos with aligned subtitle words."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator
from torch import Tensor, nn
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from agbalu.codec_tts.artifacts import digest_file, write_json
from agbalu.codec_tts.configuration import Specification
from agbalu.codec_tts.evaluation import ASR_ID, ASR_REVISION, extract_input
from agbalu.codec_tts.synthesis import (
    Candidate,
    InferenceSettings,
    choose_candidate,
    read_waveform,
    save_waveform,
)
from agbalu.codec_tts.transformers_model import SpeechConfig, SpeechModel
from agbalu.codec_tts.verification import spoken_text
from agbalu.codec_tts.waveforms import resample_waveform

VOICE_COUNT = 4
EMISSION_AXES = 2


class Voice(Specification):
    key: str = Field(pattern=r"^[a-z][a-z-]+$")
    name: str
    route: str
    reference: Path
    reference_text: str
    text: str
    identity: Literal["celebrity", "pseudonym"]
    source: str


class Showcase(Specification):
    model: Path = Path("artifacts/release/Matoub-4B")
    output: Path = Path("artifacts/runs/matoub-linkedin")
    font: Path = Path("artifacts/runs/matoub-linkedin/NotoSans.ttf")
    voices: tuple[Voice, ...]

    @model_validator(mode="after")
    def validate_voices(self) -> Showcase:
        if len(self.voices) != VOICE_COUNT or len({v.key for v in self.voices}) != VOICE_COUNT:
            message = "showcase requires four distinct voices"
            raise ValueError(message)
        return self


class Word(Specification):
    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> Word:
        if self.end_s <= self.start_s:
            message = f"invalid word interval {self.text!r}: {self.start_s}..{self.end_s}"
            raise ValueError(message)
        return self


def align_characters(
    emissions: NDArray[np.float32], tokens: list[int], blank: int
) -> list[tuple[int, int]]:
    if emissions.ndim != EMISSION_AXES or not tokens or not np.isfinite(emissions).all():
        message = f"invalid CTC input: shape={emissions.shape}, tokens={tokens}"
        raise ValueError(message)
    frames, vocabulary = emissions.shape
    if any(t < 0 or t >= vocabulary or t == blank for t in tokens):
        message = f"invalid CTC labels={tokens}, blank={blank}, vocabulary={vocabulary}"
        raise ValueError(message)
    required = len(tokens) + sum(a == b for a, b in pairwise(tokens))
    if frames < required or not 0 <= blank < vocabulary:
        message = f"CTC alignment needs {required} frames, got {frames}; blank={blank}"
        raise ValueError(message)
    labels = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    labels[1::2] = tokens
    score = np.full(labels.size, -np.inf, dtype=np.float64)
    score[:2] = emissions[0, labels[:2]]
    back = np.zeros((frames, labels.size), dtype=np.uint8)
    can_skip = (labels != blank) & (labels != np.roll(labels, 2))
    can_skip[:2] = False
    for frame in range(1, frames):
        previous = np.full_like(score, -np.inf)
        previous[1:] = score[:-1]
        skip = np.full_like(score, -np.inf)
        skip[2:] = score[:-2]
        skip[~can_skip] = -np.inf
        options = np.stack((score, previous, skip))
        choice = options.argmax(axis=0)
        back[frame] = choice
        score = options[choice, np.arange(labels.size)] + emissions[frame, labels]
    state = int(labels.size - 2 + score[-2:].argmax())
    if not np.isfinite(score[state]):
        message = "no complete CTC alignment path"
        raise ValueError(message)
    path = np.zeros(frames, dtype=np.int64)
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    result: list[tuple[int, int]] = []
    for index in range(len(tokens)):
        locations = np.flatnonzero(path == 2 * index + 1)
        if not locations.size:
            message = f"CTC omitted token index={index}, value={tokens[index]}"
            raise ValueError(message)
        result.append((int(locations[0]), int(locations[-1]) + 1))
    return result


def align_words(voice: Voice, waveform_path: Path, output: Path) -> None:
    audio = resample_waveform(read_waveform(waveform_path), 24000, 16000)
    recognizer: nn.Module = Wav2Vec2ForCTC.from_pretrained(ASR_ID, revision=ASR_REVISION)
    nn.Module.train(recognizer, False)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(ASR_ID, revision=ASR_REVISION)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(ASR_ID, revision=ASR_REVISION)
    expected = spoken_text(voice.text)
    tokens = tokenizer.encode(expected, add_special_tokens=False)
    if tokenizer.unk_token_id in tokens:
        message = f"unsupported ASR character in {voice.key}: {expected!r}"
        raise ValueError(message)
    with torch.inference_mode():
        logits = getattr(
            recognizer(input_values=extract_input(extractor, audio, torch.device("cpu"))),
            "logits",
            None,
        )
        if not isinstance(logits, Tensor):
            message = "subtitle recognizer returned no logits"
            raise TypeError(message)
        emissions = np.asarray(logits[0].log_softmax(-1).numpy(), dtype=np.float32)
    spans = align_characters(emissions, tokens, int(tokenizer.pad_token_id))
    frame_s = audio.size / 16000 / emissions.shape[0]
    words: list[Word] = []
    offset = 0
    for raw in voice.text.split():
        normalized = spoken_text(raw)
        length = len(tokenizer.encode(normalized, add_special_tokens=False))
        words.append(
            Word(
                text=raw,
                start_s=spans[offset][0] * frame_s,
                end_s=spans[offset + length - 1][1] * frame_s,
            )
        )
        offset += length + 1
    if offset - 1 != len(tokens):
        message = f"subtitle conservation failed at {voice.key}: {offset - 1}/{len(tokens)}"
        raise ValueError(message)
    write_json(
        output,
        {
            "audio_sha256": digest_file(waveform_path),
            "asr_revision": ASR_REVISION,
            "words": [w.model_dump() for w in words],
        },
    )


def generate(spec: Showcase, voice: Voice, settings: InferenceSettings | None = None) -> None:
    config = SpeechConfig.from_pretrained(spec.model)
    model = SpeechModel.from_pretrained(spec.model, config=config)
    nn.Module.to(model, device=torch.device("cuda"))
    nn.Module.train(model, False)
    if settings is not None:
        model.config.inference = settings.model_dump()
    output = spec.output / "selected" / f"{voice.key}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = model.generate(
        text=voice.text,
        reference_audio=voice.reference,
        reference_text=voice.reference_text,
        output_path=output,
    )
    if result.report is None or not result.report["complete"]:
        message = f"incomplete showcase generation for {voice.key}"
        raise ValueError(message)
    print(json.dumps({"voice": voice.key, "complete": True, "audio": str(output)}), flush=True)


def curate(spec: Showcase, voice: Voice) -> Path:
    audit = spec.output / f"{voice.key}.stats.json"
    report = json.loads(audit.read_bytes())
    if not report["complete"] or len(report["segments"]) != 1:
        message = f"curation requires one complete short utterance: {audit}"
        raise ValueError(message)
    settings = InferenceSettings.model_validate(report["settings"])
    candidates = [
        Candidate.model_validate({k: v for k, v in row.items() if k != "rejected"})
        for row in report["segments"][0]["candidates"]
    ]
    selected = choose_candidate(candidates, settings)
    if selected is None:
        message = f"no showcase candidate passes quality gates: {voice.key}"
        raise ValueError(message)
    candidate = candidates[selected]
    source = spec.output / f"{voice.key}.candidates" / f"segment-0000-candidate-{selected:02d}.wav"
    if digest_file(source) != candidate.waveform_sha256:
        message = f"candidate waveform changed: {source}"
        raise ValueError(message)
    wave = read_waveform(source)
    peak = float(np.abs(wave).max())
    gain = min(1.0, 0.99 / peak)
    destination = spec.output / "selected" / f"{voice.key}.wav"
    save_waveform(destination, np.asarray(wave * gain, dtype=np.float32))
    write_json(
        destination.with_suffix(".stats.json"),
        {
            "source_audit_sha256": digest_file(audit),
            "sha256": digest_file(destination),
            "selected": selected,
            "quality": candidate.quality.model_dump() if candidate.quality else None,
            "duration_s": wave.size / 24000,
            "peak_gain": gain,
            "quality_tolerance": settings.quality_tolerance,
        },
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["generate", "curate", "align"], default="generate")
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/runs/matoub-linkedin/storyboard.json")
    )
    parser.add_argument("--voice", required=True)
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    spec = Showcase.model_validate_json(args.manifest.read_bytes())
    voice = next((v for v in spec.voices if v.key == args.voice), None)
    if voice is None:
        message = f"unknown showcase voice={args.voice!r}"
        raise ValueError(message)
    spec.output.mkdir(parents=True, exist_ok=True)
    if args.task == "generate":
        settings = (
            InferenceSettings.model_validate_json(args.settings.read_bytes())
            if args.settings is not None
            else None
        )
        generate(spec, voice, settings)
    elif args.task == "curate":
        print(curate(spec, voice), flush=True)
    else:
        align_words(
            voice,
            spec.output / "selected" / f"{voice.key}.wav",
            spec.output / f"{voice.key}.words.json",
        )


if __name__ == "__main__":
    main()
