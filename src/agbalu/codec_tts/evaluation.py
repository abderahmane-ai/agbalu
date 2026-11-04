"""Measure codec likelihood and speech quality on fixed reference panels."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydantic import Field
from torch import Tensor, nn
from torch.nn import functional
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    WavLMForXVector,
)

from agbalu.codec_tts.artifacts import (
    atomic_binary,
    digest_file,
    provenance,
    read_recordings,
    write_json,
)
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.codec import Codec
from agbalu.codec_tts.configuration import Specification
from agbalu.codec_tts.dataset import SpeechDataset, collate_layouts
from agbalu.codec_tts.inference import Sampling, generate_codes
from agbalu.codec_tts.layout import prompt_layout
from agbalu.codec_tts.objective import codec_loss
from agbalu.codec_tts.partition import priority
from agbalu.codec_tts.waveforms import Waveform, resample_waveform, text_key

ASR_ID = "agbalu/Fadhma-300M"
ASR_REVISION = "d7f5bae148cde28d70d1bcf6df39931c24144583"
SPEAKER_ID = "microsoft/wavlm-base-plus-sv"
SPEAKER_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
MAX_CER_EXCESS = 0.01
MIN_NATURALNESS = 4.3
MIN_PRONUNCIATION = 4.5
MIN_VOICE_RETENTION = 0.95


class SpeechObservation(Specification):
    record_id: str
    speaker: str
    reference_characters: int = Field(ge=1)
    character_errors: int | None = Field(default=None, ge=0)
    real_character_errors: int | None = Field(default=None, ge=0)
    voice_cosine: float | None = None
    real_voice_cosine: float | None = None
    terminated: bool
    error: str | None = None
    waveform_sha256: str | None = None


class ListenerReview(Specification):
    evaluation_sha256: str
    naturalness_mean: float = Field(ge=1, le=5)
    pronunciation_mean: float = Field(ge=1, le=5)
    voice_retention_fraction: float = Field(ge=0, le=1)
    expression_retained: bool
    language_retained: bool
    ratings: int = Field(ge=2)
    listeners: int = Field(ge=2)


def edit_count(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, left in enumerate(reference, start=1):
        current = [row]
        for column, right in enumerate(hypothesis, start=1):
            current.append(
                min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def selection_report(
    observations: Sequence[SpeechObservation], review: ListenerReview | None = None
) -> dict[str, object]:
    support = sum(item.reference_characters for item in observations)
    measured = [
        item
        for item in observations
        if item.character_errors is not None and item.real_character_errors is not None
    ]
    measured_support = sum(item.reference_characters for item in measured)
    excess = (
        (
            sum(
                (item.character_errors or 0) - (item.real_character_errors or 0)
                for item in measured
            )
            / measured_support
        )
        if measured_support
        else None
    )
    failures = sum(not item.terminated for item in observations)
    reasons: list[str] = []
    if not observations or failures:
        reasons.append("empty panel or failed native termination")
    if len(measured) != len(observations) or excess is None or excess > MAX_CER_EXCESS:
        reasons.append("CER above matched real-audio floor is missing or exceeds 0.01")
    voices = [item.voice_cosine for item in observations if item.voice_cosine is not None]
    if len(voices) != len(observations):
        reasons.append("speaker verification is incomplete")
    if review is None:
        reasons.append("blinded voice/expression/language review is pending")
    elif (
        review.naturalness_mean < MIN_NATURALNESS
        or review.pronunciation_mean < MIN_PRONUNCIATION
        or review.voice_retention_fraction < MIN_VOICE_RETENTION
        or not review.expression_retained
        or not review.language_retained
    ):
        reasons.append("listener quality or retained-capability gate failed")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "utterances": len(observations),
        "reference_characters": support,
        "measured_characters": measured_support,
        "cer_excess": excess,
        "termination_failures": failures,
        "mean_voice_cosine": float(np.mean(voices)) if voices else None,
    }


@torch.inference_mode()
def heldout_loss(model: Generator, dataset: SpeechDataset, count: int) -> dict[str, object]:
    device = model.audio_embedding.weight.device
    was_training = model.training
    model.train(False)
    nats = torch.zeros(8, dtype=torch.float64, device=device)
    counts = torch.zeros(8, dtype=torch.long, device=device)
    order = sorted(
        range(len(dataset)),
        key=lambda index: priority(dataset.recipe.seed, dataset.pairs[index].target),
    )[:count]
    try:
        for index in order:
            text, codes, audio, valid, labels = collate_layouts([dataset.layout(index)], device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model.hidden(text, codes, audio, valid)
                loss = codec_loss(model.classify(hidden), labels)
            nats += loss.stream_nats.double()
            counts += loss.stream_counts
    finally:
        model.train(was_training)
    return {
        "audio_ce_nats": float((nats.sum() / counts.sum()).item()),
        "stream_nats": nats.cpu().tolist(),
        "stream_counts": counts.cpu().tolist(),
        "examples": len(order),
        "teacher_forced": True,
    }


def load_control_audio(dataset: SpeechDataset, record_id: str) -> Waveform:
    entry = dataset.entries[record_id]
    path = dataset.recipe.output / "evaluation" / f"{record_id}.wav"
    if (
        not entry.evaluation_waveform_sha256
        or digest_file(path) != entry.evaluation_waveform_sha256
    ):
        message = f"missing/mismatched real evaluation waveform {path}"
        raise ValueError(message)
    values, rate_hz = sf.read(path, dtype="float32")
    waveform = np.asarray(values, dtype=np.float32)
    return resample_waveform(waveform, int(rate_hz), 16000)


def extract_input(
    extractor: Wav2Vec2FeatureExtractor, audio: Waveform, device: torch.device
) -> Tensor:
    features = extractor(audio, sampling_rate=16000, return_tensors="pt")
    value = features.get("input_values")
    if not isinstance(value, Tensor):
        message = "audio feature extractor returned no input_values tensor"
        raise TypeError(message)
    return value.to(device)


@torch.inference_mode()
def measure_speech(
    model: Generator, dataset: SpeechDataset, codec: Codec, output: Path
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records = {
        item.record_id: item for item in read_recordings(dataset.recipe.output / "recordings.jsonl")
    }
    pairs = sorted(dataset.pairs, key=lambda pair: priority(dataset.recipe.seed, pair.target))[
        : dataset.recipe.evaluation_examples
    ]
    waves: dict[str, Waveform] = {}
    failures: dict[str, str] = {}
    for pair in pairs:
        target, reference = dataset.entries[pair.target], dataset.entries[pair.reference]
        prompt = prompt_layout(
            dataset.specials, target.text_ids, dataset.codes(reference), reference.text_ids
        )
        try:
            codes, trace = generate_codes(
                model, prompt, Sampling(max_new_positions=min(1500, 8192 - prompt.positions))
            )
            waveform = codec.decode(codes).numpy()
        except (ValueError, RuntimeError) as error:
            failures[pair.target] = str(error)
            continue
        path = output / f"{pair.target}.wav"
        with atomic_binary(path) as stream:
            sf.write(stream, waveform, 24000, format="WAV", subtype="FLOAT")
        write_json(
            path.with_suffix(".stats.json"),
            {
                **provenance(dataset.recipe.digest),
                **trace,
                "target": pair.target,
                "reference": pair.reference,
                "waveform_sha256": digest_file(path),
            },
        )
        waves[pair.target] = resample_waveform(np.asarray(waveform, dtype=np.float32), 24000, 16000)
    device = model.audio_embedding.weight.device
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(ASR_ID, revision=ASR_REVISION)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(ASR_ID, revision=ASR_REVISION)
    recognizer: nn.Module = Wav2Vec2ForCTC.from_pretrained(ASR_ID, revision=ASR_REVISION)
    recognizer.to(device).train(False)
    errors: dict[str, tuple[int, int]] = {}
    for pair in pairs:
        if pair.target not in waves:
            continue
        hypotheses: list[str] = []
        for audio in (waves[pair.target], load_control_audio(dataset, pair.target)):
            result = recognizer(input_values=extract_input(extractor, audio, device))
            logits = getattr(result, "logits", None)
            if not isinstance(logits, Tensor):
                message = "ASR did not return logits"
                raise TypeError(message)
            hypotheses.append(str(tokenizer.decode(logits[0].argmax(-1).cpu().tolist())))
        text = text_key(records[pair.target].text).lower()
        errors[pair.target] = (
            edit_count(text, text_key(hypotheses[0]).lower()),
            edit_count(text, text_key(hypotheses[1]).lower()),
        )
    del recognizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    verifier: nn.Module = WavLMForXVector.from_pretrained(SPEAKER_ID, revision=SPEAKER_REVISION)
    verifier.to(device).train(False)
    voice_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        SPEAKER_ID, revision=SPEAKER_REVISION
    )
    observations: list[SpeechObservation] = []
    for pair in pairs:
        text = text_key(records[pair.target].text).lower()
        if pair.target in failures:
            observations.append(
                SpeechObservation(
                    record_id=pair.target,
                    speaker=pair.speaker,
                    reference_characters=len(text),
                    terminated=False,
                    error=failures[pair.target],
                )
            )
            continue
        embeddings: list[Tensor] = []
        for audio in (
            waves[pair.target],
            load_control_audio(dataset, pair.target),
            load_control_audio(dataset, pair.reference),
        ):
            result = verifier(input_values=extract_input(voice_extractor, audio, device))
            embedding = getattr(result, "embeddings", None)
            if not isinstance(embedding, Tensor):
                message = "speaker verifier did not return embeddings"
                raise TypeError(message)
            embeddings.append(functional.normalize(embedding.float(), dim=-1))
        generated, real, voice_reference = embeddings
        observations.append(
            SpeechObservation(
                record_id=pair.target,
                speaker=pair.speaker,
                reference_characters=len(text),
                character_errors=errors[pair.target][0],
                real_character_errors=errors[pair.target][1],
                terminated=True,
                voice_cosine=float((generated * voice_reference).sum().item()),
                real_voice_cosine=float((real * voice_reference).sum().item()),
                waveform_sha256=digest_file(output / f"{pair.target}.wav"),
            )
        )
    del verifier
    report = {
        **provenance(dataset.recipe.digest),
        "observations": [item.model_dump() for item in observations],
        "selection": selection_report(observations),
        "asr_decoder": "greedy",
        "asr_revision": ASR_REVISION,
        "speaker_revision": SPEAKER_REVISION,
    }
    write_json(output / "evaluation.stats.json", report)
    return report
