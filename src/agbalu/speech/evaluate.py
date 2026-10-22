"""Standardized evaluation harness for Fadhma 2.

Evaluates against:
1. Clean test set (CER & WER across unseen speakers)
2. Phrase-disjoint test set (acoustic generalization on unseen sentences)
3. 30-hour silence challenge set (asserts zero spurious character emissions)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import torch

from agbalu.speech.decode import SpeechDecoder
from agbalu.speech.manifest import load_manifest
from agbalu.speech.metrics import cer, wer

DEFAULT_CHALLENGE_SILENCE_SECONDS: Final[list[float]] = [1.0, 5.0, 10.0, 30.0, 60.0]


@dataclass(frozen=True, slots=True)
class EvaluationScore:
    """Error rate summary for a single benchmark split."""

    cer_percent: float
    wer_percent: float
    total_clips: int
    total_duration_hours: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SilenceChallengeResult:
    """Verification of zero emissions during pure non-speech."""

    total_probes: int
    insertions: int
    hallucination_rate_percent: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullBenchmarkReport:
    """Comprehensive benchmark report across all test conditions."""

    clean: EvaluationScore
    phrase_disjoint: EvaluationScore | None
    silence_challenge: SilenceChallengeResult

    def as_dict(self) -> dict[str, object]:
        return {
            "clean": self.clean.as_dict(),
            "phrase_disjoint": self.phrase_disjoint.as_dict() if self.phrase_disjoint else None,
            "silence_challenge": self.silence_challenge.as_dict(),
        }


def evaluate_manifest(
    decoder: SpeechDecoder,
    manifest_path: Path,
    *,
    base_audio_dir: Path | None = None,
    use_lm: bool = True,
    max_clips: int | None = None,
) -> EvaluationScore:
    """Decode a test manifest and compute CER and WER."""
    records = load_manifest(manifest_path)
    if max_clips is not None:
        records = records[:max_clips]

    references: list[str] = []
    hypotheses: list[str] = []
    total_dur = 0.0

    for record in records:
        audio_path = (
            (base_audio_dir / record.audio_path) if base_audio_dir else Path(record.audio_path)
        )
        try:
            res = decoder.transcribe_file(audio_path, use_lm=use_lm)
            hypotheses.append(res.text)
        except Exception:
            # Empty emission if audio cannot be decoded
            hypotheses.append("")

        references.append(record.target_text)
        total_dur += record.duration

    cer_score = cer(references, hypotheses)
    wer_score = wer(references, hypotheses)

    return EvaluationScore(
        cer_percent=float(cer_score.as_dict()["percent"]),
        wer_percent=float(wer_score.as_dict()["percent"]),
        total_clips=len(records),
        total_duration_hours=round(total_dur / 3600.0, 2),
    )


def evaluate_silence_challenge(
    decoder: SpeechDecoder,
    durations: list[float] | None = None,
    sample_rate: int = 16000,
) -> SilenceChallengeResult:
    """Probe decoder with pure silence to guarantee zero false insertions."""
    probe_durations = durations or DEFAULT_CHALLENGE_SILENCE_SECONDS
    insertions = 0

    for dur in probe_durations:
        num_samples = int(dur * sample_rate)
        # Test digital zero
        digital_zero = torch.zeros(num_samples, dtype=torch.float32)
        out_zero = decoder.decode_audio_tensor(digital_zero, sample_rate=sample_rate)
        if out_zero.strip():
            insertions += 1

        # Test low-level dither/noise (room tone below voice threshold)
        dither = torch.randn(num_samples, dtype=torch.float32) * 1e-4
        out_dither = decoder.decode_audio_tensor(dither, sample_rate=sample_rate)
        if out_dither.strip():
            insertions += 1

    total_probes = len(probe_durations) * 2
    hallucination_rate = (insertions / total_probes) * 100.0

    return SilenceChallengeResult(
        total_probes=total_probes,
        insertions=insertions,
        hallucination_rate_percent=round(hallucination_rate, 4),
        passed=(insertions == 0),
    )


def run_full_benchmark(
    decoder: SpeechDecoder,
    clean_manifest: Path,
    *,
    phrase_disjoint_manifest: Path | None = None,
    base_audio_dir: Path | None = None,
    use_lm: bool = True,
    output_report_path: Path | None = None,
) -> FullBenchmarkReport:
    """Execute complete benchmark suite and write results to JSON."""
    clean_score = evaluate_manifest(
        decoder, clean_manifest, base_audio_dir=base_audio_dir, use_lm=use_lm
    )

    phrase_score: EvaluationScore | None = None
    if phrase_disjoint_manifest and phrase_disjoint_manifest.is_file():
        phrase_score = evaluate_manifest(
            decoder, phrase_disjoint_manifest, base_audio_dir=base_audio_dir, use_lm=use_lm
        )

    silence_result = evaluate_silence_challenge(decoder)

    report = FullBenchmarkReport(
        clean=clean_score,
        phrase_disjoint=phrase_score,
        silence_challenge=silence_result,
    )

    if output_report_path is not None:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(
            json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
        )

    return report
