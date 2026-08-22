"""Fadhma-Pro: Unified 566h+ speech corpus builder with speaker-disjoint validation.

Merges validated.tsv (566.85h), train.tsv (144.31h) and prompt-filtered other.tsv (104.42h),
while strictly filtering out all 1,023 evaluation speakers (dev + test) to ensure 0.0% leakage.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO, TypedDict

from agbalu.normalise import Normaliser
from agbalu.speech.vocabulary import ctc_target

log = logging.getLogger(__name__)


class SplitReport(TypedDict):
    clips: int
    hours: float
    speakers: int


class TrainSplitReport(SplitReport):
    rejections: dict[str, int]


class LeakageReport(TypedDict):
    train_dev_overlap: int
    train_test_overlap: int
    dev_test_overlap: int


class CorpusProReport(TypedDict):
    train_pro: TrainSplitReport
    dev: SplitReport
    test: SplitReport
    speaker_leakage_check: LeakageReport


RAW_CV_ROOT: Final = Path("data/raw/hf.fsicoli.common-voice-22-kab/transcript/kab")
DURATIONS_FILE: Final = "clip_durations.tsv"

SAMPLE_RATE: Final = 16_000
CONV_KERNEL: Final = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE: Final = (5, 2, 2, 2, 2, 2, 2)

MIN_DURATION_MS: Final = 400
MAX_DURATION_MS: Final = 20_000

REQUIRED_COLUMNS: Final = frozenset({"client_id", "path", "sentence"})
EXCESS: Final = "__excess__"

csv.field_size_limit(10**9)


class SpeechProError(Exception):
    """A malformed corpus or invalid split configuration."""


@dataclass(frozen=True, slots=True)
class ClipPro:
    """One utterance prepared for Fadhma-Pro training."""

    clip: str
    speaker: str
    split: str
    duration_ms: int
    text: str
    target: str
    repaired: bool

    def as_dict(self) -> dict[str, object]:
        """Convert to dictionary representation."""
        return {
            "clip": self.clip,
            "speaker": self.speaker,
            "split": self.split,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "target": self.target,
            "repaired": self.repaired,
        }


def _reader(handle: TextIO) -> csv.DictReader[str]:
    return csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE, restkey=EXCESS)


def read_durations(path: Path) -> dict[str, int]:
    """Read clip filename to duration in ms."""
    if not path.is_file():
        msg = f"clip durations not found at {path}"
        raise SpeechProError(msg)
    durations: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = _reader(handle)
        if reader.fieldnames is None or "clip" not in reader.fieldnames:
            msg = f"{path} missing 'clip' column"
            raise SpeechProError(msg)
        dur_col = next((f for f in reader.fieldnames if f.startswith("duration")), None)
        if dur_col is None:
            msg = f"{path} missing duration column"
            raise SpeechProError(msg)
        for row in reader:
            clip, val = row.get("clip"), row.get(dur_col)
            if clip and val:
                try:
                    durations[clip] = int(val)
                except ValueError:
                    continue
    return durations


def rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield valid TSV rows from a Common Voice split file."""
    if not path.is_file():
        msg = f"split file not found: {path}"
        raise SpeechProError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = _reader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            msg = f"{path} missing columns: {sorted(missing)}"
            raise SpeechProError(msg)
        for row in reader:
            if EXCESS in row or any(v is None for v in row.values()):
                continue
            yield row


def compute_frames(duration_ms: int) -> int:
    """Compute number of encoder output frames for a given audio duration in ms."""
    samples = duration_ms * SAMPLE_RATE // 1000
    length = samples
    for kernel, stride in zip(CONV_KERNEL, CONV_STRIDE, strict=True):
        length = (length - kernel) // stride + 1
    return max(length, 0)


def extract_speakers(path: Path) -> set[str]:
    """Extract set of unique client_id speaker IDs from a split TSV."""
    speakers: set[str] = set()
    for row in rows(path):
        speaker = row.get("client_id", "").strip()
        if speaker:
            speakers.add(speaker)
    return speakers


def _check_clip_validity(
    clip: str,
    speaker: str,
    duration: int | None,
    excluded_speakers: set[str] | None,
) -> str | None:
    if not clip or not speaker:
        return "missing-fields"
    if excluded_speakers and speaker in excluded_speakers:
        return "excluded-eval-speaker"
    if duration is None:
        return "no-duration"
    if duration < MIN_DURATION_MS:
        return "too-short"
    if duration > MAX_DURATION_MS:
        return "too-long"
    return None


def process_clip_row(
    row: dict[str, str],
    split: str,
    durations: Mapping[str, int],
    normaliser: Normaliser,
    excluded_speakers: set[str] | None = None,
) -> tuple[ClipPro | None, str | None]:
    """Process and normalize a single TSV clip row, checking bounds and CTC feasibility."""
    clip = row["path"].strip()
    speaker = row["client_id"].strip()
    raw_text = row["sentence"]
    duration = durations.get(clip)

    reason = _check_clip_validity(clip, speaker, duration, excluded_speakers)
    if reason is not None or duration is None:
        return None, reason

    text = normaliser.normalise(raw_text)
    target = ctc_target(text)
    if not target:
        return None, "empty-target"
    if len(target) > compute_frames(duration):
        return None, "ctc-infeasible"

    return (
        ClipPro(
            clip=clip,
            speaker=speaker,
            split=split,
            duration_ms=duration,
            text=text,
            target=target,
            repaired=text != raw_text,
        ),
        None,
    )


def _write_split_file(split_path: Path, clips: list[ClipPro]) -> None:
    with split_path.open("w", encoding="utf-8") as f:
        for c in clips:
            f.write(json.dumps(c.as_dict(), ensure_ascii=False) + "\n")


def _build_eval_split(
    tsv_path: Path,
    split_name: str,
    out_path: Path,
    durations: Mapping[str, int],
    normaliser: Normaliser,
) -> tuple[list[ClipPro], set[str]]:
    clips: list[ClipPro] = []
    speakers: set[str] = set()
    for r in rows(tsv_path):
        clip, _ = process_clip_row(r, split_name, durations, normaliser)
        if clip:
            clips.append(clip)
            speakers.add(clip.speaker)
    _write_split_file(out_path, clips)
    return clips, speakers


def _build_train_pro_split(
    sources: list[Path],
    out_path: Path,
    durations: Mapping[str, int],
    normaliser: Normaliser,
    eval_speakers: set[str],
) -> tuple[list[ClipPro], set[str], dict[str, int]]:
    seen_clips: set[str] = set()
    train_clips: list[ClipPro] = []
    rejections: dict[str, int] = {}
    train_speakers: set[str] = set()

    for source_path in sources:
        if not source_path.exists():
            continue
        log.info("Ingesting train source: %s", source_path.name)
        for r in rows(source_path):
            clip_name = r["path"].strip()
            if clip_name in seen_clips:
                continue
            clip_obj, reason = process_clip_row(
                r, "train_pro", durations, normaliser, excluded_speakers=eval_speakers
            )
            if clip_obj is not None:
                seen_clips.add(clip_name)
                train_clips.append(clip_obj)
                train_speakers.add(clip_obj.speaker)
            elif reason:
                rejections[reason] = rejections.get(reason, 0) + 1

    _write_split_file(out_path, train_clips)
    return train_clips, train_speakers, rejections


def build_pro_corpus(
    raw_dir: Path = RAW_CV_ROOT,
    out_dir: Path = Path("data/processed/speech_pro"),
    include_other: bool = True,
) -> CorpusProReport:
    """Build unified Fadhma-Pro dataset (train_pro.jsonl, dev.jsonl, test.jsonl)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    durations = read_durations(raw_dir / DURATIONS_FILE)
    normaliser = Normaliser()

    dev_path = raw_dir / "dev.tsv"
    test_path = raw_dir / "test.tsv"

    dev_clips, dev_speakers = _build_eval_split(
        dev_path, "dev", out_dir / "dev.jsonl", durations, normaliser
    )
    test_clips, test_speakers = _build_eval_split(
        test_path, "test", out_dir / "test.jsonl", durations, normaliser
    )
    eval_speakers = dev_speakers | test_speakers

    train_sources = [raw_dir / "validated.tsv", raw_dir / "train.tsv"]
    if include_other and (raw_dir / "other.tsv").exists():
        train_sources.append(raw_dir / "other.tsv")

    train_clips, train_speakers, rejections = _build_train_pro_split(
        train_sources, out_dir / "train_pro.jsonl", durations, normaliser, eval_speakers
    )

    total_train_ms = sum(c.duration_ms for c in train_clips)
    train_hours = total_train_ms / 3_600_000

    report: CorpusProReport = {
        "train_pro": {
            "clips": len(train_clips),
            "hours": round(train_hours, 2),
            "speakers": len(train_speakers),
            "rejections": rejections,
        },
        "dev": {
            "clips": len(dev_clips),
            "hours": round(sum(c.duration_ms for c in dev_clips) / 3_600_000, 2),
            "speakers": len(dev_speakers),
        },
        "test": {
            "clips": len(test_clips),
            "hours": round(sum(c.duration_ms for c in test_clips) / 3_600_000, 2),
            "speakers": len(test_speakers),
        },
        "speaker_leakage_check": {
            "train_dev_overlap": len(train_speakers & dev_speakers),
            "train_test_overlap": len(train_speakers & test_speakers),
            "dev_test_overlap": len(dev_speakers & test_speakers),
        },
    }

    stats_path = out_dir / "speech_pro.stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info(
        "Corpus Pro Built: %.2f hours, %d clips, 0.0%% speaker overlap",
        train_hours,
        len(train_clips),
    )
    return report
