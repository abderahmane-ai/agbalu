#!/usr/bin/env python3
"""Rapidly re-partition existing speech records without re-downloading or re-filtering.

Loads existing manifests from data/processed/speech/, re-partitions with speaker
duration caps to ensure diverse multi-speaker dev/test benchmarks, updates vocab,
and writes updated splits and statistics in seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agbalu.speech.manifest import AudioRecord, save_manifest
from agbalu.speech.partition import DEFAULT_MAX_EVAL_SPEAKER_HOURS, DEFAULT_SEED, partition_records
from agbalu.speech.vocabulary import build as build_vocabulary

DEFAULT_SPEECH_DIR = Path("data/processed/speech")
DEFAULT_VOCAB_PATH = Path("artifacts/asr/vocab.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speech-dir",
        type=Path,
        default=DEFAULT_SPEECH_DIR,
        help="Directory containing train.jsonl, dev.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Path to output CTC vocab.json",
    )
    parser.add_argument(
        "--max-eval-speaker-hours",
        type=float,
        default=DEFAULT_MAX_EVAL_SPEAKER_HOURS,
        help="Max hours per speaker in eval splits (speakers with more are locked to train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for partitioning",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[AudioRecord]:
    if not path.is_file():
        return []
    records: list[AudioRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            records.append(AudioRecord.from_dict(data))
    return records


def main() -> int:
    args = parse_args()
    speech_dir = args.speech_dir
    stats_file = speech_dir / "speech_data.stats.json"

    # Load existing rejections if available
    rejections: dict[str, int] = {}
    if stats_file.is_file():
        try:
            prev_stats = json.loads(stats_file.read_text(encoding="utf-8"))
            rejections = prev_stats.get("rejections", {})
        except Exception as exc:
            sys.stderr.write(f"Warning: could not read previous stats: {exc}\n")

    sys.stdout.write("--> Loading existing records from processed speech manifests...\n")
    all_records: list[AudioRecord] = []
    seen_ids: set[str] = set()

    for split_name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        split_path = speech_dir / split_name
        loaded = load_manifest(split_path)
        sys.stdout.write(f"    Loaded {len(loaded):,} clips from {split_name}\n")
        for rec in loaded:
            if rec.audio_id not in seen_ids:
                seen_ids.add(rec.audio_id)
                all_records.append(rec)

    if not all_records:
        sys.stderr.write(f"error: no records found in {speech_dir}\n")
        return 1

    sys.stdout.write(f"--> Total unique records to partition: {len(all_records):,}\n")
    sys.stdout.write(
        f"--> Partitioning (seed={args.seed}, "
        f"max_eval_speaker_hours={args.max_eval_speaker_hours})...\n"
    )

    dataset = partition_records(
        all_records,
        seed=args.seed,
        max_eval_speaker_hours=args.max_eval_speaker_hours,
    )
    report = dataset.report()

    # Save updated manifests
    sys.stdout.write("--> Writing updated split manifests...\n")
    save_manifest(dataset.train, speech_dir / "train.jsonl")
    save_manifest(dataset.dev, speech_dir / "dev.jsonl")
    save_manifest(dataset.test, speech_dir / "test.jsonl")
    save_manifest(dataset.test_phrase_disjoint, speech_dir / "test_phrase_disjoint.jsonl")

    # Update vocabulary
    sys.stdout.write(f"--> Generating CTC vocabulary -> {args.vocab_path}...\n")
    args.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = build_vocabulary(
        (r.target_text for r in dataset.train),
        sources=(r.raw_text for r in dataset.train),
    )
    vocab.write(args.vocab_path)

    # Save updated stats
    report_data = report.as_dict()
    report_data["rejections"] = rejections
    stats_file.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("  Speech Re-partitioning Completed Successfully\n")
    sys.stdout.write("=" * 70 + "\n")
    for split in report.splits:
        sys.stdout.write(
            f"  {split.name:22s} {split.clips:7,d} clips  {split.hours:8.2f} h  "
            f"{split.speakers:5,d} speakers\n"
        )
    sys.stdout.write(f"  Speaker isolation check: {report.speaker_overlaps}\n")
    sys.stdout.write(f"  Vocabulary classes: {len(vocab.as_mapping())}\n")
    sys.stdout.write(f"  Stats file: {stats_file}\n")
    sys.stdout.write("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
