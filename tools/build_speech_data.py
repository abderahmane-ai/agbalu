#!/usr/bin/env python3
"""AƔBALU: Multi-Source Speech Pipeline Driver for Fadhma 2.

Runs directly on the remote compute environment (e.g. HPC pod):
1. Ingests Common Voice, Tatoeba, and TutlaytAI collections from Hugging Face or local dumps.
2. Applies Dihya-5M language identification filter to eliminate sibling varieties.
3. Builds 90/5/5 speaker-disjoint splits (`train`, `dev`, `test`, `test_phrase_disjoint`).
4. Generates the canonical CTC vocabulary.
5. Saves full provenance reports to `data/processed/speech/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from agbalu.speech.manifest import (
    AudioRecord,
    filter_with_dihya,
    ingest_common_voice_tsv,
    ingest_hf_parquet_dataset,
    ingest_tatoeba_tsv,
    ingest_tutlayt_jsonl,
    read_durations_tsv,
    save_manifest,
)
from agbalu.speech.partition import partition_records
from agbalu.speech.vocabulary import build as build_vocabulary

DEFAULT_OUT_DIR = Path("data/processed/speech")
DEFAULT_AUDIO_DIR = Path("data/raw/speech/audio")
DEFAULT_VOCAB_PATH = Path("artifacts/asr/vocab.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv-repo",
        type=str,
        default="taqbaylit/common-voice-scripted-speech-kab-26-huge",
        help="Hugging Face repo for Common Voice Kabyle scripted speech",
    )
    parser.add_argument(
        "--tutlayt-repo",
        type=str,
        default="TutlaytAI/kabyle_asr",
        help="Hugging Face repo for TutlaytAI Kabyle ASR",
    )
    parser.add_argument(
        "--tutlayt-tamazight-repo",
        type=str,
        default="TutlaytAI/tamazight_asr",
        help="Hugging Face repo for TutlaytAI Tamazight ASR (filtered via Dihya)",
    )
    parser.add_argument(
        "--tatoeba-repo",
        type=str,
        default="boffire/tatoeba-kabyle-audio",
        help="Hugging Face repo for Tatoeba Kabyle audio",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory to save extracted audio clips",
    )
    parser.add_argument(
        "--max-clips-per-source",
        type=int,
        default=None,
        help="Optional limit for rapid testing / profiling",
    )
    parser.add_argument(
        "--cv-tsv",
        type=Path,
        default=None,
        help="Optional local path to Common Voice validated.tsv",
    )
    parser.add_argument(
        "--cv-durations",
        type=Path,
        default=None,
        help="Optional local path to Common Voice clip_durations.tsv",
    )
    parser.add_argument(
        "--tatoeba-tsv",
        type=Path,
        default=None,
        help="Optional local path to Tatoeba audio metadata TSV",
    )
    parser.add_argument(
        "--tutlayt-jsonl",
        type=Path,
        default=None,
        help="Optional local path to TutlaytAI JSONL manifest",
    )
    parser.add_argument(
        "--dihya",
        action="store_true",
        default=True,
        help="Enable Dihya-5M language identification filtering (default: True)",
    )
    parser.add_argument(
        "--no-dihya",
        action="store_false",
        dest="dihya",
        help="Disable Dihya-5M filtering",
    )
    parser.add_argument(
        "--dihya-dir",
        type=Path,
        default=Path("artifacts/runs/dihya"),
        help="Directory holding Dihya-5M checkpoint and calibration",
    )
    parser.add_argument(
        "--dihya-confidence",
        type=float,
        default=0.70,
        help="Minimum Dihya confidence for Kabyle classification",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for processed splits",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="Output path for CTC vocab.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic partitioning",
    )
    parser.add_argument(
        "--max-eval-speaker-hours",
        type=float,
        default=3.0,
        help="Maximum cumulative hours per speaker in dev/test splits",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.vocab_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write("  AƔBALU: Fadhma 2 Sovereign Speech Data Build Pipeline\n")
    sys.stdout.write("=" * 70 + "\n")

    records: list[AudioRecord] = []
    rejections: Counter[str] = Counter()

    # 1. Hugging Face Parquet sources
    if args.cv_repo:
        sys.stdout.write(f"--> Ingesting Common Voice from HF: {args.cv_repo}...\n")
        records.extend(
            ingest_hf_parquet_dataset(
                repo_id=args.cv_repo,
                output_audio_dir=args.audio_dir,
                source_name="common_voice_26",
                text_column="transcription",
                speaker_column="client_id",
                license_str="CC0-1.0",
                max_clips=args.max_clips_per_source,
                rejections=rejections,
            )
        )
        sys.stdout.write(f"    Total records so far: {len(records):,}\n")

    if args.tutlayt_repo:
        sys.stdout.write(f"--> Ingesting TutlaytAI Kabyle from HF: {args.tutlayt_repo}...\n")
        records.extend(
            ingest_hf_parquet_dataset(
                repo_id=args.tutlayt_repo,
                output_audio_dir=args.audio_dir,
                source_name="tutlayt_kabyle",
                text_column="Text",
                speaker_column="client_id",
                license_str="CC-BY-4.0",
                max_clips=args.max_clips_per_source,
                rejections=rejections,
            )
        )
        sys.stdout.write(f"    Total records so far: {len(records):,}\n")

    if args.tutlayt_tamazight_repo:
        sys.stdout.write(
            f"--> Ingesting TutlaytAI Tamazight from HF: {args.tutlayt_tamazight_repo}...\n"
        )
        records.extend(
            ingest_hf_parquet_dataset(
                repo_id=args.tutlayt_tamazight_repo,
                output_audio_dir=args.audio_dir,
                source_name="tutlayt_tamazight",
                text_column="Text",
                speaker_column="client_id",
                license_str="CC-BY-4.0",
                max_clips=args.max_clips_per_source,
                rejections=rejections,
            )
        )
        sys.stdout.write(f"    Total records so far: {len(records):,}\n")

    if args.tatoeba_repo:
        sys.stdout.write(f"--> Ingesting Tatoeba Kabyle from HF: {args.tatoeba_repo}...\n")
        records.extend(
            ingest_hf_parquet_dataset(
                repo_id=args.tatoeba_repo,
                output_audio_dir=args.audio_dir,
                source_name="tatoeba",
                text_column="text",
                speaker_column="sentence_id",
                license_str="CC-BY-2.0-FR",
                max_clips=args.max_clips_per_source,
                rejections=rejections,
            )
        )
        sys.stdout.write(f"    Total records so far: {len(records):,}\n")

    # 2. Local fallback files if provided
    if args.cv_tsv and args.cv_tsv.is_file():
        durations = (
            read_durations_tsv(args.cv_durations)
            if args.cv_durations and args.cv_durations.is_file()
            else {}
        )
        sys.stdout.write(f"--> Ingesting local Common Voice TSV: {args.cv_tsv}...\n")
        records.extend(
            ingest_common_voice_tsv(
                args.cv_tsv,
                durations_map=durations,
                rejections=rejections,
            )
        )

    if args.tatoeba_tsv and args.tatoeba_tsv.is_file():
        sys.stdout.write(f"--> Ingesting local Tatoeba TSV: {args.tatoeba_tsv}...\n")
        records.extend(ingest_tatoeba_tsv(args.tatoeba_tsv, rejections=rejections))

    if args.tutlayt_jsonl and args.tutlayt_jsonl.is_file():
        sys.stdout.write(f"--> Ingesting local TutlaytAI JSONL: {args.tutlayt_jsonl}...\n")
        records.extend(
            ingest_tutlayt_jsonl(args.tutlayt_jsonl, filter_kabyle_only=True, rejections=rejections)
        )

    sys.stdout.write(f"--> Ingested {len(records):,} total raw candidate clips.\n")

    # 3. Dihya language filtering (guarantee 100% clean Kabyle)
    if args.dihya:
        sys.stdout.write(
            f"--> Applying Dihya-5M language filter (threshold={args.dihya_confidence})...\n"
        )
        filtered = list(
            filter_with_dihya(
                records,
                dihya_dir=args.dihya_dir,
                min_kabyle_confidence=args.dihya_confidence,
                rejections=rejections,
            )
        )
        sys.stdout.write(f"--> Retained {len(filtered):,} clips after Dihya verification.\n")
        records = filtered

    if not records:
        sys.stderr.write("error: no speech records ingested or retained.\n")
        return 1

    # 4. Partitioning: 90% train, 5% dev, 5% test
    sys.stdout.write("--> Partitioning records (90% train, 5% dev, 5% test)...\n")
    dataset = partition_records(
        records,
        seed=args.seed,
        max_eval_speaker_hours=args.max_eval_speaker_hours,
    )
    report = dataset.report()

    save_manifest(dataset.train, args.out_dir / "train.jsonl")
    save_manifest(dataset.dev, args.out_dir / "dev.jsonl")
    save_manifest(dataset.test, args.out_dir / "test.jsonl")
    save_manifest(dataset.test_phrase_disjoint, args.out_dir / "test_phrase_disjoint.jsonl")

    # 5. Build CTC Vocabulary from train split
    sys.stdout.write(f"--> Generating CTC vocabulary -> {args.vocab_path}\n")
    vocab = build_vocabulary(
        (r.target_text for r in dataset.train),
        sources=(r.raw_text for r in dataset.train),
    )
    vocab.write(args.vocab_path)

    report_data = report.as_dict()
    report_data["rejections"] = dict(sorted(rejections.items()))
    report_file = args.out_dir / "speech_data.stats.json"
    report_file.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("  Pipeline Completed Successfully\n")
    sys.stdout.write("=" * 70 + "\n")
    for split in report.splits:
        sys.stdout.write(
            f"  {split.name:22s} {split.clips:7,d} clips  {split.hours:8.2f} h  "
            f"{split.speakers:5,d} speakers\n"
        )
    sys.stdout.write(f"  Speaker isolation check: {report.speaker_overlaps}\n")
    sys.stdout.write(f"  Vocabulary classes: {len(vocab.as_mapping())}\n")
    sys.stdout.write(f"  Stats file: {report_file}\n")
    sys.stdout.write("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
