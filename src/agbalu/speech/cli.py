"""Fadhma 2 CLI: Ingestion, Partitioning, Smoke Drill, Training, and Evaluation.

Usage:
    # 1. Ingest multi-source speech data with Dihya filtering:
    python3 -m agbalu.speech.cli manifest --cv-tsv data/raw/cv/validated.tsv --dihya

    # 2. Enforce 90/5/5 speaker-disjoint partition:
    python3 -m agbalu.speech.cli partition --manifest data/processed/speech/manifest.jsonl

    # 3. Build CTC vocabulary:
    python3 -m agbalu.speech.cli vocabulary

    # 4. Mandatory 20-step smoke drill on H100 MIG slice:
    python3 -m agbalu.speech.cli smoke --steps 20

    # 5. Full training on HPC:
    python3 -m agbalu.speech.cli train --config artifacts/runs/fadhma-v2/config.json

    # 6. Benchmark evaluation and silence challenge:
    python3 -m agbalu.speech.cli evaluate --model-dir artifacts/runs/fadhma-v2/best_model

    # 7. Standalone inference:
    python3 -m agbalu.speech.cli transcribe path/to/audio.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Final

from agbalu.speech import vocabulary
from agbalu.speech.decode import SpeechDecoder
from agbalu.speech.evaluate import run_full_benchmark
from agbalu.speech.lm import (
    LM_ORDER,
    LOCAL_LM_ARPA,
    LOCAL_LM_BINARY,
    build_arpa,
    compile_binary,
)
from agbalu.speech.manifest import (
    AudioRecord,
    filter_with_dihya,
    ingest_common_voice_tsv,
    ingest_tatoeba_tsv,
    ingest_tutlayt_jsonl,
    load_manifest,
    read_durations_tsv,
    save_manifest,
)
from agbalu.speech.partition import partition_records
from agbalu.speech.smoke import run_smoke_drill
from agbalu.speech.train import TrainingConfig, train

PROCESSED_DIR: Final = Path("data/processed/speech")
DEFAULT_MANIFEST: Final = PROCESSED_DIR / "manifest.jsonl"
DEFAULT_VOCAB: Final = Path("artifacts/asr/vocab.json")
DEFAULT_RUN_DIR: Final = Path("artifacts/runs/fadhma-v2")
DEFAULT_CV_TSV: Final = "data/raw/hf.fsicoli.common-voice-22-kab/transcript/kab/validated.tsv"
DEFAULT_CV_DURATIONS: Final = (
    "data/raw/hf.fsicoli.common-voice-22-kab/transcript/kab/clip_durations.tsv"
)


def command_manifest(args: argparse.Namespace) -> int:
    """Ingest multi-source collections, apply text normalisation, and filter via Dihya."""
    records: list[AudioRecord] = []
    rejections: Counter[str] = Counter()

    if args.cv_tsv and Path(args.cv_tsv).is_file():
        durations = (
            read_durations_tsv(Path(args.cv_durations))
            if args.cv_durations and Path(args.cv_durations).is_file()
            else None
        )
        print(f"[Manifest] Ingesting Common Voice from {args.cv_tsv}...")
        records.extend(
            ingest_common_voice_tsv(
                Path(args.cv_tsv),
                durations_map=durations,
                rejections=rejections,
            )
        )

    if args.tatoeba_tsv and Path(args.tatoeba_tsv).is_file():
        print(f"[Manifest] Ingesting Tatoeba from {args.tatoeba_tsv}...")
        records.extend(ingest_tatoeba_tsv(Path(args.tatoeba_tsv), rejections=rejections))

    if args.tutlayt_jsonl and Path(args.tutlayt_jsonl).is_file():
        print(f"[Manifest] Ingesting TutlaytAI from {args.tutlayt_jsonl}...")
        records.extend(
            ingest_tutlayt_jsonl(
                Path(args.tutlayt_jsonl), filter_kabyle_only=True, rejections=rejections
            )
        )

    print(f"[Manifest] Ingested {len(records):,} initial candidate speech records.")

    if args.dihya:
        print("[Manifest] Running Dihya-5M dialect filter to eliminate sibling varieties...")
        filtered = list(
            filter_with_dihya(
                records,
                dihya_dir=Path(args.dihya_dir) if args.dihya_dir else None,
                min_kabyle_confidence=args.dihya_confidence,
                rejections=rejections,
            )
        )
        print(f"[Manifest] Retained {len(filtered):,} records after Dihya verification.")
        records = filtered

    out_path = Path(args.out)
    report = save_manifest(records, out_path)
    report_dict = report.as_dict()
    report_dict["rejected_counts"] = dict(sorted(rejections.items()))

    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")

    info = (
        f"{report.total_records:,} clips, {report.total_hours:.2f} h, "
        f"{report.unique_speakers} speakers"
    )
    print(f"-> Manifest: {out_path} ({info})")
    print(f"-> Stats: {stats_path}")
    if rejections:
        print("  Rejections:")
        for r, cnt in sorted(rejections.items()):
            print(f"    - {r}: {cnt:,}")
    return 0


def command_partition(args: argparse.Namespace) -> int:
    """Enforce 90/5/5 speaker-disjoint partition across manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"error: manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    records = load_manifest(manifest_path)
    print(f"[Partition] Partitioning {len(records):,} records with seed {args.seed}...")
    dataset = partition_records(
        records, dev_ratio=args.dev_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    report = dataset.report()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(dataset.train, out_dir / "train.jsonl")
    save_manifest(dataset.dev, out_dir / "dev.jsonl")
    save_manifest(dataset.test, out_dir / "test.jsonl")
    save_manifest(dataset.test_phrase_disjoint, out_dir / "test_phrase_disjoint.jsonl")

    report_path = out_dir / "partition.stats.json"
    report_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")

    print("[Partition] Completed successfully:")
    for s in report.splits:
        print(f"  {s.name:22s} {s.clips:7,d} clips  {s.hours:8.2f} h  {s.speakers:5,d} speakers")
    print(f"  Speaker overlaps: {report.speaker_overlaps}")
    print(f"-> Stats: {report_path}")
    return 0


def command_vocabulary(args: argparse.Namespace) -> int:
    """Derive CTC vocabulary from partitioned train/dev splits."""
    train_manifest = Path(args.train_manifest)
    if not train_manifest.is_file():
        print(f"error: train manifest not found at {train_manifest}", file=sys.stderr)
        return 1

    records = load_manifest(train_manifest)
    built = vocabulary.build(
        (r.target_text for r in records), sources=(r.raw_text for r in records)
    )
    vocab_path = Path(args.vocab)
    built.write(vocab_path)

    stats_path = vocab_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(built.as_dict(), indent=2) + "\n", encoding="utf-8")

    n_cls = len(built.as_mapping())
    print(f"[Vocabulary] Built {n_cls} CTC classes from {len(records):,} transcripts.")
    print(f"  Characters: {''.join(c for c in built.characters if c != ' ')}")
    if built.unexpected:
        print(f"  Audit unexpected dropped: {dict(sorted(built.unexpected.items()))}")
    print(f"-> {vocab_path}")
    print(f"-> {stats_path}")
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    """Execute mandatory 20-step smoke validation drill on target GPU."""
    train_manifest = Path(args.train_manifest)
    vocab_file = Path(args.vocab)
    out_dir = Path(args.out_dir)

    if not train_manifest.is_file():
        print(f"error: train manifest missing: {train_manifest}", file=sys.stderr)
        return 1
    if not vocab_file.is_file():
        print(f"error: vocabulary file missing: {vocab_file}", file=sys.stderr)
        return 1

    report = run_smoke_drill(train_manifest, vocab_file, out_dir, device_str=args.device)
    return 0 if report.resumed_cleanly else 1


def command_train(args: argparse.Namespace) -> int:
    """Execute full Fadhma 2 training loop."""
    config = TrainingConfig(
        train_manifest=str(args.train_manifest),
        dev_manifest=str(args.dev_manifest),
        vocab_file=str(args.vocab),
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        seed=args.seed,
    )
    train(config)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    """Execute benchmark evaluation across clean, phrase-disjoint, and silence sets."""
    model_dir = Path(args.model_dir)
    vocab_file = Path(args.vocab)
    clean_manifest = Path(args.test_manifest)
    phrase_manifest = (
        Path(args.phrase_manifest)
        if args.phrase_manifest and Path(args.phrase_manifest).is_file()
        else None
    )

    decoder = SpeechDecoder.load(
        model_dir,
        vocab_file,
        lm_path=Path(args.lm) if args.lm else None,
        device_str=args.device,
    )

    report_path = model_dir / "evaluation_report.json"
    report = run_full_benchmark(
        decoder,
        clean_manifest,
        phrase_disjoint_manifest=phrase_manifest,
        output_report_path=report_path,
    )

    print("\n[Evaluation Complete]")
    print(
        f"  Clean Test CER: {report.clean.cer_percent:.2f}% | WER: {report.clean.wer_percent:.2f}%"
    )
    if report.phrase_disjoint:
        print(
            f"  Phrase-Disjoint CER: {report.phrase_disjoint.cer_percent:.2f}% | "
            f"WER: {report.phrase_disjoint.wer_percent:.2f}%"
        )
    sil = report.silence_challenge
    print(f"  Silence Hallucinations: {sil.insertions}/{sil.total_probes} (Passed: {sil.passed})")
    print(f"-> {report_path}")
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    """Transcribe an audio file using standalone decoder."""
    model_dir = Path(args.model_dir)
    vocab_file = Path(args.vocab)
    audio_path = Path(args.audio)

    decoder = SpeechDecoder.load(
        model_dir,
        vocab_file,
        lm_path=Path(args.lm) if args.lm else None,
        device_str=args.device,
    )

    res = decoder.transcribe_file(audio_path, use_lm=not args.greedy)
    print(res.text)
    return 0


def command_lm(args: argparse.Namespace) -> int:
    """Build the KenLM 5-gram language model."""
    if not Path(args.text).exists():
        print(f"error: {args.text} not found", file=sys.stderr)
        return 1

    plain_path = Path(args.plain)
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    records = lines = 0
    with (
        plain_path.open("w", encoding="utf-8") as sink,
        Path(args.text).open(encoding="utf-8") as src,
    ):
        for line in src:
            t = json.loads(line).get("text", "").strip()
            if t:
                sink.write(t + "\n")
                records += 1
                lines += t.count("\n") + 1
    print(f"  {lines:,} lines from {records:,} records -> {args.plain}")

    build_arpa(plain_path, Path(args.arpa), order=args.order)
    compile_binary(Path(args.arpa), Path(args.binary))
    if not args.keep_arpa and Path(args.arpa).exists():
        Path(args.arpa).unlink()
    print(f"-> LM Binary: {args.binary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agbalu-speech", description="Fadhma 2 Sovereign Kabyle ASR Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Manifest
    m_cmd = sub.add_parser("manifest", help="build multi-source speech manifest")
    m_cmd.add_argument("--cv-tsv", default=DEFAULT_CV_TSV)
    m_cmd.add_argument("--cv-durations", default=DEFAULT_CV_DURATIONS)
    m_cmd.add_argument("--tatoeba-tsv", default=None)
    m_cmd.add_argument("--tutlayt-jsonl", default=None)
    m_cmd.add_argument("--dihya", action="store_true", help="filter transcripts through Dihya-5M")
    m_cmd.add_argument("--dihya-dir", default="artifacts/runs/dihya")
    m_cmd.add_argument("--dihya-confidence", type=float, default=0.70)
    m_cmd.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)

    # Partition
    p_cmd = sub.add_parser("partition", help="partition manifest into 90/5/5 disjoint splits")
    p_cmd.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_cmd.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p_cmd.add_argument("--dev-ratio", type=float, default=0.05)
    p_cmd.add_argument("--test-ratio", type=float, default=0.05)
    p_cmd.add_argument("--seed", type=int, default=42)

    # Vocabulary
    v_cmd = sub.add_parser("vocabulary", help="build CTC vocabulary from train manifest")
    v_cmd.add_argument("--train-manifest", type=Path, default=PROCESSED_DIR / "train.jsonl")
    v_cmd.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)

    # LM
    lm_cmd = sub.add_parser("lm", help="build KenLM 5-gram language model")
    lm_cmd.add_argument("--text", default="data/processed/text/agbalu-text-v1.jsonl")
    lm_cmd.add_argument("--plain", default=PROCESSED_DIR / "lm_sentences.txt")
    lm_cmd.add_argument("--arpa", default=LOCAL_LM_ARPA)
    lm_cmd.add_argument("--binary", default=LOCAL_LM_BINARY)
    lm_cmd.add_argument("--order", type=int, default=LM_ORDER)
    lm_cmd.add_argument("--keep-arpa", action="store_true")

    # Smoke
    s_cmd = sub.add_parser("smoke", help="run 20-step smoke drill on target GPU")
    s_cmd.add_argument("--train-manifest", type=Path, default=PROCESSED_DIR / "train.jsonl")
    s_cmd.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    s_cmd.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR / "smoke")
    s_cmd.add_argument("--device", default=None)

    # Train
    t_cmd = sub.add_parser("train", help="run full training loop")
    t_cmd.add_argument("--train-manifest", type=Path, default=PROCESSED_DIR / "train.jsonl")
    t_cmd.add_argument("--dev-manifest", type=Path, default=PROCESSED_DIR / "dev.jsonl")
    t_cmd.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    t_cmd.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    t_cmd.add_argument("--max-steps", type=int, default=6400)
    t_cmd.add_argument("--seed", type=int, default=42)

    # Evaluate
    e_cmd = sub.add_parser("evaluate", help="run benchmark evaluation")
    e_cmd.add_argument("--model-dir", type=Path, default=DEFAULT_RUN_DIR / "best_model")
    e_cmd.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    e_cmd.add_argument("--test-manifest", type=Path, default=PROCESSED_DIR / "test.jsonl")
    e_cmd.add_argument(
        "--phrase-manifest", type=Path, default=PROCESSED_DIR / "test_phrase_disjoint.jsonl"
    )
    e_cmd.add_argument("--lm", type=Path, default=LOCAL_LM_BINARY)
    e_cmd.add_argument("--device", default=None)

    # Transcribe
    tr_cmd = sub.add_parser("transcribe", help="transcribe audio file")
    tr_cmd.add_argument("audio", type=Path)
    tr_cmd.add_argument("--model-dir", type=Path, default=DEFAULT_RUN_DIR / "best_model")
    tr_cmd.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    tr_cmd.add_argument("--lm", type=Path, default=None)
    tr_cmd.add_argument("--greedy", action="store_true")
    tr_cmd.add_argument("--device", default=None)

    return parser


COMMANDS: Final[dict[str, Callable[[argparse.Namespace], int]]] = {
    "manifest": command_manifest,
    "partition": command_partition,
    "vocabulary": command_vocabulary,
    "lm": command_lm,
    "smoke": command_smoke,
    "train": command_train,
    "evaluate": command_evaluate,
    "transcribe": command_transcribe,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
