"""Command-line interface for document OCR, published as `agbalu/Feraoun-36M`.

Training runs on Modal (`make modal-ocr`). What this offers locally is the renderer,
inference over a scan or a whole scanned book, and the held-out evaluation the card's
numbers come from.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

from agbalu.ocr.dataset import (
    SyntheticDataset,
    build_dual_script_lines,
    chunk_text_into_lines,
    collate_lines,
    load_corpus_sentences,
)
from agbalu.ocr.evaluate import EvaluationError, evaluate_ocr_model
from agbalu.ocr.infer import Recognizer
from agbalu.ocr.synthetic import get_available_fonts, render_text_line

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("agbalu.ocr.cli")

DEFAULT_CORPUS = Path("data/processed/text/agbalu-text-v1.jsonl")
HELDOUT_TIFINAGH = Path("data/tasks/tifinagh/script_conversion/test.parquet")
"""The **test** split. Training reads `train.parquet`, so scoring the dual-script condition
on that file measures what the model was fitted on."""
DEFAULT_CHECKPOINT = Path("artifacts/runs/feraoun-36m/best.pt")
DEFAULT_RESULT = Path("data/processed/bench/feraoun-v1-heldout.json")

TRAINED_PREFIX = 100_000
"""Where a held-out draw begins. The training loader reads `max_sentences` from the head of
a source-ordered file, so every sentence inside this prefix has been seen."""

EVAL_SEED = 4711


class CorpusError(FileNotFoundError):
    """The corpus a command needs is not on this machine."""


def _read_corpus_lines(input_path: Path, max_lines: int | None = None) -> list[str]:
    """Sentences from a JSONL, TXT or parquet corpus, chunked into document lines.

    Raises on a missing corpus. A renderer given a handful of repeated sentences produces
    plausible images and a meaningless dataset.
    """
    if not input_path.exists():
        message = f"{input_path} not found; run `make extract` to build the corpus"
        raise CorpusError(message)

    sentences: list[str] = []
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{"):
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                text = str(obj.get("text", obj.get("kab", ""))).strip()
                if text:
                    sentences.append(text)
            else:
                sentences.append(s)
            if max_lines and len(sentences) >= max_lines:
                break
    return chunk_text_into_lines(sentences)


def handle_generate(args: argparse.Namespace) -> None:
    """Generate synthetic degraded line images for inspection or offline training."""
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = _read_corpus_lines(Path(args.input), max_lines=args.lines)
    log.info("Rendering %d synthetic Kabyle text lines into %s...", len(lines), out_dir)

    for i, line in enumerate(lines):
        img = render_text_line(line, augment=True)
        img_path = out_dir / f"line_{i:06d}.png"
        img.save(img_path)
        with (out_dir / f"line_{i:06d}.txt").open("w", encoding="utf-8") as f:
            f.write(line)

    log.info("Generated %d synthetic document line images.", len(lines))


def handle_evaluate(args: argparse.Namespace) -> None:
    """Score a checkpoint on a held-out draw and write the result beside the other benchmarks.

    The draw starts past `TRAINED_PREFIX` and is seeded, so it is reproducible and disjoint
    from what the release read. The renderer is the un-augmented, seeded one, which is what
    makes two checkpoints comparable on this set at all.
    """
    import torch

    corpus = Path(args.input)
    if not corpus.is_file():
        message = f"{corpus} not found; run `make extract` to build the corpus"
        raise CorpusError(message)

    sentences = load_corpus_sentences(corpus, max_sentences=TRAINED_PREFIX + args.pool)
    unseen = sentences[TRAINED_PREFIX:]
    if len(unseen) < args.lines:
        message = (
            f"{corpus} holds {len(unseen)} sentences past the trained prefix, "
            f"fewer than the {args.lines} asked for"
        )
        raise CorpusError(message)

    lines = chunk_text_into_lines(unseen)
    if args.tifinagh_ratio > 0.0:
        tifinagh = load_corpus_sentences(Path(args.tifinagh_input), max_sentences=args.pool)
        lines = build_dual_script_lines(unseen, tifinagh, tifinagh_ratio=args.tifinagh_ratio)
    drawn = random.Random(EVAL_SEED).sample(lines, min(args.lines, len(lines)))

    device = torch.device(args.device)
    recognizer = Recognizer.load(checkpoint_path=args.checkpoint, device=device)
    loader = DataLoader(
        SyntheticDataset(drawn, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_lines,
    )

    log.info("Scoring %s over %d held-out lines on %s", args.checkpoint, len(drawn), device)
    metrics = evaluate_ocr_model(recognizer.model, loader, device=device, max_batches=None)

    payload = {
        "checkpoint": str(args.checkpoint),
        "corpus": str(corpus),
        "lines": len(drawn),
        "trained_prefix_sentences": TRAINED_PREFIX,
        "seed": EVAL_SEED,
        "tifinagh_ratio": args.tifinagh_ratio,
        "device": str(device),
        "fonts": sorted(get_available_fonts()),
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "exact_match": metrics["exact_match"],
        "diacritic_precision": metrics["diacritic_precision"],
        "diacritic_recall": metrics["diacritic_recall"],
        "diacritic_f1": metrics["diacritic_f1"],
        "diacritic_support": metrics["diacritic_support"],
        "val_loss": metrics["val_loss"],
        "samples": [
            {"reference": reference, "hypothesis": hypothesis}
            for reference, hypothesis in metrics["samples"]
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log.info(
        "CER %.4f | WER %.4f | EM %.2f%% | diacritic F1 %.4f over %d glyphs -> %s",
        payload["cer"],
        payload["wer"],
        float(payload["exact_match"]) * 100.0,
        payload["diacritic_f1"],
        payload["diacritic_support"],
        out,
    )


def handle_infer(args: argparse.Namespace) -> None:
    """Transcribe a document line or page image."""
    img_path = Path(args.image)
    if not img_path.is_file():
        message = f"image {img_path} not found"
        raise FileNotFoundError(message)

    ocr = Recognizer.load(checkpoint_path=args.checkpoint)
    img = Image.open(img_path)
    log.info("Transcription: %s", ocr.recognize_page(img) if args.page else ocr.recognize_line(img))


def handle_transcribe_book(args: argparse.Namespace) -> None:
    """Transcribe an entire scanned Kabyle book directory."""
    from agbalu.ocr.adlis import transcribe_book

    book_dir = Path(args.book_dir)
    if not book_dir.is_dir():
        message = f"book directory {book_dir} not found"
        raise NotADirectoryError(message)

    ocr = Recognizer.load(checkpoint_path=args.checkpoint)
    transcribe_book(
        book_dir=book_dir,
        ocr=ocr,
        output_path=Path(args.output) if args.output else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kabyle document OCR (Feraoun-36M)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="render degraded line images")
    generate.add_argument("--input", type=Path, default=DEFAULT_CORPUS, help="corpus path")
    generate.add_argument("--output", default="data/ocr/synthetic", help="output directory")
    generate.add_argument("--lines", type=int, default=100, help="number of lines")

    evaluate = subparsers.add_parser("evaluate", help="score a checkpoint on a held-out draw")
    evaluate.add_argument("--input", type=Path, default=DEFAULT_CORPUS, help="corpus path")
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    evaluate.add_argument("--lines", type=int, default=500, help="held-out lines to score")
    evaluate.add_argument(
        "--pool",
        type=int,
        default=20_000,
        help="sentences to read past the trained prefix before drawing",
    )
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--tifinagh-ratio", type=float, default=0.0)
    evaluate.add_argument("--tifinagh-input", type=Path, default=HELDOUT_TIFINAGH)

    infer = subparsers.add_parser("infer", help="transcribe a line or page image")
    infer.add_argument("--image", required=True, help="path to an image file")
    infer.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    infer.add_argument("--page", action="store_true", help="segment a full page first")

    book = subparsers.add_parser("transcribe-book", help="transcribe a scanned book directory")
    book.add_argument("--book-dir", required=True, help="directory holding crops/")
    book.add_argument("--output", default=None, help="output transcription file")
    book.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)

    return parser


HANDLERS = {
    "generate": handle_generate,
    "evaluate": handle_evaluate,
    "infer": handle_infer,
    "transcribe-book": handle_transcribe_book,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        HANDLERS[args.command](args)
    except (OSError, EvaluationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
