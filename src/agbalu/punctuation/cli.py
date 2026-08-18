"""Punctuation and casing restoration over Fadhma's output.

python3 -m agbalu.punctuation.cli corpus
python3 -m agbalu.punctuation.cli train --epochs 3
python3 -m agbalu.punctuation.cli evaluate --split test
python3 -m agbalu.punctuation.cli --run artifacts/runs/punctuation-v2 restore --text "azul"

`--run` is a global option and precedes the subcommand. `restore` reads stdin one line per
utterance, so `agbalu.speech.cli transcribe` pipes straight into it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from agbalu.punctuation import corpus as corpus_module
from agbalu.punctuation.corpus import OUTPUT_DIR, CorpusError, read_split
from agbalu.punctuation.evaluate import Prediction, Report, render, score, trivial_baseline
from agbalu.punctuation.labels import annotate
from agbalu.punctuation.release import write_config
from agbalu.tokenizer.evaluate import load as load_tokenizer

log: Final = logging.getLogger("agbalu.punctuation")

TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
RUN_DIR: Final = Path("artifacts/runs/punctuation-v1")
RESULTS: Final = Path("data/processed/bench/punctuation.json")
RELEASE_CONFIG: Final = Path("artifacts/punctuation/config.json")


def command_corpus(args: argparse.Namespace) -> int:
    stats = corpus_module.build(args.text_corpus, args.speech_dir, args.out, limit=args.limit)
    for name, split in stats["splits"].items():
        log.info("%-5s %8d rows  %9d words", name, split["rows"], split["words"])
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    import torch

    from agbalu.model.trainer import select_device
    from agbalu.punctuation.dataset import encode_corpus
    from agbalu.punctuation.model import build as build_model
    from agbalu.punctuation.train import Trainer, TrainSettings

    device = select_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    settings = TrainSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
    )
    torch.manual_seed(settings.seed)

    train = encode_corpus(read_split(args.data / "train.jsonl"), tokenizer, args.max_length)
    dev = encode_corpus(read_split(args.data / "dev.jsonl"), tokenizer, args.max_length)
    log.info("encoded train=%d dev=%d rows", len(train), len(dev))

    model = build_model(args.encoder, dropout=settings.dropout, device=device)
    if settings.freeze_encoder:
        model.freeze_encoder()

    trainer = Trainer(model, train, dev, settings, args.run, device)
    if trainer.maybe_resume():
        log.info("resumed at step %d", trainer.state.step)
    summary = trainer.run()

    if summary.best is not None:
        log.info(
            "best loss=%.4f punctuation_f1=%.4f case_f1=%.4f",
            summary.best.loss,
            summary.best.punctuation_macro_f1,
            summary.best.case_noninitial_f1,
        )
    return 0


def _report(split: Path, args: argparse.Namespace) -> tuple[Report, Report]:
    import torch

    from agbalu.model.trainer import select_device
    from agbalu.punctuation.infer import predict, truncated_gold
    from agbalu.punctuation.model import build as build_model

    device = select_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    rows = read_split(split)
    model = build_model(args.encoder, device=device)
    state = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])

    restorations = predict(
        model, tokenizer, [row.text for row in rows], device=device, max_length=args.max_length
    )
    predictions = [
        Prediction(
            truncated_gold(row.text, len(restoration.words)),
            restoration.punctuation,
            restoration.case,
        )
        for row, restoration in zip(rows, restorations, strict=True)
    ]
    baseline = [trivial_baseline(annotate(row.text)) for row in rows]
    return score(predictions), score(baseline)


def command_evaluate(args: argparse.Namespace) -> int:
    split = args.data / f"{args.split}.jsonl"
    model_report, baseline_report = _report(split, args)
    print(render(baseline_report, f"BASELINE  capitalise-first + final period  [{args.split}]"))
    print()
    print(render(model_report, f"MODEL     {args.run / args.checkpoint}  [{args.split}]"))

    args.results.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "model": model_report, "baseline": baseline_report}
    args.results.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    log.info("wrote %s", args.results)
    return 0


def command_release(args: argparse.Namespace) -> int:
    """Write the architecture record the checkpoint does not carry."""
    written = write_config(args.out)
    print(f"{written}")
    return 0


def read_utterances(file: Path | None, text: str | None) -> list[str]:
    """One utterance per line, from a file, an argument or stdin.

    Stdin is split rather than taken whole: an ASR transcript arrives one hypothesis per
    line, and reading it as a single string restores a whole batch as one sentence.
    """
    if file is not None:
        source = file.read_text(encoding="utf-8")
    elif text is not None:
        return [text]
    else:
        source = sys.stdin.read()
    return [line.strip() for line in source.splitlines() if line.strip()]


def command_restore(args: argparse.Namespace) -> int:
    import torch

    from agbalu.model.trainer import select_device
    from agbalu.punctuation.infer import predict
    from agbalu.punctuation.model import build as build_model

    device = select_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    lines = read_utterances(args.file, args.text)
    model = build_model(args.encoder, device=device)
    state = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])

    for restoration in predict(model, tokenizer, lines, device=device, max_length=args.max_length):
        print(restoration.text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu.punctuation", description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument("--device", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("corpus", help="build the decontaminated splits")
    build.add_argument("--text-corpus", type=Path, default=corpus_module.TEXT_CORPUS)
    build.add_argument("--speech-dir", type=Path, default=corpus_module.SPEECH_DIR)
    build.add_argument("--out", type=Path, default=OUTPUT_DIR)
    build.add_argument("--limit", type=int, default=None)
    build.set_defaults(func=command_corpus)

    train = commands.add_parser("train", help="fine-tune the encoder for both heads")
    train.add_argument("--encoder", type=Path, default=Path("artifacts/runs/agbalu-encoder-v1"))
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--encoder-lr", type=float, default=3e-5)
    train.add_argument("--head-lr", type=float, default=1e-3)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--freeze-encoder", action="store_true")
    train.add_argument("--seed", type=int, default=17)
    train.set_defaults(func=command_train)

    evaluate = commands.add_parser("evaluate", help="score a checkpoint against the rule baseline")
    evaluate.add_argument("--encoder", type=Path, default=Path("artifacts/runs/agbalu-encoder-v1"))
    evaluate.add_argument("--split", default="test", choices=("dev", "test", "ood"))
    evaluate.add_argument("--checkpoint", default="best.pt")
    evaluate.add_argument("--results", type=Path, default=RESULTS)
    evaluate.add_argument("--max-length", type=int, default=128)
    evaluate.set_defaults(func=command_evaluate)

    restore = commands.add_parser("restore", help="put marks and capitals back on ASR output")
    restore.add_argument("--encoder", type=Path, default=Path("artifacts/runs/agbalu-encoder-v1"))
    restore.add_argument("--text", default=None)
    restore.add_argument("--file", type=Path, default=None)
    restore.add_argument("--checkpoint", default="best.pt")
    restore.add_argument("--max-length", type=int, default=128)
    restore.set_defaults(func=command_restore)

    release = commands.add_parser("release", help="write config.json for the published repo")
    release.add_argument("--out", type=Path, default=RELEASE_CONFIG)
    release.set_defaults(func=command_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except CorpusError as error:
        log.error("%s", error)
        return 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
