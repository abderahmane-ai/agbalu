"""Command line for the Berber language identifier.

python -m agbalu.lid.cli train
python -m agbalu.lid.cli evaluate
python -m agbalu.lid.cli calibrate
python -m agbalu.lid.cli predict --text "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch

from agbalu.lid.calibrate import CalibrationError, calibrate, save
from agbalu.lid.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_EVAL_PATH,
    ConfigError,
    ModelConfig,
    TrainConfig,
)
from agbalu.lid.dataset import Corpus, CorpusError, Record, read_split
from agbalu.lid.infer import Identifier, InferenceError
from agbalu.lid.train import TrainingError, build_corpora, held_out_report, train_identifier

log: Final = logging.getLogger("agbalu.lid")

DEFAULTS: Final = TrainConfig()
"""An instance, not the class: `slots=True` makes class-level attribute access return
the slot descriptor, so `TrainConfig.seed` as an argparse default silently passes a
`member_descriptor` into `torch.manual_seed`."""


def held_out_records(path: Path) -> list[Record]:
    """The held-out set as corpus rows, so it is scored by the same code as dev."""
    if not path.is_file():
        msg = f"held-out set not found: {path} — run `make bench TASK=lid`"
        raise CorpusError(msg)
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            language, domain = row["language"], row.get("domain", "unknown")
            records.append(
                Record(
                    text=row["text"],
                    language=language,
                    domain=domain,
                    group=f"{language}/{domain}",
                    source=row.get("source", "unknown"),
                )
            )
    if not records:
        msg = f"{path} carried no rows"
        raise CorpusError(msg)
    return records


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def command_train(args: argparse.Namespace) -> int:
    model_config = ModelConfig()
    settings = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        group_dro_step=args.group_dro_step,
        seed=args.seed,
    )
    summary = train_identifier(
        model_config=model_config,
        settings=settings,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=resolve_device(args.device),
        resume=not args.no_resume,
    )
    if summary.best is None:
        log.warning("no checkpoint was selected; the run completed no validation")
        return 1
    log.info("best dev score=%.4f macro_f1=%.4f", summary.best.score, summary.best.macro_f1)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    identifier = Identifier.load(args.checkpoint_dir, device=str(device))
    train_records = read_split(args.data_dir / "train.jsonl")
    groups = Corpus.observed_groups(train_records)
    report = held_out_report(
        identifier.model,
        held_out_records(args.eval_path),
        groups,
        batch_size=args.batch_size,
        device=device,
    )
    destination = args.checkpoint_dir / "held-out-report.json"
    destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    worst, recall = report.worst_group
    log.info(
        "held-out accuracy=%.4f macro_f1=%.4f worst=%s %.4f written=%s",
        report.accuracy,
        report.macro_f1,
        worst,
        recall,
        destination,
    )
    return 0


def stored_settings(checkpoint_dir: Path) -> TrainConfig:
    """The `TrainConfig` the checkpoint was trained under, not this session's defaults.

    Calibration reads the class counts and the logit-adjustment `tau` from it; taking
    them from a fresh default would calibrate one model against another's prior.
    """
    path = checkpoint_dir / "config.json"
    if not path.is_file():
        msg = f"config not found at {path}; the checkpoint's settings are unknown"
        raise CorpusError(msg)
    stored = json.loads(path.read_text(encoding="utf-8"))["train"]
    known = set(TrainConfig.__dataclass_fields__)
    return TrainConfig(**{key: value for key, value in stored.items() if key in known})


def command_calibrate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    identifier = Identifier.load(args.checkpoint_dir, device=str(device))
    settings = stored_settings(args.checkpoint_dir)
    _, dev = build_corpora(args.data_dir, identifier.model.config, settings)
    calibration = calibrate(
        identifier.model,
        dev,
        settings=settings,
        target_coverage=args.coverage,
        device=str(device),
    )
    save(calibration, args.checkpoint_dir / "calibration.json")
    log.info(
        "calibrated q_hat=%.4f fitted_on=%d measured_on=%d coverage=%.4f mean_set=%.2f",
        calibration.q_hat,
        calibration.fitted_on,
        calibration.measured_on,
        calibration.empirical_coverage,
        calibration.mean_set_size,
    )
    return 0


def command_predict(args: argparse.Namespace) -> int:
    identifier = Identifier.load(args.checkpoint_dir, device=str(resolve_device(args.device)))
    for text, prediction in zip(args.text, identifier.predict(args.text), strict=True):
        ranked = sorted(prediction.probabilities.items(), key=lambda item: -item[1])
        log.info(
            "%s -> %s %.4f set=%s top=%s",
            text[:60],
            prediction.language,
            prediction.confidence,
            list(prediction.prediction_set),
            [f"{name} {value:.3f}" for name, value in ranked[:3]],
        )
    return 0


COMMANDS: Final = {
    "train": command_train,
    "evaluate": command_evaluate,
    "calibrate": command_calibrate,
    "predict": command_predict,
}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Berber language identification.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="auto")
    sub = parser.add_subparsers(dest="command", required=True)

    trainer = sub.add_parser("train", help="train and select on dev")
    trainer.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    trainer.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    trainer.add_argument("--contrastive-weight", type=float, default=DEFAULTS.contrastive_weight)
    trainer.add_argument("--group-dro-step", type=float, default=DEFAULTS.group_dro_step)
    trainer.add_argument("--seed", type=int, default=DEFAULTS.seed)
    trainer.add_argument("--no-resume", action="store_true")

    scorer = sub.add_parser("evaluate", help="score the held-out set once")
    scorer.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    scorer.add_argument("--batch-size", type=int, default=DEFAULTS.eval_batch_size)

    calibrator = sub.add_parser("calibrate", help="fit the conformal quantile on dev")
    calibrator.add_argument("--coverage", type=float, default=0.99)

    predictor = sub.add_parser("predict", help="classify one or more lines")
    predictor.add_argument("--text", nargs="+", required=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    args = parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ConfigError, CorpusError, CalibrationError, InferenceError, TrainingError) as error:
        log.error("%s: %s", type(error).__name__, error)  # noqa: TRY400 — CLI reports, not traces
        return 1


if __name__ == "__main__":
    sys.exit(main())
