"""Masinissa-Sentiment: the published config, and the calibrated classifier locally.

python3 -m agbalu.sentiment.cli release
python3 -m agbalu.sentiment.cli predict --text "Ulac d acu i ixuṣṣen, kulci yelha aṭas."

Training runs on Modal — `make modal-sentiment TASK=train` — because it fine-tunes the
encoder five times.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from agbalu.sentiment.infer import InferenceError, SentimentClassifier
from agbalu.sentiment.release import ReleaseError, write_config

log: Final = logging.getLogger("agbalu.sentiment")

CHECKPOINT_DIR: Final = Path("artifacts/masinissa-sentiment")
TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")


def command_release(args: argparse.Namespace) -> int:
    """Write the architecture and calibration record the checkpoint does not carry."""
    print(f"{write_config(args.dir)}")
    return 0


def command_predict(args: argparse.Namespace) -> int:
    lines = (
        [args.text]
        if args.text is not None
        else [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
    )
    classifier = SentimentClassifier.load(args.dir, args.tokenizer, device=args.device)
    for prediction in classifier.predict(lines):
        print(json.dumps(prediction.as_dict(), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu.sentiment", description=__doc__)
    parser.add_argument("--dir", type=Path, default=CHECKPOINT_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    release = commands.add_parser("release", help="write config.json for the published repo")
    release.set_defaults(func=command_release)

    predict = commands.add_parser("predict", help="classify text, or stdin one line each")
    predict.add_argument("--text", default=None)
    predict.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(func=command_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except (InferenceError, ReleaseError) as error:
        log.error("%s", error)
        return 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
