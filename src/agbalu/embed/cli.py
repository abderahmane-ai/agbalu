"""Kabyle sentence-embedding CLI.

python -m agbalu.embed.cli coverage
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from transformers import AutoTokenizer

from agbalu.embed.backbone import CANDIDATES, encoder, unknown_id, widen
from agbalu.embed.vocabulary import Coverage, coverage
from agbalu.llm.fertility import sample
from agbalu.normalise import Normaliser

DEFAULT_TEXT: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
DEFAULT_COVERAGE: Final = Path("data/processed/embed/coverage.stats.json")
SAMPLE_SIZE: Final = 20_000
SAMPLE_SEED: Final = 0

log: Final = logging.getLogger("agbalu.embed")


def _condition(measured: Coverage) -> dict[str, float | int]:
    return {
        "tokens_per_word": round(measured.tokens_per_word, 4),
        "unknown_rate": round(measured.unknown_rate, 6),
        "words": measured.words,
    }


def measure_coverage(raw: Sequence[str], normalised: Sequence[str]) -> dict[str, object]:
    """The 2x2 grid per candidate: raw or normalised text, stock or repaired vocabulary.

    Both axes are reported because neither substitutes for the other. Normalisation alone
    moves the unknown rate the wrong way: it rewrites Cyrillic `Ԑ` onto Latin `Ɛ`, which
    the stock vocabularies cannot encode either.
    """
    results: list[dict[str, object]] = []
    for label, name in CANDIDATES.items():
        tokenizer = AutoTokenizer.from_pretrained(name)
        unk = unknown_id(tokenizer)
        stock_raw = coverage(encoder(tokenizer), unk, raw)
        stock_clean = coverage(encoder(tokenizer), unk, normalised)
        repaired = widen(tokenizer)
        fixed_raw = coverage(encoder(tokenizer), unk, raw)
        fixed_clean = coverage(encoder(tokenizer), unk, normalised)
        results.append(
            {
                "candidate": label,
                "source": name,
                "vocabulary_before": repaired.vocabulary_before,
                "vocabulary_after": repaired.vocabulary_after,
                "missing": list(repaired.added),
                "donors": dict(repaired.donors),
                "conditions": {
                    "raw_stock": _condition(stock_raw),
                    "normalised_stock": _condition(stock_clean),
                    "raw_repaired": _condition(fixed_raw),
                    "normalised_repaired": _condition(fixed_clean),
                },
            }
        )
        log.info(
            "%-22s missing %d (%s)  unknown %.4f%% -> %.4f%%",
            label,
            len(repaired.added),
            " ".join(repaired.added) or "none",
            100 * stock_raw.unknown_rate,
            100 * fixed_clean.unknown_rate,
        )
    return {"candidates": results}


def command_coverage(args: argparse.Namespace) -> int:
    raw = sample(args.text, SAMPLE_SIZE, seed=SAMPLE_SEED)
    normaliser = Normaliser()
    normalised = [normaliser.normalise(sentence) for sentence in raw]
    changed = sum(1 for before, after in zip(raw, normalised, strict=True) if before != after)
    log.info("sampled %d sentences, %d changed by normalisation", len(raw), changed)

    report = measure_coverage(raw, normalised)
    report["sample"] = {"sentences": len(raw), "requested": SAMPLE_SIZE, "seed": SAMPLE_SEED}
    report["normaliser"] = normaliser.version
    report["normalisation_changed"] = changed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="agbalu.embed", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cover = sub.add_parser("coverage", help="what each candidate vocabulary does to Kabyle")
    cover.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    cover.add_argument("--output", type=Path, default=DEFAULT_COVERAGE)
    cover.set_defaults(handler=command_coverage)

    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
