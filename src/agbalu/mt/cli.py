"""Build the MT corpora. Training and synthesis run on Modal, never here.

`corpus` builds the fine-tuning set; `pivot` selects the sentences whose English or French
side can carry a translation into a third language, which `modal_app.synth` then generates.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final

from agbalu.bench.lid_systems import LidModelError
from agbalu.bench.lid_systems import build as build_identifier
from agbalu.mt.consistency import common_words, measure
from agbalu.mt.data import DEV_PAIRS, OUTPUT_DIR, PARALLEL_DIR, build
from agbalu.mt.pivot import OUTPUT as PIVOT_OUTPUT
from agbalu.mt.pivot import build as build_pivot

LID_SYSTEM: Final = "nllb-lid218e"
"""Its label space is NLLB's own language codes, so a pivot side it calls `fra_Latn` is
the language the generation step will then declare as the source."""

log: Final = logging.getLogger("agbalu.mt")


DOCUMENTS: Final = Path("data/documents")
TRANSLATIONS: Final = Path("artifacts/translations")
CONSISTENCY_OUTPUT: Final = OUTPUT_DIR / "consistency.json"

CORPUS_SAMPLE: Final = 400_000
"""Rows of the training corpus read to rank each source language's own frequent words.

The whole file would work and takes minutes; the frequency ordering of the top few thousand
words is settled long before this."""


def _source_frequencies(corpus: Path) -> dict[str, frozenset[str]]:
    """The most frequent words of every source language present in the training corpus.

    From the corpus rather than a shipped stop list, so the exclusion is the distribution
    the model was fitted on and a third source language costs nothing to add.
    """
    texts: dict[str, list[str]] = {}
    with corpus.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if count >= CORPUS_SAMPLE:
                break
            row = json.loads(line)
            direction = str(row["direction"]).split("-")
            for language, side in zip(direction, ("source", "target"), strict=True):
                if language != "kab":
                    texts.setdefault(language, []).append(str(row[side]))
    return {language: common_words(sides) for language, sides in texts.items()}


def _consistency(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        log.error("%s missing; run `make mt TASK=corpus` first", args.corpus)
        return 1

    common = _source_frequencies(args.corpus)
    reports: list[dict[str, object]] = []
    for translated in sorted(args.translations.glob("*/*.txt")):
        language = translated.parent.name.split("-")[0]
        source = args.documents / language / translated.name
        if not source.is_file():
            log.warning("%s has no source at %s", translated, source)
            continue
        report = measure(
            translated.stem,
            source.read_text(encoding="utf-8"),
            translated.read_text(encoding="utf-8"),
            common.get(language, frozenset()),
        )
        reports.append({"direction": translated.parent.name, **report.as_dict()})
        log.info(
            "%-28s %-8s segments %5d | skipped %5.1f%% | terms %4d | consistency %.3f",
            report.document,
            translated.parent.name,
            report.segments,
            100 * report.skip_rate,
            len(report.terms),
            report.consistency,
        )

    if not reports:
        log.error("no translated documents under %s", args.translations)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", args.output)
    return 0


def _corpus(args: argparse.Namespace) -> int:
    if not args.parallel_dir.is_dir():
        log.error("%s missing; run `make parallel` first", args.parallel_dir)
        return 1

    stats = build(
        args.parallel_dir,
        args.output_dir,
        include_mined=args.include_mined,
        dev_pairs=args.dev_pairs,
        seed=args.seed,
    )
    log.info(
        "read %s | mined %s | hard %s | duplicate %s -> %s pairs, %s examples",
        f"{stats['read']:,}",
        f"{stats['mined_excluded']:,}",
        f"{stats['hard_defective']:,}",
        f"{stats['duplicate']:,}",
        f"{stats['kept_pairs']:,}",
        f"{stats['examples']:,}",
    )
    log.info("train %s | dev %s", f"{stats['train']:,}", f"{stats['dev']:,}")
    for direction, count in sorted(stats["by_direction"].items()):
        log.info("  %-9s %s", direction, f"{count:,}")
    return 0


def _pivot(args: argparse.Namespace) -> int:
    if not args.parallel_dir.is_dir():
        log.error("%s missing; run `make parallel` first", args.parallel_dir)
        return 1
    try:
        identifier = build_identifier(LID_SYSTEM)
    except LidModelError:
        log.exception("run `make acquire-siblings` and install the models extra")
        return 1

    stats = build_pivot(identifier, args.parallel_dir, args.output)
    log.info(
        "read %s | mined %s | hard %s -> %s pivot sentences",
        f"{stats['read']:,}",
        f"{stats['mined_excluded']:,}",
        f"{stats['hard_defective']:,}",
        f"{stats['kept']:,}",
    )
    for side, dropped in sorted(stats["wrong_language"].items()):
        log.info("  %s dropped by language id  %s", side, f"{dropped:,}")
    log.info(
        "two-teacher %s | eng only %s | fra only %s",
        f"{stats['two_teacher']:,}",
        f"{stats['eng_only']:,}",
        f"{stats['fra_only']:,}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-dir", type=Path, default=PARALLEL_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="the fine-tuning train/dev split")
    corpus.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    corpus.add_argument("--dev-pairs", type=int, default=DEV_PAIRS)
    corpus.add_argument("--seed", type=int, default=0)
    corpus.add_argument(
        "--include-mined",
        action="store_true",
        help="keep NLLB's own mined output, which it has already been trained on",
    )
    corpus.set_defaults(run=_corpus)

    pivot = sub.add_parser("pivot", help="sentences whose pivot side can carry a third language")
    pivot.add_argument("--output", type=Path, default=PIVOT_OUTPUT)
    pivot.set_defaults(run=_pivot)

    stable = sub.add_parser(
        "consistency", help="whether a translated document renders a term the same way twice"
    )
    stable.add_argument("--documents", type=Path, default=DOCUMENTS)
    stable.add_argument("--translations", type=Path, default=TRANSLATIONS)
    stable.add_argument("--corpus", type=Path, default=OUTPUT_DIR / "train.jsonl")
    stable.add_argument("--output", type=Path, default=CONSISTENCY_OUTPUT)
    stable.set_defaults(run=_consistency)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
