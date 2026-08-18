"""Pronunciation CLI.

python -m agbalu.g2p.cli build
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Final

from agbalu.g2p.align import DEFAULT_G2P, G2PError, Pronunciations, align, read_pairs
from agbalu.normalise import Normaliser
from agbalu.tts.g2p import PhonemeError, phonemize_word


def _segments(ipa: str) -> int:
    return sum(1 for char in ipa if char not in "ˤː͡")


def repair_readings(word: str, readings: Counter[str]) -> Counter[str] | None:
    """Restore characters the source generator dropped without comment.

    It emits nothing for a character it has no rule for, losing `o` in 128 words and
    `ţ` — real Kabyle, `docs/orthography.md` §4 — in 73. A reading shorter than the
    reference table's is that defect, and only those are rewritten: the table does not
    model `i`/`u` laxing, so regenerating a sound entry would replace an attested
    allophone with an approximation.

    Returns `None` when the word is outside the writing system, which is a refusal to
    guess rather than a repair.
    """
    try:
        reference = phonemize_word(word)
    except PhonemeError:
        return None
    out: Counter[str] = Counter()
    for ipa, count in readings.items():
        out[reference if _segments(ipa) < _segments(reference) else ipa] += count
    return out


DEFAULT_OUT: Final = Path("data/processed/lexicon/agbalu-pronunciations-v1.jsonl")

TOP_AMBIGUOUS: Final = 10

log: Final = logging.getLogger("agbalu.g2p")


def command_build(args: argparse.Namespace) -> int:
    alignments, report = align(read_pairs(args.source))
    if not alignments:
        msg = f"no sentence aligned from {args.source}"
        raise G2PError(msg)
    lexicon = Pronunciations.from_alignments(alignments)
    ambiguous = lexicon.ambiguous()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".part")
    repaired_entries = 0
    unrepairable: list[str] = []
    phonemes: Counter[str] = Counter()
    with partial.open("w", encoding="utf-8") as handle:
        for word in sorted(lexicon.entries):
            readings = repair_readings(word, lexicon.entries[word])
            if readings is None:
                unrepairable.append(word)
                readings = lexicon.entries[word]
                was_repaired = False
            else:
                was_repaired = readings != lexicon.entries[word]
            repaired_entries += was_repaired
            for ipa in readings:
                phonemes.update(ipa)
            handle.write(
                json.dumps(
                    {
                        "word": word,
                        "ipa": readings.most_common(1)[0][0],
                        "variants": [{"ipa": ipa, "count": n} for ipa, n in readings.most_common()],
                        "repaired": was_repaired,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    partial.replace(args.out)

    stats = {
        "source": str(args.source),
        "sentences": report.sentences,
        "aligned_sentences": report.aligned,
        "normaliser_version": Normaliser().version,
        "alignment_rate": round(report.rate, 6),
        "aligned_words": report.words,
        "distinct_words": len(lexicon),
        "ambiguous_words": len(ambiguous),
        "ambiguity_rate": round(len(ambiguous) / len(lexicon), 6) if len(lexicon) else 0.0,
        "phoneme_inventory": len(phonemes),
        "skipped_by_delta": dict(sorted(report.skipped.items())),
        "repaired_entries": repaired_entries,
        "outside_writing_system": sorted(unrepairable),
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{report.sentences:,} sentences, {report.aligned:,} aligned ({report.rate:.1%})")
    print(f"  aligned words      {report.words:,}")
    print(f"  distinct words     {len(lexicon):,}")
    print(
        f"  ambiguous          {len(ambiguous):,} "
        f"({len(ambiguous) / len(lexicon):.1%} of distinct words)"
    )
    print(f"  phoneme inventory  {len(phonemes)}")
    print("  most-variant words:")
    for word, readings in sorted(ambiguous.items(), key=lambda kv: -len(kv[1]))[:TOP_AMBIGUOUS]:
        shown = ", ".join(f"{ipa} x{n}" for ipa, n in readings.most_common(3))
        print(f"    {word:16} {len(readings):>2} readings  {shown}")
    print(f"\n  lexicon -> {args.out}")
    print(f"  stats   -> {stats_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-g2p", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="word-align the pronunciation data")
    build.add_argument("--source", type=Path, default=DEFAULT_G2P)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return command_build(args)
    except G2PError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
