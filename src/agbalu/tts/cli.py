"""TTS front-end CLI.

python -m agbalu.tts.cli validate
python -m agbalu.tts.cli prompts
python -m agbalu.tts.cli cycle

Repair is not here: it is applied when the lexicon is built, by
`agbalu.g2p.cli.repair_readings`, so there is one pronunciation artifact and it is
correct rather than a defective one beside a corrected copy.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

from agbalu.speech import corpus
from agbalu.speech.corpus import read as read_clips
from agbalu.tts import cycle, scripture, training
from agbalu.tts import prompts as prompt_set
from agbalu.tts.g2p import PhonemeError, inventory, phonemize_word
from agbalu.tts.g2p import phonemize as g2p_phonemize
from agbalu.tts.kokoro import fold
from agbalu.tts.vocabulary import Vocabulary

DEFAULT_LEXICON: Final = Path("data/processed/lexicon/agbalu-pronunciations-v1.jsonl")
DEFAULT_STATS: Final = Path("data/processed/tts/g2p.stats.json")

DEFAULT_SPEECH: Final = Path("data/processed/speech")
DEFAULT_PROMPTS: Final = Path("data/processed/tts/prompts.jsonl")
DEFAULT_PROMPT_STATS: Final = Path("data/processed/tts/prompts.stats.json")

DEFAULT_RESULT: Final = Path("data/processed/bench/tts-baseline.json")

DEFAULT_TEXT: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
DEFAULT_OOD: Final = Path("data/processed/tts/ood_texts.txt")
DEFAULT_OOD_STATS: Final = Path("data/processed/tts/ood.stats.json")
OOD_SIZE: Final = 5000
OOD_MIN_CHARACTERS: Final = 60
OOD_OVERSAMPLE: Final = 4
OOD_SEED: Final = 0

SAMPLE: Final = 12

log: Final = logging.getLogger("agbalu.tts")


def read_lexicon(path: Path) -> Iterator[tuple[str, str]]:
    if not path.is_file():
        message = f"pronunciation lexicon not found: {path}"
        raise FileNotFoundError(message)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            yield record["word"], record["ipa"]


def _units(ipa: str) -> int:
    """IPA characters that carry a segment, ignoring length and pharyngealization."""
    return sum(1 for char in ipa if char not in "ˤː͡")


def command_validate(args: argparse.Namespace) -> int:
    """Reproduce the lexicon from the table, and report where the two disagree.

    A shortfall in the *source* is the interesting direction: it means the upstream
    generator emitted nothing for a character, which is how `o` and `ţ` were lost.

    This measures; it does not gate. The rates it writes are pinned by
    `tests/integration/test_tts_g2p_lexicon.py`, so a regression fails `make check`
    rather than turning this target permanently red over eight junk entries.
    """
    exact = 0
    total = 0
    source_short = 0
    disagree: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []
    for word, attested in read_lexicon(args.lexicon):
        total += 1
        try:
            derived = phonemize_word(word)
        except PhonemeError as error:
            failures.append((word, str(error)))
            continue
        if derived == attested:
            exact += 1
        else:
            if _units(attested) < _units(derived):
                source_short += 1
            if len(disagree) < SAMPLE:
                disagree.append((word, attested, derived))

    rate = exact / total if total else 0.0
    stats = {
        "lexicon": str(args.lexicon),
        "entries": total,
        "exact_match": exact,
        "exact_match_rate": round(rate, 6),
        "source_missing_symbols": source_short,
        "no_rule": len(failures),
        "inventory": "".join(sorted(inventory())),
        "inventory_size": len(inventory()),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log.info("entries=%d exact=%d (%.2f%%)", total, exact, 100 * rate)
    log.info("source missing a symbol the table emits: %d", source_short)
    for word, attested, derived in disagree:
        log.info("  %-18s source=%-22s table=%s", word, attested, derived)
    if failures:
        log.warning("outside the writing system, no rule: %d", len(failures))
        for _word, message in failures[:SAMPLE]:
            log.warning("  %s", message)
    log.info("wrote %s", args.stats)
    return 0


def command_prompts(args: argparse.Namespace) -> int:
    """Build the prompt set every Phase 12 system is scored on.

    The texts of train and dev are excluded, not just their speakers: a sentence
    Fadhma was trained on measures its memory rather than the synthesis under test.
    """
    verses = scripture.load(args.bible)
    seen = {
        clip.target
        for split in ("train", "dev")
        for clip in read_clips(args.speech / f"{split}.jsonl")
    }
    selection = prompt_set.select(
        read_clips(args.speech / "test.jsonl"),
        exclude=verses.match,
        seen=seen,
        size=args.size,
    )
    prompt_set.write(args.out, selection.prompts)

    stats = {
        "source": str(args.speech / "test.jsonl"),
        "prompts_path": str(args.out),
        "held_out_texts": len(seen),
        "scripture": {
            "source": str(args.bible),
            "verses": verses.verses,
            "ngram": scripture.NGRAM,
            "control": "passed",
        },
        **selection.as_dict(),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log.info(
        "prompts=%d of %d candidates, %d speakers, %.1f s of reference audio",
        len(selection.prompts),
        selection.candidates,
        selection.speakers,
        selection.seconds,
    )
    for reason, count in sorted(selection.rejected.items()):
        log.info("  rejected %-18s %d", reason, count)
    for prompt in selection.prompts[:SAMPLE]:
        log.info("  %s", prompt.text)
    log.info("wrote %s and %s", args.out, args.stats)
    return 0


def command_ood(args: argparse.Namespace) -> int:
    """Build the Kabyle out-of-distribution text the adversarial branch reads.

    The recipe ships German lines, which would steer the SLM loss toward German
    phonotactics from the epoch it switches on. Drawn from AƔBALU-Text v1 rather than from
    the clip transcripts for two reasons: only 2.0% of the female voice's clips reach the
    50-phoneme floor the sampler needs, and a transcript is not out of distribution.
    """
    vocabulary = Vocabulary.load()
    held_out = {
        clip.target
        for split in corpus.SPLITS
        for clip in read_clips(args.speech / f"{split}.jsonl")
        if (args.speech / f"{split}.jsonl").is_file()
    }
    held_out.update(prompt.target for prompt in prompt_set.read(args.prompts))
    if not held_out:
        message = (
            f"the exclusion set is empty, so nothing could be held out of the OOD text; "
            f"check {args.speech} and {args.prompts}"
        )
        raise FileNotFoundError(message)

    selection = training.select_ood(
        _sampled(args.text, minimum=args.min_characters, size=args.size, seed=args.seed),
        vocabulary,
        phonemize=lambda text: fold(g2p_phonemize(text)),
        exclude=held_out,
        size=args.size,
        minimum=training.MIN_OOD_PHONEMES,
    )
    written = training.write_ood(args.out, selection.lines)

    stats = {
        "source": str(args.text),
        "held_out_texts": len(held_out),
        **selection.as_dict(),
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log.info("ood lines=%d of %d considered", written, selection.considered)
    log.info("shortest=%d phonemes, floor=%d", selection.shortest, training.MIN_OOD_PHONEMES)
    for reason, count in sorted(selection.rejected.items()):
        log.info("  rejected %-20s %d", reason, count)
    log.info("wrote %s and %s", args.out, args.stats)
    return 0


def _sampled(path: Path, *, minimum: int, size: int, seed: int) -> list[str]:
    """A seeded reservoir over the whole corpus, never a prefix.

    AƔBALU-Text v1 is written source by source, so its first records are one source: taking
    the head of the file gave 5,179 rows all from `hf.abdelhaqueidali.kab-latn-tfng`, which
    would have made the adversarial branch read one register of Kabyle. The reservoir is
    drawn over every record clearing the character floor, and it is oversampled because
    phonemisation rejects some of what it returns.

    The character floor runs ahead of the phoneme floor only to keep the cost down:
    phonemising 3,041,989 records to keep a few thousand is minutes of work for a file the
    sampler reads a few thousand lines of.
    """
    if not path.is_file():
        message = f"text corpus not found: {path}"
        raise FileNotFoundError(message)
    wanted = OOD_OVERSAMPLE * size
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if len(text) < minimum:
                continue
            seen += 1
            if len(reservoir) < wanted:
                reservoir.append(text)
                continue
            position = rng.randrange(seen)
            if position < wanted:
                reservoir[position] = text
    log.info("reservoir: %d of %d eligible records", len(reservoir), seen)
    rng.shuffle(reservoir)
    return reservoir


def _rate(value: float | None) -> str:
    return f"{value:7.4f}" if value is not None else "    n/a"


def command_cycle(args: argparse.Namespace) -> int:
    """Read a Cycle-CER result back and report what it is allowed to claim.

    Non-zero when the control fails. A floor far from Fadhma's published rate means the
    decoder is not the published one, and every delta measured beside it is void — which
    is a gate rather than a line of output, because the deltas still look plausible.
    """
    if not args.result.is_file():
        log.error("no Cycle-CER result at %s", args.result)
        return 1
    report = cycle.read_result(json.loads(args.result.read_text(encoding="utf-8")))
    control = report.control

    log.info("%s", args.result)
    for scored in report.conditions:
        log.info(
            "  %-24s cer=%s wer=%s  n=%d",
            scored.name,
            _rate(scored.cer_percent),
            _rate(scored.wer_percent),
            scored.utterances,
        )
    for name, value in report.deltas.items():
        log.info("  cycle-cer delta over the floor  %-24s %+.4f", name, value)
    log.info(
        "  control: floor %.4f against the published %.4f, gap %+.4f, tolerance %.2f",
        control.measured,
        control.published,
        control.gap,
        control.tolerance,
    )
    if not control.holds:
        log.error("  control FAILED — the decoder is not the published one, deltas are void")
        return 1
    log.info("  control passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="agbalu.tts")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="reproduce the lexicon from the rule table")
    validate.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    validate.set_defaults(handler=command_validate)

    prompts = sub.add_parser("prompts", help="build the held-out, non-biblical prompt set")
    prompts.add_argument("--speech", type=Path, default=DEFAULT_SPEECH)
    prompts.add_argument("--bible", type=Path, default=scripture.BIBLE)
    prompts.add_argument("--size", type=int, default=prompt_set.DEFAULT_SIZE)
    prompts.add_argument("--out", type=Path, default=DEFAULT_PROMPTS)
    prompts.add_argument("--stats", type=Path, default=DEFAULT_PROMPT_STATS)
    prompts.set_defaults(handler=command_prompts)

    ood = sub.add_parser("ood", help="build the Kabyle out-of-distribution text for the SLM branch")
    ood.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    ood.add_argument("--speech", type=Path, default=DEFAULT_SPEECH)
    ood.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ood.add_argument("--size", type=int, default=OOD_SIZE)
    ood.add_argument("--min-characters", type=int, default=OOD_MIN_CHARACTERS)
    ood.add_argument("--seed", type=int, default=OOD_SEED)
    ood.add_argument("--out", type=Path, default=DEFAULT_OOD)
    ood.add_argument("--stats", type=Path, default=DEFAULT_OOD_STATS)
    ood.set_defaults(handler=command_ood)

    cycle_check = sub.add_parser("cycle", help="read a Cycle-CER result and check its control")
    cycle_check.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    cycle_check.set_defaults(handler=command_cycle)

    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
