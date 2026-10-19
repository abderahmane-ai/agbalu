"""Phase 5 build steps. All need a CPU and the transcripts, and no GPU.

python3 -m agbalu.speech.cli corpus
python3 -m agbalu.speech.cli vocabulary
python3 -m agbalu.speech.cli lm
python3 -m agbalu.speech.cli release
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Final

from agbalu.speech import corpus, vocabulary
from agbalu.speech.corpus import SPLITS, SpeechError
from agbalu.speech.lm import (
    LM_ORDER,
    LOCAL_LM_ARPA,
    LOCAL_LM_BINARY,
    LanguageModelError,
    build_arpa,
    compile_binary,
)
from agbalu.speech.release import ReleaseError, write_config
from agbalu.speech.transcribe import (
    DEFAULT_DEVICE,
    RELEASE_DIR,
    TranscriptionError,
    transcribe,
)

TRANSCRIPTS: Final = Path("data/raw/hf.fsicoli.common-voice-22-kab/transcript/kab")
PROCESSED: Final = Path("data/processed/speech")
CORPUS_STATS: Final = PROCESSED / "speech.stats.json"
VOCAB: Final = Path("artifacts/asr/vocab.json")
VOCAB_STATS: Final = PROCESSED / "vocabulary.stats.json"

TEXT_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
"""Normalised Kabyle text corpus produced by Phase 1–2 text pipeline."""

LM_PLAIN_TEXT: Final = PROCESSED / "lm_sentences.txt"
"""One normalised sentence per line; intermediate input for lmplz."""


def command_corpus(args: argparse.Namespace) -> int:
    """Normalise, filter and split the transcripts, verifying speaker disjointness."""
    report = corpus.build(args.transcripts, args.out, splits=args.splits)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dropped: Counter[str] = Counter()
    for split in report.splits:
        dropped.update(split.rejected)
        print(
            f"  {split.split:6s} {split.kept:7,d} clips  {split.hours:8.2f} h  "
            f"{split.speakers:5,d} speakers  {split.repaired:6,d} repaired  "
            f"{sum(split.rejected.values()):5,d} dropped"
        )
    for reason, count in sorted(dropped.items()):
        print(f"    dropped {reason}: {count:,}")
    print(f"  {report.hours:.2f} h kept, speaker overlap {report.overlaps}")
    print(f"-> {args.stats}")
    return 0


def command_vocabulary(args: argparse.Namespace) -> int:
    """Take the CTC inventory from the built splits, and audit what it excluded."""
    clips = [clip for split in args.splits for clip in corpus.read(args.out / f"{split}.jsonl")]
    built = vocabulary.build((c.target for c in clips), sources=(c.text for c in clips))
    built.write(args.vocab)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps(built.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  {len(built.as_mapping())} CTC classes over {len(clips):,} clips")
    print(f"  characters {''.join(c for c in built.characters if c != ' ')}")
    if built.unexpected:
        print(f"  unspeakable and dropped: {dict(sorted(built.unexpected.items()))}")
    print(f"-> {args.vocab}")
    print(f"-> {args.stats}")
    return 0


def command_lm(args: argparse.Namespace) -> int:
    """Build the n-gram the CTC decoder is shallow-fused with.

    Needs `lmplz` and `build_binary` on PATH. Without them, `make modal-asr TASK=lm`
    builds the same model in a container that has the toolchain.
    """
    if not args.text.exists():
        print(f"error: {args.text} not found; run `make extract` first", file=sys.stderr)
        return 1

    args.plain.parent.mkdir(parents=True, exist_ok=True)
    records = lines = 0
    with args.plain.open("w", encoding="utf-8") as sink, args.text.open(encoding="utf-8") as source:
        for line in source:
            text = json.loads(line).get("text", "").strip()
            if text:
                sink.write(text + "\n")
                records += 1
                lines += text.count("\n") + 1
    # `lmplz` reads a sentence per *line*, and 204,185 records of AƔBALU-Text v1 carry an
    # internal newline, so the record count is 6.7% short of what the model is built from.
    # Reported separately rather than reconciled: the published `5gram.klm` was built from
    # these lines, and flattening them now would make the binary unreproducible.
    print(f"  {lines:,} lines from {records:,} records -> {args.plain}")

    build_arpa(args.plain, args.arpa, order=args.order)
    compile_binary(args.arpa, args.binary)
    print(f"-> {args.binary}")

    if not args.keep_arpa:
        args.arpa.unlink()
    return 0


def command_release(args: argparse.Namespace) -> int:
    """Write the architecture a downloader needs beside the published weights.

    The training checkpoint carries the optimizer, the scheduler and the RNG and no config
    at all, so without this the weights publish with nothing that can construct them.
    """
    if not args.vocab.is_file():
        print(f"error: {args.vocab} not found; run `make speech TASK=vocabulary`", file=sys.stderr)
        return 1
    vocabulary = json.loads(args.vocab.read_text(encoding="utf-8"))
    for path in write_config(vocabulary, args.out):
        print(f"-> {path}")
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    """Print one hypothesis per line, so the output pipes into punctuation restoration."""
    for result in transcribe(
        args.audio,
        args.release,
        device=args.device,
        fuse=not args.greedy,
        limit=args.limit,
    ):
        print(result.text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-speech", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    corpus_cmd = sub.add_parser("corpus", help="normalise and split the Common Voice transcripts")
    corpus_cmd.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    corpus_cmd.add_argument("--out", type=Path, default=PROCESSED)
    corpus_cmd.add_argument("--stats", type=Path, default=CORPUS_STATS)
    corpus_cmd.add_argument("--splits", nargs="+", default=list(SPLITS))

    vocab_cmd = sub.add_parser("vocabulary", help="build the CTC vocabulary from the splits")
    vocab_cmd.add_argument("--out", type=Path, default=PROCESSED)
    vocab_cmd.add_argument("--vocab", type=Path, default=VOCAB)
    vocab_cmd.add_argument("--stats", type=Path, default=VOCAB_STATS)
    vocab_cmd.add_argument("--splits", nargs="+", default=list(SPLITS))

    lm_cmd = sub.add_parser("lm", help="build the KenLM 5-gram from AƔBALU-Text v1")
    lm_cmd.add_argument("--text", type=Path, default=TEXT_CORPUS)
    lm_cmd.add_argument("--plain", type=Path, default=LM_PLAIN_TEXT)
    lm_cmd.add_argument("--arpa", type=Path, default=LOCAL_LM_ARPA)
    lm_cmd.add_argument("--binary", type=Path, default=LOCAL_LM_BINARY)
    lm_cmd.add_argument("--order", type=int, default=LM_ORDER)
    lm_cmd.add_argument("--keep-arpa", action="store_true", help="keep the intermediate .arpa")

    release = sub.add_parser("release", help="write config.json and preprocessor_config.json")
    release.add_argument("--vocab", type=Path, default=VOCAB)
    release.add_argument("--out", type=Path, default=VOCAB.parent)

    transcribe = sub.add_parser("transcribe", help="decode audio with the released Fadhma")
    transcribe.add_argument("audio", type=Path, help="a clip, or a directory of clips")
    transcribe.add_argument("--release", type=Path, default=RELEASE_DIR)
    transcribe.add_argument("--device", default=DEFAULT_DEVICE)
    transcribe.add_argument("--limit", type=int, default=None)
    transcribe.add_argument(
        "--greedy", action="store_true", help="skip 5-gram fusion (CER 8.53 against 8.01)"
    )

    return parser


COMMANDS: Final[dict[str, Callable[[argparse.Namespace], int]]] = {
    "corpus": command_corpus,
    "vocabulary": command_vocabulary,
    "lm": command_lm,
    "release": command_release,
    "transcribe": command_transcribe,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (SpeechError, LanguageModelError, ReleaseError, TranscriptionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
