"""Training and evaluation splits, decontaminated against the audio the model is scored on.

58.2% of the Common Voice transcripts are also in AƔBALU-Text v1, because most of them are
Tatoeba sentences and Tatoeba is in the corpus. A model trained on the corpus has therefore
already read the punctuated form of most of the audio test set, and so has Masinissa, which
was pretrained on the same file. Excluding those clips is what makes the evaluation mean
anything, and it decontaminates the encoder for free.

A transcript with no final mark is dropped rather than labelled `NONE`. Reading them settles
what they are: no punctuation *and* no capitals, a contributor who typed neither. That is
transcriber habit, and training on it teaches typing style rather than Kabyle.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from agbalu.punctuation.labels import CASE, PUNCTUATION, annotate, collation_key

log: Final = logging.getLogger("agbalu.punctuation")

TEXT_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
SPEECH_DIR: Final = Path("data/processed/speech")
OUTPUT_DIR: Final = Path("data/processed/punctuation")

#: The Common Voice split names `read_speech` reads. Not what `build` writes — `ood` comes
#: from the text corpus and has no audio side.
SPLITS: Final[tuple[str, ...]] = ("train", "dev", "test")

#: Every split `build` writes, and the list anything staging them must iterate. A second copy
#: of this list left `ood` off the Modal upload and the evaluation failed on the volume.
WRITTEN_SPLITS: Final[tuple[str, ...]] = ("train", "dev", "test", "ood")

FINAL_MARKS: Final = ".?!"

#: Sources whose record boundaries were chosen by a bitext miner rather than by a writer.
#: `opus.nllb-kab` records end mid-sentence often enough that a survivor still asserts the
#: miner cut in the right place, and sentence-final punctuation is exactly what is predicted.
UNTRUSTED_SOURCES: Final[frozenset[str]] = frozenset({"opus.nllb-kab"})

#: Held out of training entirely and written as the `ood` split. Half the training text is
#: Tatoeba-derived and so is Common Voice, so `dev` and `test` measure the model on the shape
#: it was trained on. This source is real long-form prose at 19.6 words a record, and it is
#: the only evidence available for whether the model works on a Kabyle book. It is 3.5% of
#: training rows, small enough that holding it out does not confound the comparison.
OOD_SOURCE: Final = "hf.imsidag.kabyle-corpus-hca"

#: Format specifiers, markup and editorial brackets: the record is a localisation string or
#: a table row, not prose.
NON_PROSE: Final = re.compile(r"%[0-9$sdfl]|\{\d|</?[a-z]+>|\[[^\]]*\]|https?://|_{2,}")

MIN_CHARS: Final = 12
MIN_WORDS: Final = 3
MAX_WORDS: Final = 60
MAX_DIGIT_SHARE: Final = 0.15


class CorpusError(Exception):
    pass


class SplitStats(TypedDict):
    rows: int
    words: int
    punctuation: dict[str, int]
    case: dict[str, int]


class Excluded(TypedDict):
    """Why clips did not survive into an evaluation split. Each of these can fire."""

    clips: int
    no_final_mark: int
    in_text_corpus: int
    in_speech_train: int
    kept: int


class SourceYield(TypedDict):
    source: str
    records: int
    kept: int


class BuildStats(TypedDict):
    text_records: int
    wellformed: int
    untrusted_dropped: int
    speech_added: int
    excluded: dict[str, Excluded]
    splits: dict[str, SplitStats]
    by_source: list[SourceYield]


@dataclass(frozen=True, slots=True)
class Row:
    text: str
    source: str


def is_wellformed(text: str) -> bool:
    """Whether a record is a written sentence rather than a fragment or a table cell."""
    words = text.split()
    if len(text) < MIN_CHARS or not MIN_WORDS <= len(words) <= MAX_WORDS:
        return False
    if not text[0].isupper() or text[-1] not in FINAL_MARKS:
        return False
    if NON_PROSE.search(text):
        return False
    return sum(char.isdigit() for char in text) <= len(text) * MAX_DIGIT_SHARE


def read_speech(speech_dir: Path, split: str) -> list[str]:
    path = speech_dir / f"{split}.jsonl"
    if not path.is_file():
        msg = f"no speech split at {path} — run `make speech TASK=corpus` first"
        raise CorpusError(msg)
    with path.open(encoding="utf-8") as handle:
        return [str(json.loads(line)["text"]).strip() for line in handle]


def _stats(rows: list[Row]) -> SplitStats:
    punctuation: Counter[str] = Counter()
    case: Counter[str] = Counter()
    words = 0
    for row in rows:
        annotation = annotate(row.text)
        words += len(annotation.words)
        punctuation.update(PUNCTUATION[label] for label in annotation.punctuation)
        case.update(CASE[label] for label in annotation.case)
    return {
        "rows": len(rows),
        "words": words,
        "punctuation": dict(punctuation.most_common()),
        "case": dict(case.most_common()),
    }


@dataclass(frozen=True, slots=True)
class TextPass:
    records: int
    wellformed: int
    untrusted: int
    keys: set[str]
    rows: list[Row]
    per_source: Counter[str]
    kept_per_source: Counter[str]


def scan_text_corpus(path: Path, limit: int | None = None) -> TextPass:
    """One pass: every collation key, for contamination, and the sentences worth training on."""
    if not path.is_file():
        msg = f"no text corpus at {path} — run `make extract` first"
        raise CorpusError(msg)

    keys: set[str] = set()
    rows: list[Row] = []
    per_source: Counter[str] = Counter()
    kept_per_source: Counter[str] = Counter()
    records = wellformed = untrusted = 0

    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            payload = json.loads(line)
            text = (payload["text"] or "").strip()
            source = str(payload["source"])
            records += 1
            per_source[source] += 1
            if not text:
                continue
            keys.add(collation_key(text))
            if not is_wellformed(text):
                continue
            wellformed += 1
            kept_per_source[source] += 1
            if source in UNTRUSTED_SOURCES:
                untrusted += 1
                continue
            rows.append(Row(text, source))

    return TextPass(records, wellformed, untrusted, keys, rows, per_source, kept_per_source)


def _write_split(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row.text, "source": row.source}, ensure_ascii=False))
            handle.write("\n")


def read_split(path: Path) -> list[Row]:
    if not path.is_file():
        msg = f"no split at {path} — run `make punctuation TASK=corpus` first"
        raise CorpusError(msg)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [Row(str(row["text"]), str(row["source"])) for row in rows]


def build(
    text_corpus: Path = TEXT_CORPUS,
    speech_dir: Path = SPEECH_DIR,
    output_dir: Path = OUTPUT_DIR,
    limit: int | None = None,
) -> BuildStats:
    """Write `WRITTEN_SPLITS` and the statistics the label distribution is read from."""
    text = scan_text_corpus(text_corpus, limit)
    speech = {split: read_speech(speech_dir, split) for split in SPLITS}
    speech_train_keys = {collation_key(line) for line in speech["train"] if line}

    def scorable(split: str) -> tuple[list[Row], Excluded]:
        rows: list[Row] = []
        excluded: Excluded = {
            "clips": len(speech[split]),
            "no_final_mark": 0,
            "in_text_corpus": 0,
            "in_speech_train": 0,
            "kept": 0,
        }
        for line in speech[split]:
            if not line or line[-1] not in FINAL_MARKS:
                excluded["no_final_mark"] += 1
                continue
            key = collation_key(line)
            if key in text.keys:
                excluded["in_text_corpus"] += 1
                continue
            if key in speech_train_keys:
                excluded["in_speech_train"] += 1
                continue
            rows.append(Row(line, f"speech.{split}"))
        excluded["kept"] = len(rows)
        return rows, excluded

    scored = {split: scorable(split) for split in ("dev", "test")}
    evaluation: dict[str, list[Row]] = {split: rows for split, (rows, _) in scored.items()}
    excluded = {split: counts for split, (_, counts) in scored.items()}
    evaluation["ood"] = [row for row in text.rows if row.source == OOD_SOURCE]
    held_out = {collation_key(row.text) for rows in evaluation.values() for row in rows}

    train = [
        row
        for row in text.rows
        if row.source != OOD_SOURCE and collation_key(row.text) not in held_out
    ]

    speech_added = 0
    for line in speech["train"]:
        if not line or line[-1] not in FINAL_MARKS:
            continue
        if collation_key(line) in held_out:
            continue
        train.append(Row(line, "speech.train"))
        speech_added += 1

    splits = {"train": train, **evaluation}
    for name in WRITTEN_SPLITS:
        rows = splits[name]
        _write_split(output_dir / f"{name}.jsonl", rows)
        log.info("wrote %s: %d rows", name, len(rows))

    stats: BuildStats = {
        "text_records": text.records,
        "wellformed": text.wellformed,
        "untrusted_dropped": text.untrusted,
        "speech_added": speech_added,
        "excluded": excluded,
        "splits": {name: _stats(rows) for name, rows in splits.items()},
        "by_source": [
            {"source": source, "records": text.per_source[source], "kept": kept}
            for source, kept in text.kept_per_source.most_common()
        ],
    }
    stats_path = output_dir / "punctuation.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats
