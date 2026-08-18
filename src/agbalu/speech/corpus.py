"""Common Voice Kabyle transcripts into a speech corpus Fadhma can train on.

Three things happen here that a stock Common Voice recipe does not do. Transcripts
are normalised before anything reads their characters, so the homoglyph corruption
never reaches the CTC vocabulary. Speaker disjointness across splits is *verified*
rather than assumed, because a leaked speaker inflates the test score in the
direction of looking good. And every clip is checked for CTC feasibility: the loss
is infinite when the target is longer than the frame sequence, which surfaces as a
NaN thousands of steps into a run rather than as a rejected row here.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from agbalu.normalise import Normaliser
from agbalu.speech.vocabulary import ctc_target

SPLITS: Final = ("train", "dev", "test")
"""Common Voice's own buckets. `other` and `invalidated` are unvalidated audio and
are built only when asked for by name."""

DURATIONS: Final = "clip_durations.tsv"

SAMPLE_RATE: Final = 16_000

CONV_KERNEL: Final = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDE: Final = (5, 2, 2, 2, 2, 2, 2)
"""The wav2vec 2.0 convolutional front end, read from the base checkpoint's own
`config.json`. Total stride is 320 samples — 20 ms at 16 kHz — but the *frame count* is
not the duration over 20 ms: each layer loses `kernel - 1` samples at the edge, and the
chain costs exactly one frame. `_get_feat_extract_output_lengths` is the formula this
mirrors, and mirroring it is what makes the CTC feasibility check exact rather than one
frame optimistic."""

MIN_DURATION_MS: Final = 400
MAX_DURATION_MS: Final = 20_000
"""Below the floor there is no room for a word; above the ceiling one clip's
activations dominate a batch. Common Voice Kabyle has a 3.46 s mean and no clip
over 30 s, so the ceiling costs almost nothing and bounds the memory."""

REQUIRED_COLUMNS: Final = frozenset({"client_id", "path", "sentence"})

csv.field_size_limit(10**9)


EXCESS: Final = "__excess__"
"""`restkey` for columns beyond the header. Named rather than left as `None`, so a
row carrying one is distinguishable from a row missing one."""


def _reader(handle: TextIO) -> csv.DictReader[str]:
    """Common Voice writes unquoted TSV and 663 transcripts contain a double quote.

    `QUOTE_NONE` is mandatory: the default swallows every line after such a quote,
    which cost 5,904 sentences from the Tatoeba seed before it was caught.
    """
    return csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE, restkey=EXCESS)


class SpeechError(Exception):
    """A malformed corpus, or one whose splits cannot be trusted."""


@dataclass(frozen=True, slots=True)
class Clip:
    """One utterance, ready for the collator."""

    clip: str
    speaker: str
    split: str
    duration_ms: int
    text: str
    target: str
    repaired: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "clip": self.clip,
            "speaker": self.speaker,
            "split": self.split,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "target": self.target,
            "repaired": self.repaired,
        }


@dataclass(frozen=True, slots=True)
class SplitReport:
    """What one split kept, and why it dropped the rest."""

    split: str
    kept: int
    rejected: Mapping[str, int]
    duration_ms: int
    speakers: int
    repaired: int

    @property
    def hours(self) -> float:
        return self.duration_ms / 3_600_000

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "clips": self.kept,
            "hours": round(self.hours, 4),
            "speakers": self.speakers,
            "repaired": self.repaired,
            "rejected": dict(sorted(self.rejected.items())),
        }


@dataclass(frozen=True, slots=True)
class CorpusReport:
    """Every split, plus the speaker-overlap check that licenses the test score."""

    splits: tuple[SplitReport, ...]
    overlaps: Mapping[str, int]
    paths: tuple[Path, ...]

    @property
    def hours(self) -> float:
        return sum(s.hours for s in self.splits)

    def as_dict(self) -> dict[str, object]:
        return {
            "normaliser_version": Normaliser().version,
            "conv_kernel": list(CONV_KERNEL),
            "conv_stride": list(CONV_STRIDE),
            "min_duration_ms": MIN_DURATION_MS,
            "max_duration_ms": MAX_DURATION_MS,
            "hours": round(self.hours, 4),
            "clips": sum(s.kept for s in self.splits),
            "splits": [s.as_dict() for s in self.splits],
            "speaker_overlap": dict(sorted(self.overlaps.items())),
            "files": [str(p) for p in self.paths],
        }


def read_durations(path: Path) -> dict[str, int]:
    """Clip name to duration in milliseconds."""
    if not path.is_file():
        message = f"clip durations not found: {path}"
        raise SpeechError(message)
    durations: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = _reader(handle)
        if reader.fieldnames is None or "clip" not in reader.fieldnames:
            message = f"{path} has no `clip` column"
            raise SpeechError(message)
        column = next((f for f in reader.fieldnames if f.startswith("duration")), None)
        if column is None:
            message = f"{path} has no duration column"
            raise SpeechError(message)
        for number, row in enumerate(reader, start=2):
            clip, value = row.get("clip"), row.get(column)
            if not clip or not value:
                continue
            try:
                durations[clip] = int(value)
            except ValueError as error:
                message = f"{path}:{number} duration is not an integer: {value!r}"
                raise SpeechError(message) from error
    if not durations:
        message = f"{path} carries no durations"
        raise SpeechError(message)
    return durations


def rows(path: Path) -> Iterator[dict[str, str]]:
    """One Common Voice TSV, read under the quoting policy the whole project depends on."""
    if not path.is_file():
        message = f"split not found: {path}"
        raise SpeechError(message)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = _reader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            message = f"{path} is missing {', '.join(sorted(missing))}"
            raise SpeechError(message)
        for row in reader:
            # A short row leaves later columns None; a long one is a transcript
            # holding an unescaped tab, so its `sentence` is a truncated prefix.
            # Either way the row is not the schema the header declares.
            if EXCESS in row or any(value is None for value in row.values()):
                continue
            yield row


def frames(duration_ms: int) -> int:
    """Encoder frames a clip of this duration yields, by the convolutional chain itself.

    CTC loss is infinite when the target is longer than this, so an over-estimate admits
    rows that produce a NaN thousands of steps into a run.
    """
    length = duration_ms * SAMPLE_RATE // 1000
    for kernel, stride in zip(CONV_KERNEL, CONV_STRIDE, strict=True):
        length = (length - kernel) // stride + 1
    return max(length, 0)


def _clips(path: Path, split: str, durations: Mapping[str, int]) -> Iterator[Clip | str]:
    """Yield a `Clip`, or the reason the row was rejected."""
    normaliser = Normaliser()
    for row in rows(path):
        clip = row["path"].strip()
        raw = row["sentence"]
        duration = durations.get(clip)
        if not clip:
            yield "no-clip"
            continue
        if duration is None:
            yield "no-duration"
            continue
        if duration < MIN_DURATION_MS:
            yield "too-short"
            continue
        if duration > MAX_DURATION_MS:
            yield "too-long"
            continue
        text = normaliser.normalise(raw)
        target = ctc_target(text)
        if not target:
            yield "empty-transcript"
            continue
        if len(target) > frames(duration):
            yield "ctc-infeasible"
            continue
        yield Clip(
            clip=clip,
            speaker=row["client_id"].strip(),
            split=split,
            duration_ms=duration,
            text=text,
            target=target,
            repaired=text != raw,
        )


def _overlaps(speakers: Mapping[str, set[str]]) -> dict[str, int]:
    names = sorted(speakers)
    return {
        f"{left}-{right}": len(speakers[left] & speakers[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def build(root: Path, out: Path, *, splits: Sequence[str] = SPLITS) -> CorpusReport:
    """Write one JSONL per split, and refuse a corpus whose splits share a speaker."""
    if not splits:
        message = "no splits requested"
        raise SpeechError(message)

    durations = read_durations(root / DURATIONS)
    out.mkdir(parents=True, exist_ok=True)

    reports: list[SplitReport] = []
    paths: list[Path] = []
    speakers: dict[str, set[str]] = {}

    for split in splits:
        rejected: dict[str, int] = {}
        kept = repaired = total_ms = 0
        seen: set[str] = set()
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as sink:
            for item in _clips(root / f"{split}.tsv", split, durations):
                if isinstance(item, str):
                    rejected[item] = rejected.get(item, 0) + 1
                    continue
                sink.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")
                kept += 1
                repaired += int(item.repaired)
                total_ms += item.duration_ms
                seen.add(item.speaker)
        if not kept:
            message = f"split {split!r} kept no clips"
            raise SpeechError(message)
        speakers[split] = seen
        paths.append(path)
        reports.append(
            SplitReport(
                split=split,
                kept=kept,
                rejected=rejected,
                duration_ms=total_ms,
                speakers=len(seen),
                repaired=repaired,
            )
        )

    overlaps = _overlaps(speakers)
    leaking = {pair: count for pair, count in overlaps.items() if count}
    if leaking:
        detail = ", ".join(f"{pair}: {count}" for pair, count in sorted(leaking.items()))
        message = f"splits share speakers, so no held-out score is defensible ({detail})"
        raise SpeechError(message)

    return CorpusReport(splits=tuple(reports), overlaps=overlaps, paths=tuple(paths))


def read(path: Path) -> list[Clip]:
    """The clips of one built split, in file order."""
    if not path.is_file():
        message = f"split not built: {path}"
        raise SpeechError(message)
    out: list[Clip] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                out.append(
                    Clip(
                        clip=row["clip"],
                        speaker=row["speaker"],
                        split=row["split"],
                        duration_ms=int(row["duration_ms"]),
                        text=row["text"],
                        target=row["target"],
                        repaired=bool(row["repaired"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                message = f"{path}:{number} is not a clip record"
                raise SpeechError(message) from error
    if not out:
        message = f"split is empty: {path}"
        raise SpeechError(message)
    return out
