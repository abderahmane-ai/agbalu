"""Turning 12.4's per-voice filelists into the two lists the base recipe reads.

Stage 1 is multi-speaker and Stage 2 is one voice, so the same rows are assembled twice:
merged across voices for the base, then per voice for the fine-tune. The `cycle` split
enters neither. It is 12.5's measurement set, and training on it would turn Cycle-CER into
a memorisation score — the same reason 12.1's prompt set excludes anything Fadhma saw.

Every row is re-encoded through `agbalu.tts.vocabulary` on the way past, and the two
failures are not the same kind of thing. A phoneme with no embedding row is **corruption**:
12.4 folded the tie bar when it wrote these lists and the table covers everything the G2P
can emit, so an unmapped symbol means the list is not what it claims and the run stops. A
sequence past PL-BERT's window is **data**: a long clip legitimately produces one, and it is
dropped with a count. The recipe's own cleaner makes neither distinction — it drops both,
silently, into the training target.

The merged list is shuffled under a fixed seed rather than left in voice order. Whether the
recipe's dataloader shuffles is its business; concatenating two speakers and trusting
something downstream to interleave them is how a loss curve ends up tracking which source
the batch came from.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from agbalu.tts.g2p import PhonemeError
from agbalu.tts.vocabulary import PLBERT_LIMIT, VocabularyError

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable, Mapping, Sequence
    from pathlib import Path

    from agbalu.tts.vocabulary import Vocabulary

DEFAULT_RUN_NAME: Final = "matoub-v1"
"""What a Matoub launch is labelled when the operator names nothing.

`--run-name` is shared across every function the launcher spawns and defaults to the
encoder's, so a run that forwards it raw reports itself as `agbalu-encoder-v1`."""

TRAIN_LIST: Final = "train_list.txt"
VAL_LIST: Final = "val_list.txt"
"""The filenames the recipe's config points at. One definition, because the config and the
writer must agree and they are edited at different times."""

OOD_LIST: Final = "OOD_texts.txt"
MIN_OOD_LINES: Final = 2
"""Fewest lines the recipe's sampler can address. It draws `randint(0, len - 1)`, which
raises at one line and never selects the last element at any size."""

MIN_OOD_PHONEMES: Final = 50
"""Shortest OOD line the recipe will accept.

Its sampler is `while len(ps) < min_length: ps = <random line>`, so a file whose lines are
all shorter never terminates — an infinite loop inside a paid GPU container. Every line
written is at or above this, which makes the loop exit on its first draw. The measured
reason it matters here: only 2.0% of the female voice's clips and 7.0% of the male's reach
50 phoneme characters, so the transcripts cannot supply this file.
"""

FIELDS: Final = 3
SEED: Final = 0

UNMAPPED: Final = "unmapped-phoneme"
OVERLONG: Final = "over-plbert-window"


class TrainingError(Exception):
    """A filelist the recipe cannot be handed."""


def voice_list(base: str, voice: str) -> str:
    """The per-voice counterpart of a merged list filename.

    One definition, for the same reason `TRAIN_LIST` is one: a Stage 2 left pointing at the
    merged list fine-tunes on both voices and raises nothing, and the checkpoint it writes
    is indistinguishable from the single-speaker one it was launched to produce.
    """
    stem, dot, suffix = base.rpartition(".")
    if not dot:
        message = f"no suffix in {base!r} to insert a voice before"
        raise TrainingError(message)
    return f"{stem}{dot}{voice}{dot}{suffix}"


@dataclass(frozen=True, slots=True)
class Row:
    """One line of a recipe filelist: where the audio is, what it says, who said it."""

    audio: str
    ipa: str
    speaker: str

    def rendered(self) -> str:
        return f"{self.audio}|{self.ipa}|{self.speaker}"


@dataclass(frozen=True, slots=True)
class Selection:
    """Rows the recipe may read, and everything that did not survive the way here."""

    rows: tuple[Row, ...]
    rejected: Mapping[str, int]
    unmapped: Mapping[str, int]
    longest: int

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(sorted({row.speaker for row in self.rows}))

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": len(self.rows),
            "speakers": list(self.speakers),
            "longest_token_sequence": self.longest,
            "plbert_limit": PLBERT_LIMIT,
            "rejected": dict(sorted(self.rejected.items())),
            "unmapped_symbols": {
                f"{symbol} U+{ord(symbol):04X}": count
                for symbol, count in sorted(self.unmapped.items())
            },
        }


def parse(line: str) -> Row:
    """One filelist line, strictly. A missing field is a corrupt list, not a default."""
    parts = line.rstrip("\n").split("|")
    if len(parts) != FIELDS:
        message = f"expected {FIELDS} fields separated by '|', got {len(parts)}: {line!r}"
        raise TrainingError(message)
    audio, ipa, speaker = (part.strip() for part in parts)
    if not audio or not speaker:
        message = f"a row must name its audio and its speaker: {line!r}"
        raise TrainingError(message)
    return Row(audio=audio, ipa=ipa, speaker=speaker)


def read_list(path: Path) -> tuple[Row, ...]:
    if not path.is_file():
        message = f"filelist not found: {path}"
        raise TrainingError(message)
    rows: list[Row] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(parse(line))
        except TrainingError as error:
            message = f"{path}:{number}: {error}"
            raise TrainingError(message) from error
    return tuple(rows)


def select(rows: Iterable[Row], vocabulary: Vocabulary, *, limit: int = PLBERT_LIMIT) -> Selection:
    """Keep the rows the base model can be trained on, and count what the rest were."""
    kept: list[Row] = []
    rejected: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    longest = 0
    for row in rows:
        missing = vocabulary.unmapped(row.ipa)
        if missing:
            rejected[UNMAPPED] += 1
            unmapped.update(missing)
            continue
        length = len(row.ipa)
        longest = max(longest, length)
        if length > limit:
            rejected[OVERLONG] += 1
            continue
        kept.append(row)
    return Selection(
        rows=tuple(kept),
        rejected=dict(rejected),
        unmapped=dict(unmapped),
        longest=longest,
    )


def require_encodable(selection: Selection) -> None:
    """Refuse a selection that dropped a phoneme, before anything reaches a GPU.

    Separate from `select` so one pass diagnoses every bad symbol in the corpus rather
    than the first: a container that stops at row one costs the same as a container that
    stops at the end, and only the second says what is wrong.
    """
    if not selection.unmapped:
        return
    named = ", ".join(
        f"{symbol!r} U+{ord(symbol):04X} in {count} rows"
        for symbol, count in sorted(selection.unmapped.items())
    )
    message = (
        f"{selection.rejected.get(UNMAPPED, 0)} rows carry a phoneme with no embedding "
        f"row: {named}. 12.4 folds the tie bar and the table covers the G2P, so these "
        f"lists are not what they claim"
    )
    raise VocabularyError(message)


def merge(selections: Sequence[Selection], *, seed: int = SEED) -> tuple[Row, ...]:
    """Every voice's rows in one list, interleaved deterministically."""
    rows = [row for selection in selections for row in selection.rows]
    random.Random(seed).shuffle(rows)
    return tuple(rows)


def assign_speaker_ids(names: Iterable[str]) -> dict[str, int]:
    """Voice name to the integer the recipe requires, assigned in sorted order.

    Not cosmetic. `meldataset._load_tensor` runs `speaker_id = int(speaker_id)`, and its
    reference lookup compares the column against `str(speaker_id)`, so a name in that
    column raises `ValueError` on the first batch — after the container and the GPU have
    been paid for. 12.4 writes the voice name there because that is what identifies an arm
    on the volume; the translation happens here, once, on the way into the recipe.
    """
    return {name: index for index, name in enumerate(sorted(set(names)))}


def renumber(rows: Iterable[Row], ids: Mapping[str, int]) -> tuple[Row, ...]:
    """The same rows with the speaker column replaced by its integer id."""
    renumbered: list[Row] = []
    for row in rows:
        if row.speaker not in ids:
            message = f"no speaker id assigned for {row.speaker!r}; have {sorted(ids)}"
            raise TrainingError(message)
        renumbered.append(Row(audio=row.audio, ipa=row.ipa, speaker=str(ids[row.speaker])))
    return tuple(renumbered)


@dataclass(frozen=True, slots=True)
class OodSelection:
    """Phoneme lines for the adversarial branch, and why the rest were not taken."""

    lines: tuple[str, ...]
    considered: int
    rejected: Mapping[str, int]

    @property
    def shortest(self) -> int:
        return min((len(line) for line in self.lines), default=0)

    def as_dict(self) -> dict[str, object]:
        return {
            "lines": len(self.lines),
            "considered": self.considered,
            "shortest": self.shortest,
            "minimum": MIN_OOD_PHONEMES,
            "rejected": dict(sorted(self.rejected.items())),
        }


def select_ood(
    candidates: Iterable[str],
    vocabulary: Vocabulary,
    *,
    phonemize: Callable[[str], str],
    exclude: Container[str],
    size: int,
    minimum: int = MIN_OOD_PHONEMES,
) -> OodSelection:
    """Phoneme lines the SLM branch can read, drawn from text rather than transcripts.

    `exclude` is checked on the raw text before phonemisation, and it must hold both the
    prompt set and the speech transcripts: the adversarial branch reads these sentences
    every step, so a line that is also an evaluation prompt makes 12.1's comparison a
    measurement of what the model was trained on.
    """
    lines: list[str] = []
    rejected: Counter[str] = Counter()
    considered = 0
    seen: set[str] = set()
    for text in candidates:
        considered += 1
        if len(lines) >= size:
            break
        if text in exclude:
            rejected["held-out-elsewhere"] += 1
            continue
        try:
            ipa = phonemize(text)
        except PhonemeError:
            rejected["no-phoneme-rule"] += 1
            continue
        if vocabulary.unmapped(ipa):
            rejected[UNMAPPED] += 1
            continue
        if len(ipa) < minimum:
            rejected["too-short"] += 1
            continue
        if ipa in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(ipa)
        lines.append(ipa)
    return OodSelection(lines=tuple(lines), considered=considered, rejected=dict(rejected))


def write_ood(path: Path, lines: Sequence[str], *, minimum: int = MIN_OOD_PHONEMES) -> int:
    """Write the OOD file, refusing anything that would hang or crash the sampler.

    Two failures live here and neither announces itself. A file whose lines are all under
    `minimum` spins the sampler forever; a file of one line makes its `randint(0, n - 1)`
    raise, since the recipe never draws the last element.
    """
    if len(lines) < MIN_OOD_LINES:
        message = f"the OOD sampler cannot draw from {len(lines)} lines: {path}"
        raise TrainingError(message)
    short = [line for line in lines if len(line) < minimum]
    if short:
        message = (
            f"{len(short)} OOD lines are under {minimum} phonemes, which makes the "
            f"recipe's sampler loop until it happens to draw a longer one"
        )
        raise TrainingError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_list(path: Path, rows: Sequence[Row]) -> int:
    """Write a filelist and return the rows written.

    Two things are refused here rather than discovered on a GPU. An empty list, because
    the recipe reads it as a dataset of nothing and trains on it without complaining. And
    a speaker column that is not a decimal integer, because `int(speaker_id)` is the first
    thing its loader does with that field.
    """
    if not rows:
        message = f"refusing to write an empty filelist: {path}"
        raise TrainingError(message)
    named = sorted({row.speaker for row in rows if not row.speaker.isdecimal()})
    if named:
        message = (
            f"the speaker column must hold integers the recipe can cast, got {named}; "
            f"pass the rows through `renumber` first"
        )
        raise TrainingError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row.rendered() for row in rows) + "\n", encoding="utf-8")
    return len(rows)
