"""The prompt set Matoub and every system it is compared against are scored on.

One set, built once, read by the baseline (task 12.1) and by the cycle harness (12.5),
because a Cycle-CER is only interpretable against the same text decoded by the same
decoder. Four properties, each a filter here rather than an assumption downstream:

- **The decoder has not heard the sentence.** Drawn from the speaker-disjoint test
  split, and further filtered against the *text* of train and dev: Common Voice
  sentences recur across splits under different speakers, and 220 of the 9,494
  candidates carry a sentence Fadhma trained on. A prompt it memorised measures its
  memory, not the synthesis.
- **No scripture**, per `agbalu.tts.scripture`.
- **Long enough to score.** The corpus mean is 4.58 words; a two-word prompt makes a
  character error rate that moves in steps of several points.
- **Spread across speakers**, because the real audio of these same clips is the floor
  the synthetic conditions are read against, and a floor measured on one voice is that
  voice's number.

The selection is a seeded shuffle of the candidates sorted by clip name, so it is a
function of the *set* of clips rather than of the order the file was written in.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Final

from agbalu.speech.corpus import Clip

MIN_WORDS: Final = 4
"""Words in the shortest admissible prompt.

Cumulative over the 15,003-clip test split: 5,509 clips are three words or fewer and
9,494 are four or more, so this is the largest floor that still leaves a pool an order
of magnitude above `DEFAULT_SIZE`."""

SPEAKER_CAP: Final = 2
"""Clips one speaker may contribute. 778 speakers clear `MIN_WORDS` and one of them
holds 87 clips, so an uncapped sample of 1,000 would be measurably that speaker's."""

DEFAULT_SIZE: Final = 1_000
"""Prompts in the set. At the split's mean length this is ~4,600 reference characters
per condition, and one synthesis and one decode each — minutes of A10, not hours."""

SEED: Final = 12

TOO_SHORT: Final = "too-short"
DUPLICATE: Final = "duplicate-text"
SEEN: Final = "seen-in-training"
CAPPED: Final = "speaker-cap"


class PromptError(Exception):
    """A prompt set that cannot be built or read."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """One sentence, with the clip whose real audio is its floor."""

    clip: str
    speaker: str
    duration_ms: int
    text: str
    target: str

    @property
    def words(self) -> int:
        return len(self.target.split())

    def as_dict(self) -> dict[str, object]:
        return {
            "clip": self.clip,
            "speaker": self.speaker,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "target": self.target,
        }

    def as_clip(self, duration_ms: int | None = None) -> Clip:
        """The record the ASR scorer batches and scores.

        `duration_ms` overrides the human clip's duration, which is what a synthesised
        condition passes: batches are bucketed by duration, and a synthesis is not the
        length of the recording it was derived from.
        """
        return Clip(
            clip=self.clip,
            speaker=self.speaker,
            split="prompt",
            duration_ms=self.duration_ms if duration_ms is None else duration_ms,
            text=self.text,
            target=self.target,
            repaired=False,
        )


@dataclass(frozen=True, slots=True)
class Selection:
    """The set, and what the pool it came from lost on the way."""

    prompts: tuple[Prompt, ...]
    candidates: int
    rejected: Mapping[str, int]
    size: int
    seed: int

    @property
    def speakers(self) -> int:
        return len({prompt.speaker for prompt in self.prompts})

    @property
    def seconds(self) -> float:
        return sum(prompt.duration_ms for prompt in self.prompts) / 1000

    @property
    def words(self) -> int:
        return sum(prompt.words for prompt in self.prompts)

    def as_dict(self) -> dict[str, object]:
        kept = len(self.prompts)
        return {
            "prompts": kept,
            "requested": self.size,
            "candidates": self.candidates,
            "rejected": dict(sorted(self.rejected.items())),
            "speakers": self.speakers,
            "words": self.words,
            "words_mean": round(self.words / kept, 3) if kept else 0.0,
            "reference_seconds": round(self.seconds, 1),
            "min_words": MIN_WORDS,
            "speaker_cap": SPEAKER_CAP,
            "seed": self.seed,
        }


def select(
    clips: Sequence[Clip],
    *,
    exclude: Callable[[str], str | None],
    seen: Container[str] = frozenset(),
    size: int = DEFAULT_SIZE,
    min_words: int = MIN_WORDS,
    speaker_cap: int = SPEAKER_CAP,
    seed: int = SEED,
) -> Selection:
    """Filter `clips` to a prompt set of at most `size`.

    `exclude` names why a text is inadmissible, or returns `None`; `seen` holds the
    targets the decoder was trained or validated on. Both are supplied rather than
    resolved here, so this stays free of any import that reads a corpus.
    """
    if size < 1:
        message = f"a prompt set of {size} cannot be scored"
        raise PromptError(message)

    rejected: dict[str, int] = {}
    pool: list[Clip] = []
    texts: set[str] = set()

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for clip in sorted(clips, key=lambda c: c.clip):
        if len(clip.target.split()) < min_words:
            reject(TOO_SHORT)
        elif clip.target in texts:
            reject(DUPLICATE)
        elif clip.target in seen:
            reject(SEEN)
        elif (reason := exclude(clip.text)) is not None:
            reject(reason)
        else:
            texts.add(clip.target)
            pool.append(clip)

    if not pool:
        detail = ", ".join(f"{reason}: {count}" for reason, count in sorted(rejected.items()))
        message = f"no clip survived the filters, so there is nothing to score ({detail})"
        raise PromptError(message)

    order = list(pool)
    Random(seed).shuffle(order)

    taken: dict[str, int] = {}
    chosen: list[Clip] = []
    for clip in order:
        if len(chosen) == size:
            break
        if taken.get(clip.speaker, 0) >= speaker_cap:
            reject(CAPPED)
            continue
        taken[clip.speaker] = taken.get(clip.speaker, 0) + 1
        chosen.append(clip)

    prompts = tuple(
        Prompt(
            clip=clip.clip,
            speaker=clip.speaker,
            duration_ms=clip.duration_ms,
            text=clip.text,
            target=clip.target,
        )
        for clip in sorted(chosen, key=lambda c: c.clip)
    )
    return Selection(prompts=prompts, candidates=len(pool), rejected=rejected, size=size, seed=seed)


def write(path: Path, prompts: Sequence[Prompt]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for prompt in prompts:
            sink.write(json.dumps(prompt.as_dict(), ensure_ascii=False) + "\n")
    return path


def read(path: Path) -> list[Prompt]:
    """The built prompt set, in file order."""
    if not path.is_file():
        message = f"prompt set not built: {path}; run `make tts TASK=prompts`"
        raise PromptError(message)
    out: list[Prompt] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                out.append(
                    Prompt(
                        clip=row["clip"],
                        speaker=row["speaker"],
                        duration_ms=int(row["duration_ms"]),
                        text=row["text"],
                        target=row["target"],
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                message = f"{path}:{number} is not a prompt record"
                raise PromptError(message) from error
    if not out:
        message = f"prompt set is empty: {path}"
        raise PromptError(message)
    return out
