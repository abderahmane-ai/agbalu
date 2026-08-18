"""Which speakers of the built speech corpus can carry a voice.

Ranking is over the train split alone. Dev and test are the ASR evaluation splits, and a
voice built from a speaker in either would put synthetic audio of a held-out speaker into
the set Fadhma is scored on; `Voice.other_split_clips` is that check carried in the payload
rather than assumed.

Demographics are a property of the account and are written per *row*, mostly blank: the
second-ranked speaker carries `male_masculine` on 5,428 of its 16,002 train rows and nothing
on the rest, so reading the first row calls the voice unlabelled. Resolution is a majority
over the non-empty values of every table the speaker appears in.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Final

from agbalu.speech.corpus import Clip, rows

CLIP_TABLES: Final = ("train", "dev", "test", "validated", "invalidated", "other")
"""Common Voice tables keyed by `client_id`. `*_sentences.tsv` carry text without a
speaker and are not read here."""

LONG_MS: Final = 8_000
"""A clip long enough to carry phrase-level prosody. Under 1.2% of either candidate
reaches it, which is the constraint this phase is shaped by."""

TOP: Final = 2
"""Candidates profiled. Rank 3 drops to 6.94 h and carries no demographic label at all."""


class VoiceError(Exception):
    """A corpus from which no voice can be identified."""


@dataclass(frozen=True, slots=True)
class Demographics:
    """What the transcripts say about one account, and how much of it they say."""

    gender: str
    age: str
    labelled_rows: int
    total_rows: int

    def as_dict(self) -> dict[str, object]:
        return {
            "gender": self.gender,
            "age": self.age,
            "labelled_rows": self.labelled_rows,
            "total_rows": self.total_rows,
        }


@dataclass(frozen=True, slots=True)
class Voice:
    """One candidate speaker, ranked by how much audio the train split holds for them."""

    speaker: str
    rank: int
    clips: int
    duration_ms: int
    mean_ms: int
    median_ms: int
    max_ms: int
    long_clips: int
    other_split_clips: int
    demographics: Demographics

    @property
    def hours(self) -> float:
        return self.duration_ms / 3_600_000

    def as_dict(self) -> dict[str, object]:
        return {
            "speaker": self.speaker,
            "rank": self.rank,
            "clips": self.clips,
            "hours": round(self.hours, 4),
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "max_ms": self.max_ms,
            "long_clips": self.long_clips,
            "long_share": round(self.long_clips / self.clips, 6) if self.clips else None,
            "other_split_clips": self.other_split_clips,
            **self.demographics.as_dict(),
        }


def _resolve(values: Counter[str]) -> str:
    """The most frequent non-empty value, ties broken by sort order so it is a function
    of the set rather than of the order the tables were read in."""
    if not values:
        return ""
    return min(values.items(), key=lambda item: (-item[1], item[0]))[0]


def demographics(transcripts: Path, speakers: frozenset[str]) -> dict[str, Demographics]:
    """Gender and age per speaker, over every table that names them.

    A speaker with no non-empty value anywhere gets an empty string, which is a measured
    absence: Common Voice makes both fields optional and most contributors leave them.
    """
    genders: dict[str, Counter[str]] = defaultdict(Counter)
    ages: dict[str, Counter[str]] = defaultdict(Counter)
    seen: Counter[str] = Counter()

    for name in CLIP_TABLES:
        path = transcripts / f"{name}.tsv"
        if not path.is_file():
            continue
        for row in rows(path):
            client = row["client_id"]
            if client not in speakers:
                continue
            seen[client] += 1
            if row.get("gender"):
                genders[client][row["gender"]] += 1
            if row.get("age"):
                ages[client][row["age"]] += 1

    return {
        speaker: Demographics(
            gender=_resolve(genders[speaker]),
            age=_resolve(ages[speaker]),
            labelled_rows=sum(genders[speaker].values()),
            total_rows=seen[speaker],
        )
        for speaker in speakers
    }


def identify(
    train: list[Clip],
    held_out: list[Clip],
    transcripts: Path,
    *,
    top: int = TOP,
) -> tuple[Voice, ...]:
    """The `top` train speakers by total duration, with their demographics and leak check.

    `held_out` is dev and test together: its speakers are counted per candidate, never
    used to filter, so a corpus whose splits stopped being disjoint reports the overlap
    instead of quietly dropping the voice it should have refused.
    """
    if top < 1:
        message = f"top must be at least 1, got {top}"
        raise VoiceError(message)

    durations: dict[str, list[int]] = defaultdict(list)
    for clip in train:
        durations[clip.speaker].append(clip.duration_ms)
    if not durations:
        message = "no train clips: nothing to rank"
        raise VoiceError(message)

    elsewhere: Counter[str] = Counter(clip.speaker for clip in held_out)
    ordered = sorted(durations.items(), key=lambda item: (-sum(item[1]), item[0]))[:top]
    resolved = demographics(transcripts, frozenset(speaker for speaker, _ in ordered))

    return tuple(
        Voice(
            speaker=speaker,
            rank=position,
            clips=len(lengths),
            duration_ms=sum(lengths),
            mean_ms=round(sum(lengths) / len(lengths)),
            median_ms=round(median(lengths)),
            max_ms=max(lengths),
            long_clips=sum(1 for length in lengths if length >= LONG_MS),
            other_split_clips=elsewhere[speaker],
            demographics=resolved[speaker],
        )
        for position, (speaker, lengths) in enumerate(ordered, start=1)
    )
