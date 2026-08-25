"""Task MP.B2: turning a speaker's clips into joined, punctuated training utterances.

Three things separate this from `agbalu.tts.corpus`, and each answers a measured defect in
what Matoub-82M was trained on (`docs/matoub_pro_design.md` §1).

- **The phoneme target keeps its punctuation**, and it is built from `Clip.text` — already
  normalised by `agbalu.speech.corpus`, homoglyphs repaired, marks and casing intact — rather
  than from the ASR target, which never carried one.
- **Consecutive Common Voice ids are one recording session**, so clips are joined into
  multi-sentence utterances. 80.8% of the target speaker's clips sit in a run of two or more.
  That is what supervises a pause after a full stop, and what lifts the corpus past the
  5-second floor `SLMAdversarialLoss` gates on.
- **The seam is not trimmed.** Inside a joined utterance each clip keeps its own trailing and
  leading silence exactly as recorded, so the inter-sentence pause and any breath taken in it
  are the speaker's rather than a constant chosen here. Only the outer edges are cut, and only
  down to `LEAD_KEEP_MS`/`TRAIL_KEEP_MS`, which is what v1 threw away — a median 370 ms per
  clip across 14,674 clips.

Levelling is unchanged and is taken per source clip on the raw audio, before joining: two
pieces levelled differently meet at an audible step.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, replace
from itertools import pairwise
from statistics import median
from typing import TYPE_CHECKING, Final

from agbalu.tts.corpus import (
    CLIPPED,
    MISMATCHED,
    NO_RULE,
    OUTPUT_RATE,
    PEAK_CEILING_DBFS,
    SILENT,
    TOO_LONG,
    TOO_SHORT,
    CorpusError,
    level_gain,
)
from agbalu.tts.g2p import PhonemeError
from agbalu.tts.g2p_pro import phonemize

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    from agbalu.speech.corpus import Clip
    from agbalu.tts.acoustics import ClipProfile

PACK_SECONDS: Final = 10.0
"""Longest utterance a run is packed into.

Measured on the target speaker: at this target the same 13.78 hours yield 8,737 utterances
carrying **4,920 internal sentence boundaries**, against 6,525 utterances and 3,337 boundaries
at a 20-second target. A boundary is an event, not a duration, and greedy packing to a longer
target makes fewer utterances out of the same clips. It also bounds the three memory terms
`max_len` does not — the aligner, the full-utterance style encoders and `mel_len_st`."""

MAX_SECONDS: Final = 14.0
"""Hard ceiling. A single source clip can exceed `PACK_SECONDS` on its own — the corpus's
longest is 10.5 s — and is kept; past this it is dropped by name."""

MIN_SECONDS: Final = 1.0
"""Below this an utterance is an interjection with no prosody to learn from. v1's floor was
2.0 s applied *after* trimming, which cost 1,319 of this speaker's clips for no reason a join
does not remove."""

SEAM_MAX_S: Final = 1.0
"""Longest silence a join may carry across a seam. Beyond it the run is split rather than the
gap clamped: a clamped gap is a fabricated pause, and a contributor who paused four seconds
between two recordings would otherwise teach a four-second full stop."""

LEAD_KEEP_MS: Final = 300.0
TRAIL_KEEP_MS: Final = 300.0
"""Silence kept outside the first and last speech in a joined utterance. Long enough to hold
an inhalation and a final release, capped so the duration predictor sees one figure rather
than whatever Common Voice padded. v1 kept 50 and 100."""

DEV_UTTERANCES: Final = 200
SMOKE_DEV: Final = 2
"""What the recipe validates on. The evaluation sets are not drawn here: `RESERVED` inherits
12.4's own `dev` and `cycle` clips, so Pro and Matoub-82M are scored on the same prompts and
Pro never trains on them."""

SEED: Final = 12

SLMADV_FLOOR_S: Final = 5.0
"""Reported because it is the number the adversarial branch turns on at.
`SLMAdversarialLoss` skips every sample whose `mel_input_length` is under
`slmadv_params.min_len`, so the share of the corpus above this is the share of joint steps
that can have an adversary at all. v1's median utterance was 2.87 s."""

RESERVED: Final = "held-out-by-12.4"
UNSEATED: Final = "no-session-id"

CLIP_ID: Final = re.compile(r"common_voice_kab_(\d+)\.mp3$")
"""Common Voice numbers a clip when it is uploaded, so consecutive ids are one sitting. The
claim the build tests rather than assumes: `report` carries the seam-gap distribution, and a
run whose members did not come from one session shows up there as a long gap and is split."""


@dataclass(frozen=True, slots=True)
class Source:
    """One clip, with everything the join needs and nothing it does not."""

    clip: str
    text: str
    ipa: str
    order: int
    duration_s: float
    speech_start_s: float
    speech_end_s: float
    speech_dbfs: float
    peak: float
    cer: float | None

    @property
    def lead_s(self) -> float:
        return self.speech_start_s

    @property
    def trail_s(self) -> float:
        return max(0.0, self.duration_s - self.speech_end_s)

    def stem(self) -> str:
        return self.clip.rsplit(".", 1)[0]


@dataclass(frozen=True, slots=True)
class Member:
    """One clip inside a joined utterance: where to cut it and how loud to make it."""

    clip: str
    start_s: float
    stop_s: float
    gain: float

    @property
    def seconds(self) -> float:
        return self.stop_s - self.start_s

    def as_dict(self) -> dict[str, object]:
        return {
            "clip": self.clip,
            "start_s": round(self.start_s, 4),
            "stop_s": round(self.stop_s, 4),
            "gain": round(self.gain, 6),
        }


@dataclass(frozen=True, slots=True)
class Joined:
    """One training utterance: an ordered run of clips from one recording session."""

    name: str
    voice: str
    text: str
    ipa: str
    members: tuple[Member, ...]
    seams: tuple[float, ...]
    split: str = "train"

    @property
    def seconds(self) -> float:
        return sum(member.seconds for member in self.members)

    @property
    def clips(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "voice": self.voice,
            "split": self.split,
            "text": self.text,
            "ipa": self.ipa,
            "seconds": round(self.seconds, 4),
            "seams": [round(seam, 4) for seam in self.seams],
            "members": [member.as_dict() for member in self.members],
        }


def clip_order(clip: str) -> int | None:
    """The recording's own id, or `None` for a name this scheme cannot read."""
    found = CLIP_ID.search(clip)
    return int(found.group(1)) if found else None


def verdict(
    clip: Clip,
    profile: ClipProfile,
    cer: float | None,
    *,
    reserved: frozenset[str],
    cer_limit: float,
    lexicon: Mapping[str, str] | None = None,
) -> Source | str:
    """One clip's verdict: a source to join, or the reason it was dropped."""
    if clip.clip in reserved:
        return RESERVED
    if profile.clipped_runs:
        return CLIPPED
    if profile.speech_dbfs is None or profile.speech_frame_share <= 0.0:
        return SILENT
    if profile.lead_silence_ms is None or profile.trail_silence_ms is None:
        return SILENT
    if cer is not None and cer > cer_limit:
        return MISMATCHED
    order = clip_order(clip.clip)
    if order is None:
        return UNSEATED
    try:
        ipa = phonemize(clip.text, lexicon=lexicon)
    except PhonemeError:
        return NO_RULE

    duration = profile.duration_s
    start = min(duration, profile.lead_silence_ms / 1000.0)
    stop = max(start, duration - profile.trail_silence_ms / 1000.0)
    return Source(
        clip=clip.clip,
        text=clip.text,
        ipa=ipa,
        order=order,
        duration_s=duration,
        speech_start_s=start,
        speech_end_s=stop,
        speech_dbfs=profile.speech_dbfs,
        peak=profile.peak,
        cer=cer,
    )


def target_level(sources: Sequence[Source]) -> float:
    """The level every clip is brought to: the set's own median gated speech level."""
    if not sources:
        message = "no surviving clips: there is no level to normalise to"
        raise CorpusError(message)
    return float(median(source.speech_dbfs for source in sources))


def sessions(
    sources: Iterable[Source], *, seam_max_s: float = SEAM_MAX_S
) -> Iterator[list[Source]]:
    """Runs of consecutive ids, split wherever the silence between two clips is too long."""
    run: list[Source] = []
    for source in sorted(sources, key=lambda item: item.order):
        if run and (source.order != run[-1].order + 1 or seam(run[-1], source) > seam_max_s):
            yield run
            run = []
        run.append(source)
    if run:
        yield run


def seam(before: Source, after: Source) -> float:
    """The silence a join carries: what the first clip trails plus what the second leads."""
    return before.trail_s + after.lead_s


def span(members: Sequence[Source], index: int) -> tuple[float, float]:
    """Where one member of a joined utterance is cut.

    The outer edges are trimmed to `LEAD_KEEP_MS`/`TRAIL_KEEP_MS` of their own silence; every
    interior edge keeps the whole clip, which is what puts the speaker's real pause — and the
    breath in it — into the training target instead of a constant.
    """
    source = members[index]
    start = 0.0 if index else max(0.0, source.speech_start_s - LEAD_KEEP_MS / 1000.0)
    last = index == len(members) - 1
    stop = (
        min(source.duration_s, source.speech_end_s + TRAIL_KEEP_MS / 1000.0)
        if last
        else source.duration_s
    )
    return start, max(start, stop)


def _seconds(members: Sequence[Source]) -> float:
    """How long the utterance these members would make is, cut as `span` cuts them."""
    total = 0.0
    for index in range(len(members)):
        start, stop = span(members, index)
        total += stop - start
    return total


def pack(run: Sequence[Source], *, pack_seconds: float = PACK_SECONDS) -> Iterator[list[Source]]:
    """A session cut into utterances no longer than `pack_seconds`, in recording order."""
    current: list[Source] = []
    for source in run:
        candidate = [*current, source]
        if current and _seconds(candidate) > pack_seconds:
            yield current
            current = [source]
            continue
        current = candidate
    if current:
        yield current


def join(
    members: Sequence[Source],
    voice: str,
    target_dbfs: float,
    *,
    ceiling_dbfs: float = PEAK_CEILING_DBFS,
) -> Joined | str:
    """One utterance from an ordered run, or the reason it cannot be one."""
    if not members:
        message = "an utterance needs at least one clip"
        raise CorpusError(message)
    cuts: list[Member] = []
    for index, source in enumerate(members):
        start, stop = span(members, index)
        cuts.append(
            Member(
                clip=source.clip,
                start_s=start,
                stop_s=stop,
                gain=level_gain(
                    source.speech_dbfs, source.peak, target_dbfs, ceiling_dbfs=ceiling_dbfs
                ),
            )
        )
    seconds = sum(cut.seconds for cut in cuts)
    if seconds < MIN_SECONDS:
        return TOO_SHORT
    if seconds > MAX_SECONDS:
        return TOO_LONG
    return Joined(
        name=f"{members[0].stem()}-{len(members)}",
        voice=voice,
        text=" ".join(source.text for source in members),
        ipa=" ".join(source.ipa for source in members),
        members=tuple(cuts),
        seams=tuple(seam(before, after) for before, after in pairwise(members)),
    )


def build(
    sources: Sequence[Source],
    voice: str,
    *,
    pack_seconds: float = PACK_SECONDS,
    seam_max_s: float = SEAM_MAX_S,
) -> tuple[list[Joined], dict[str, int], float]:
    """Every utterance a speaker's surviving clips make, and what the packing dropped."""
    target = target_level(sources)
    utterances: list[Joined] = []
    dropped: dict[str, int] = {}
    for run in sessions(sources, seam_max_s=seam_max_s):
        for members in pack(run, pack_seconds=pack_seconds):
            result = join(members, voice, target)
            if isinstance(result, str):
                dropped[result] = dropped.get(result, 0) + len(members)
                continue
            utterances.append(result)
    return utterances, dropped, target


def assign_dev(
    utterances: Sequence[Joined], *, dev: int = DEV_UTTERANCES, seed: int = SEED
) -> list[Joined]:
    """Draw the recipe's validation list, seeded, over the whole set rather than a prefix."""
    if dev < 0:
        message = f"the validation list cannot hold {dev} utterances"
        raise CorpusError(message)
    if dev >= len(utterances):
        message = f"{dev} validation utterances leave nothing to train on from {len(utterances)}"
        raise CorpusError(message)
    order = list(range(len(utterances)))
    random.Random(seed).shuffle(order)
    chosen = set(order[:dev])
    return [
        replace(utterance, split="dev" if index in chosen else "train")
        for index, utterance in enumerate(utterances)
    ]


def held_out(manifest: Path) -> frozenset[str]:
    """The clips 12.4 kept back, read from its own manifest.

    Inheriting them is what makes MP.A1 and MP.E5 the same measurement: Matoub-82M's floor,
    baseline and cycle conditions were drawn from these, and an utterance Pro trained on
    would turn its own Cycle-CER into a memorisation score.
    """
    if not manifest.is_file():
        message = f"12.4's manifest is not on the volume: {manifest}"
        raise CorpusError(message)
    reserved: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") in {"dev", "cycle"}:
            reserved.add(str(row["clip"]))
    if not reserved:
        message = f"{manifest} reserves no clips, so nothing is held out from Pro"
        raise CorpusError(message)
    return frozenset(reserved)


def assign_speaker_ids(names: Iterable[str]) -> dict[str, int]:
    """Voice name to the integer the recipe requires, assigned in sorted order."""
    return {name: index for index, name in enumerate(sorted(set(names)))}


def training_list(
    utterances: Sequence[Joined],
    *,
    audio_dir: str,
    speaker_ids: Mapping[str, int] | None = None,
) -> Iterator[str]:
    """The recipe's list format: audio path, phoneme string, integer speaker id."""
    for utterance in utterances:
        if speaker_ids is not None and utterance.voice in speaker_ids:
            speaker_str = str(speaker_ids[utterance.voice])
        elif utterance.voice.isdigit():
            speaker_str = utterance.voice
        else:
            speaker_str = "0"
        yield f"{audio_dir}/{utterance.name}.wav|{utterance.ipa}|{speaker_str}"


def report(
    voice: str,
    speaker: str,
    utterances: Sequence[Joined],
    dropped: Mapping[str, int],
    target_dbfs: float,
    *,
    considered: int,
) -> dict[str, object]:
    """What the build made, what it dropped, and the distributions the joins rest on."""
    seconds = [utterance.seconds for utterance in utterances]
    seams = sorted(seam for utterance in utterances for seam in utterance.seams)
    clips = sum(utterance.clips for utterance in utterances)
    symbols = sorted({symbol for utterance in utterances for symbol in utterance.ipa})
    lengths = sorted(len(utterance.ipa) for utterance in utterances)
    return {
        "voice": voice,
        "speaker": speaker,
        "utterances": len(utterances),
        "clips": clips,
        "considered": considered,
        "dropped": dict(sorted(dropped.items())),
        "seconds": round(sum(seconds), 2),
        "hours": round(sum(seconds) / 3600, 4),
        "duration_s": _spread(sorted(seconds)),
        "over_slmadv_floor": sum(1 for value in seconds if value > SLMADV_FLOOR_S),
        "with_internal_boundary": sum(1 for u in utterances if u.clips > 1),
        "seam_s": _spread(seams),
        "phonemes": _spread([float(value) for value in lengths]),
        "splits": {
            split: sum(1 for u in utterances if u.split == split) for split in ("train", "dev")
        },
        "inventory": "".join(symbols),
        "inventory_size": len(symbols),
        "output_rate": OUTPUT_RATE,
        "target_speech_dbfs": round(target_dbfs, 3),
        "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
        "pack": {
            "pack_seconds": PACK_SECONDS,
            "seam_max_s": SEAM_MAX_S,
            "lead_keep_ms": LEAD_KEEP_MS,
            "trail_keep_ms": TRAIL_KEEP_MS,
            "min_seconds": MIN_SECONDS,
            "max_seconds": MAX_SECONDS,
        },
    }


def _spread(values: Sequence[float]) -> dict[str, float | int | None]:
    """A distribution, reported rather than averaged: a mean hides the tail that decides."""
    if not values:
        return {"measured": 0, "p10": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "measured": len(ordered),
        "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
        "median": round(float(median(ordered)), 4),
        "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
        "max": round(ordered[-1], 4),
    }
