"""Task 12.4: turning a speaker's share of an ASR corpus into a voice corpus.

Two arms are built from one set of decisions. `restored` is the clips through Sidon;
`raw` is the same clips untouched. Every other operation — which clips survive, where
each is cut, how much gain it takes, which split it lands in — is computed **once, from
the raw audio, and applied to both**, so 12.7's ablation varies restoration and nothing
else. A filter run separately per arm would be measuring its own thresholds.

Three decisions are worth stating because none is inherited from the base recipe, which
performs no trimming and no normalisation at all:

- **The cut comes from the profile, not a new detector.** `acoustics.profile` already
  reports where speech starts and ends, under the gate 12.3 was validated with, so the
  corpus is cut by the instrument that measured it rather than by a second opinion.
- **The level target is the corpus's own median gated speech level**, so the gain applied
  is the smallest that makes the set consistent. Consistency is the property being bought:
  rank 1 spans 17.8 dB between its 10th and 90th percentile, and a duration predictor
  trained across that learns loudness as a property of the sentence.
- **The transcript gate is deliberately loose.** Fadhma trained on both these speakers, so
  its error rate here is optimistically biased and cannot set a tight threshold. It is used
  only to catch a clip whose transcript is a different sentence; the full distribution goes
  into the stats so a later pass can tighten it against evidence rather than against taste.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING, Final

from agbalu.tts.g2p import PhonemeError, phonemize
from agbalu.tts.kokoro import fold
from agbalu.tts.prompts import MIN_WORDS

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from agbalu.speech.corpus import Clip
    from agbalu.tts.acoustics import ClipProfile

ARMS: Final = ("restored", "raw")
"""Restoration first: it is the primary arm and `raw` is what 12.7 measures it against."""

VOICE_NAMES: Final[Mapping[str, str]] = {
    "female_feminine": "kab_female",
    "male_masculine": "kab_male",
}
"""Common Voice's own gender values, as `agbalu.tts.voices` resolves them. A voice with no
label is refused rather than numbered: a corpus filename is a claim about who is speaking."""

OUTPUT_RATE: Final = 24_000
"""What the base synthesises at, and therefore what it trains on."""

LEAD_PAD_MS: Final = 50.0
TRAIL_PAD_MS: Final = 100.0
"""Silence kept either side of the speech. Chosen, not measured — what matters is that
every clip carries the same amount, since the model learns the envelope along with the
speech. The trail is the longer of the two because a final release cut flush clicks."""

MIN_SECONDS: Final = 2.0
MAX_SECONDS: Final = 30.0
"""The base recipe's window, applied *after* trimming. Nothing in either voice approaches
the ceiling; the floor is what trimming pushes clips through, and the count that falls
past it is reported rather than absorbed."""

PEAK_CEILING_DBFS: Final = -1.0
"""Headroom left after gain. A clip that would clip is attenuated to this instead."""

CER_LIMIT: Final = 0.5
"""Half the characters wrong is not a transcription error, it is a different sentence."""

DEV_CLIPS: Final = 200
CYCLE_CLIPS: Final = 300
"""`dev` is the recipe's validation list. `cycle` is 12.5's floor condition — this
speaker's own held-out audio against the same text — and it is never trained on."""

SMOKE_DEV: Final = 2
SMOKE_CYCLE: Final = 3
"""What a capped build holds out instead. A 20-clip cap survives as about a dozen
utterances, which 500 held-out clips leave nothing to train on — so the recipe's sizes make
the smoke the one call that cannot run. These keep every split non-empty at that size."""

SEED: Final = 12

CLIPPED: Final = "clipped"
SILENT: Final = "no-speech-detected"
MISMATCHED: Final = "transcript-mismatch"
TOO_SHORT: Final = "too-short-after-trim"
TOO_LONG: Final = "too-long"
NO_RULE: Final = "no-phoneme-rule"

TRAIN: Final = "train"
DEV: Final = "dev"
CYCLE: Final = "cycle"
SPLITS: Final = (TRAIN, DEV, CYCLE)


class CorpusError(Exception):
    """A voice from which no corpus can be cut."""


@dataclass(frozen=True, slots=True)
class Utterance:
    """One surviving clip: where to cut it, how loud to make it, and what it says."""

    clip: str
    text: str
    target: str
    ipa: str
    start_s: float
    stop_s: float
    speech_dbfs: float
    peak: float
    cer: float | None
    split: str = TRAIN
    gain: float = 1.0

    @property
    def seconds(self) -> float:
        return self.stop_s - self.start_s

    @property
    def words(self) -> int:
        return len(self.target.split())

    def stem(self) -> str:
        return self.clip.rsplit(".", 1)[0]

    def as_dict(self) -> dict[str, object]:
        return {
            "clip": self.clip,
            "split": self.split,
            "text": self.text,
            "target": self.target,
            "ipa": self.ipa,
            "start_s": round(self.start_s, 4),
            "stop_s": round(self.stop_s, 4),
            "seconds": round(self.seconds, 4),
            "gain": round(self.gain, 6),
            "speech_dbfs": round(self.speech_dbfs, 3),
            "peak": round(self.peak, 6),
            "cer": None if self.cer is None else round(self.cer, 6),
        }


def name_for(gender: str) -> str:
    """The corpus name for a voice, from what the transcripts say about the account."""
    resolved = VOICE_NAMES.get(gender)
    if resolved is None:
        message = (
            f"voice carries gender {gender!r}, which names no corpus; the two candidates "
            f"resolve to {sorted(VOICE_NAMES)}"
        )
        raise CorpusError(message)
    return resolved


def bounds(
    profile: ClipProfile,
    *,
    lead_pad_ms: float = LEAD_PAD_MS,
    trail_pad_ms: float = TRAIL_PAD_MS,
) -> tuple[float, float]:
    """Padded speech bounds in seconds, clamped to the clip.

    A clip whose gate found no speech returns its whole length; `consider` is what drops
    it, so the two decisions stay separable.
    """
    duration = profile.duration_s
    if profile.lead_silence_ms is None or profile.trail_silence_ms is None:
        return 0.0, duration
    start = max(0.0, profile.lead_silence_ms / 1000.0 - lead_pad_ms / 1000.0)
    stop = min(duration, duration - profile.trail_silence_ms / 1000.0 + trail_pad_ms / 1000.0)
    if stop <= start:
        return 0.0, duration
    return start, stop


def consider(
    clip: Clip,
    profile: ClipProfile,
    cer: float | None,
    *,
    lexicon: Mapping[str, str] | None = None,
    cer_limit: float = CER_LIMIT,
) -> Utterance | str:
    """One clip's verdict: an utterance to keep, or the reason it was dropped."""
    if profile.clipped_runs:
        return CLIPPED
    if profile.speech_dbfs is None or profile.speech_frame_share <= 0.0:
        return SILENT
    if cer is not None and cer > cer_limit:
        return MISMATCHED

    start, stop = bounds(profile)
    seconds = stop - start
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return TOO_SHORT if seconds < MIN_SECONDS else TOO_LONG

    try:
        ipa = fold(phonemize(clip.target, lexicon=lexicon))
    except PhonemeError:
        return NO_RULE

    return Utterance(
        clip=clip.clip,
        text=clip.text,
        target=clip.target,
        ipa=ipa,
        start_s=start,
        stop_s=stop,
        speech_dbfs=profile.speech_dbfs,
        peak=profile.peak,
        cer=cer,
    )


def target_level(utterances: Sequence[Utterance]) -> float:
    """The level every clip is brought to: the set's own median gated speech level."""
    if not utterances:
        message = "no surviving clips: there is no level to normalise to"
        raise CorpusError(message)
    return float(median(utterance.speech_dbfs for utterance in utterances))


def _ceiling(ceiling_dbfs: float) -> float:
    return float(10.0 ** (ceiling_dbfs / 20.0))


def gain_for(
    utterance: Utterance, target_dbfs: float, *, ceiling_dbfs: float = PEAK_CEILING_DBFS
) -> float:
    """Linear gain onto `target_dbfs`, attenuated further if the peak would pass the ceiling."""
    wanted = float(10.0 ** ((target_dbfs - utterance.speech_dbfs) / 20.0))
    if utterance.peak <= 0.0:
        return wanted
    return float(min(wanted, _ceiling(ceiling_dbfs) / utterance.peak))


def levelled(utterances: Sequence[Utterance]) -> tuple[list[Utterance], float]:
    """Every utterance with its gain resolved, and the level they were brought to."""
    target = target_level(utterances)
    return [
        replace(utterance, gain=gain_for(utterance, target)) for utterance in utterances
    ], target


def split_sizes(limit: int) -> tuple[int, int]:
    """The `dev` and `cycle` sizes a build holds out, given the cap on its clips."""
    return (SMOKE_DEV, SMOKE_CYCLE) if limit > 0 else (DEV_CLIPS, CYCLE_CLIPS)


def assign_splits(
    utterances: Sequence[Utterance],
    *,
    dev: int = DEV_CLIPS,
    cycle: int = CYCLE_CLIPS,
    min_words: int = MIN_WORDS,
    seed: int = SEED,
) -> list[Utterance]:
    """Draw the held-out sets, seeded, over the whole corpus rather than a prefix.

    The cycle set takes the same length floor as 12.1's prompt set, because it is scored
    by the same instrument and a two-word utterance measures the decoder's priors.
    """
    if dev < 0 or cycle < 0:
        message = f"split sizes must not be negative, got dev={dev} cycle={cycle}"
        raise CorpusError(message)
    if dev + cycle >= len(utterances):
        message = (
            f"{dev} dev and {cycle} cycle clips leave nothing to train on from "
            f"{len(utterances)} utterances"
        )
        raise CorpusError(message)

    draw = random.Random(seed)
    order = list(utterances)
    draw.shuffle(order)

    long_enough = [u for u in order if u.words >= min_words]
    if len(long_enough) < cycle:
        message = (
            f"only {len(long_enough)} utterances reach {min_words} words; the cycle set "
            f"needs {cycle}"
        )
        raise CorpusError(message)

    chosen: dict[str, str] = {u.clip: CYCLE for u in long_enough[:cycle]}
    for utterance in order:
        if len(chosen) >= cycle + dev:
            break
        chosen.setdefault(utterance.clip, DEV)
    return [replace(u, split=chosen.get(u.clip, TRAIN)) for u in utterances]


def apply(
    audio: NDArray[np.float32],
    rate: int,
    utterance: Utterance,
    *,
    ceiling_dbfs: float = PEAK_CEILING_DBFS,
) -> NDArray[np.float32]:
    """Cut `audio` to the utterance's bounds, apply its gain, and scale it under the ceiling.

    The gain is the decision and is identical across arms. The ceiling is a property of the
    waveform being written rather than of the decision: it was resolved against the raw
    clip's peak, and restoration moves that peak, so each arm is re-checked against what it
    actually writes. An arm that passes the ceiling is attenuated onto it, never clamped —
    a clamp flattens the top of the wave, which is what `consider` drops a raw clip for.
    """
    import numpy as np

    if rate < 1:
        message = f"sample rate must be positive, got {rate}"
        raise CorpusError(message)
    start = max(0, round(utterance.start_s * rate))
    stop = min(int(audio.size), round(utterance.stop_s * rate))
    if stop <= start:
        message = f"{utterance.clip} cuts to nothing at {rate} Hz"
        raise CorpusError(message)
    cut = np.asarray(audio[start:stop], dtype=np.float32) * np.float32(utterance.gain)
    peak = float(np.abs(cut).max())
    if not np.isfinite(peak):
        message = f"{utterance.clip} carries non-finite samples at {rate} Hz"
        raise CorpusError(message)
    ceiling = _ceiling(ceiling_dbfs)
    if peak > ceiling:
        cut = cut * np.float32(ceiling / peak)
    return np.ascontiguousarray(cut, dtype=np.float32)


def training_list(utterances: Sequence[Utterance], voice: str, *, audio_dir: str) -> Iterator[str]:
    """The base recipe's list format: audio path, phoneme string, speaker name."""
    for utterance in utterances:
        yield f"{audio_dir}/{utterance.stem()}.wav|{utterance.ipa}|{voice}"


def write_lists(root: Path, voice: str, utterances: Sequence[Utterance], *, audio_dir: str) -> None:
    """One list per split, written beside the arm whose audio they name.

    Each arm gets its own lists because each names its own wav files; the manifest that
    carries everything a list drops is written once, at the voice, since it is a property
    of the decisions rather than of the audio they were applied to.
    """
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [u for u in utterances if u.split == split]
        lines = list(training_list(rows, voice, audio_dir=audio_dir))
        (root / f"{voice}.{split}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def write_manifest(root: Path, voice: str, utterances: Sequence[Utterance]) -> Path:
    """Every decision taken per clip, including the ones the lists cannot carry."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{voice}.manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for utterance in utterances:
            handle.write(json.dumps(utterance.as_dict(), ensure_ascii=False) + "\n")
    return path


def report(
    voice: str,
    speaker: str,
    utterances: Sequence[Utterance],
    dropped: Mapping[str, int],
    target_dbfs: float,
) -> dict[str, object]:
    """What the build kept, what it dropped and why, and the levels it moved things to."""
    rates = sorted(u.cer for u in utterances if u.cer is not None)
    symbols = sorted({symbol for utterance in utterances for symbol in utterance.ipa})
    per_split = {
        split: {
            "clips": sum(1 for u in utterances if u.split == split),
            "seconds": round(sum(u.seconds for u in utterances if u.split == split), 2),
        }
        for split in SPLITS
    }
    gains = sorted(u.gain for u in utterances)
    return {
        "voice": voice,
        "speaker": speaker,
        "clips": len(utterances),
        "seconds": round(sum(u.seconds for u in utterances), 2),
        "hours": round(sum(u.seconds for u in utterances) / 3600, 4),
        "considered": len(utterances) + sum(dropped.values()),
        "dropped": dict(sorted(dropped.items())),
        "splits": per_split,
        # What the base has to be able to represent. Recorded rather than asserted here:
        # the token table lives with the checkpoint, and a symbol it cannot hold is dropped
        # silently at synthesis rather than raised.
        "inventory": "".join(symbols),
        "inventory_size": len(symbols),
        "output_rate": OUTPUT_RATE,
        "target_speech_dbfs": round(target_dbfs, 3),
        "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
        "gain": {
            "min": round(gains[0], 4) if gains else None,
            "median": round(float(median(gains)), 4) if gains else None,
            "max": round(gains[-1], 4) if gains else None,
        },
        "transcript_cer": {
            "measured": len(rates),
            "median": round(float(median(rates)), 6) if rates else None,
            "limit": CER_LIMIT,
        },
        "trim": {
            "lead_pad_ms": LEAD_PAD_MS,
            "trail_pad_ms": TRAIL_PAD_MS,
            "min_seconds": MIN_SECONDS,
            "max_seconds": MAX_SECONDS,
        },
    }
