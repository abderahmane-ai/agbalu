"""Speaker-stratified disjoint partitioning for Fadhma 2.

Enforces strict speaker-level isolation across train (90%), dev (5%), and test (5%).
Constructs an acoustic generalization challenge set (`test_phrase_disjoint`) whose
target transcripts never appear in the training or validation splits.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final

from agbalu.speech.manifest import AudioRecord

DEFAULT_SEED: Final[int] = 42
TARGET_DEV_RATIO: Final[float] = 0.05
TARGET_TEST_RATIO: Final[float] = 0.05
DEFAULT_MAX_EVAL_SPEAKER_HOURS: Final[float] = 3.0


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Statistics for an individual partition."""

    name: str
    clips: int
    hours: float
    speakers: int


@dataclass(frozen=True, slots=True)
class PartitionReport:
    """Summary of the full partitioned dataset with isolation verification."""

    splits: list[SplitSummary]
    speaker_overlaps: dict[str, int]
    phrase_disjoint_clips: int
    phrase_disjoint_hours: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitDataset:
    """Container holding the partitioned splits."""

    train: list[AudioRecord]
    dev: list[AudioRecord]
    test: list[AudioRecord]
    test_phrase_disjoint: list[AudioRecord]

    def report(self) -> PartitionReport:
        train_spk = {r.speaker_id for r in self.train}
        dev_spk = {r.speaker_id for r in self.dev}
        test_spk = {r.speaker_id for r in self.test}

        overlaps = {
            "train_dev": len(train_spk & dev_spk),
            "train_test": len(train_spk & test_spk),
            "dev_test": len(dev_spk & test_spk),
        }

        def summarize(name: str, records: Sequence[AudioRecord]) -> SplitSummary:
            hrs = round(sum(r.duration for r in records) / 3600.0, 2)
            spks = len({r.speaker_id for r in records})
            return SplitSummary(name=name, clips=len(records), hours=hrs, speakers=spks)

        phrase_hrs = round(sum(r.duration for r in self.test_phrase_disjoint) / 3600.0, 2)

        return PartitionReport(
            splits=[
                summarize("train", self.train),
                summarize("dev", self.dev),
                summarize("test", self.test),
                summarize("test_phrase_disjoint", self.test_phrase_disjoint),
            ],
            speaker_overlaps=overlaps,
            phrase_disjoint_clips=len(self.test_phrase_disjoint),
            phrase_disjoint_hours=phrase_hrs,
        )


def verify_speaker_isolation(splits: SplitDataset) -> bool:
    """Assert zero speaker overlap between train, dev, and test partitions."""
    train_spk = {r.speaker_id for r in splits.train}
    dev_spk = {r.speaker_id for r in splits.dev}
    test_spk = {r.speaker_id for r in splits.test}

    if train_spk & dev_spk:
        return False
    if train_spk & test_spk:
        return False
    return not bool(dev_spk & test_spk)


def _allocate_speakers(
    speaker_durations: dict[str, float],
    target_dev_dur: float,
    target_test_dur: float,
    max_eval_sec: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Partition speaker IDs into dev, test, and train sets."""
    rng = random.Random(seed)
    locked_train = {s for s, d in speaker_durations.items() if d > max_eval_sec}
    candidates = [s for s, d in speaker_durations.items() if d <= max_eval_sec]

    candidate_dur = sum(speaker_durations[s] for s in candidates)
    if candidate_dur < (target_dev_dur + target_test_dur):
        candidates = list(speaker_durations.keys())
        locked_train = set()

    candidates.sort()
    rng.shuffle(candidates)

    dev_speakers: set[str] = set()
    test_speakers: set[str] = set()
    train_speakers: set[str] = set(locked_train)

    current_dev_dur = 0.0
    current_test_dur = 0.0

    for spk in candidates:
        dur = speaker_durations[spk]
        if current_dev_dur < target_dev_dur:
            dev_speakers.add(spk)
            current_dev_dur += dur
        elif current_test_dur < target_test_dur:
            test_speakers.add(spk)
            current_test_dur += dur
        else:
            train_speakers.add(spk)

    return train_speakers, dev_speakers, test_speakers


def partition_records(
    records: Sequence[AudioRecord],
    *,
    dev_ratio: float = TARGET_DEV_RATIO,
    test_ratio: float = TARGET_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    max_eval_speaker_hours: float = DEFAULT_MAX_EVAL_SPEAKER_HOURS,
) -> SplitDataset:
    """Group clips by speaker and allocate full speakers to train, dev, and test.

    Speakers with cumulative duration exceeding `max_eval_speaker_hours` are locked
    into the training set to prevent mega-contributors from monopolizing validation
    splits and to maximize acoustic diversity during training.

    Remaining candidate speakers are shuffled and allocated to dev and test until
    the target ratios are satisfied, ensuring realistic multi-speaker benchmarks.
    """
    speaker_durations: dict[str, float] = defaultdict(float)
    for record in records:
        speaker_durations[record.speaker_id] += record.duration

    total_duration = sum(speaker_durations.values())
    target_dev_dur = total_duration * dev_ratio
    target_test_dur = total_duration * test_ratio
    max_eval_sec = max_eval_speaker_hours * 3600.0

    _train_spks, dev_spks, test_spks = _allocate_speakers(
        speaker_durations, target_dev_dur, target_test_dur, max_eval_sec, seed
    )

    train_records: list[AudioRecord] = []
    dev_records: list[AudioRecord] = []
    test_records: list[AudioRecord] = []

    for record in records:
        spk = record.speaker_id
        if spk in dev_spks:
            dev_records.append(record)
        elif spk in test_spks:
            test_records.append(record)
        else:
            train_records.append(record)

    # Construct phrase-disjoint test set: target texts unseen in train or dev
    train_dev_phrases = {r.target_text for r in train_records} | {
        r.target_text for r in dev_records
    }
    test_phrase_disjoint = [r for r in test_records if r.target_text not in train_dev_phrases]

    dataset = SplitDataset(
        train=train_records,
        dev=dev_records,
        test=test_records,
        test_phrase_disjoint=test_phrase_disjoint,
    )

    if not verify_speaker_isolation(dataset):
        msg = "Failed speaker isolation invariant in partitioning."
        raise ValueError(msg)

    return dataset
