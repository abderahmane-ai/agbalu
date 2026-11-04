"""Assign independent speaker and script partitions."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from agbalu.codec_tts.artifacts import Recording, Split, verify_conservation
from agbalu.codec_tts.configuration import Recipe


@dataclass(frozen=True)
class Partition:
    assignments: dict[str, Split]
    excluded: dict[str, str]


def priority(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest(), "big")


def speaker_split(speaker: str, recipe: Recipe) -> Split:
    draw = priority(recipe.seed, speaker) / 2**256
    if draw < recipe.test_fraction:
        return "test"
    if draw < recipe.test_fraction + recipe.validation_fraction:
        return "validation"
    return "train"


def partition_recordings(records: Sequence[Recording], recipe: Recipe) -> Partition:
    ownership: dict[str, set[Split]] = defaultdict(set)
    ranks: dict[Split, int] = {"train": 0, "validation": 1, "test": 2}
    identifiers: set[str] = set()
    hashes: set[str] = set()
    for record in records:
        if record.record_id in identifiers or record.audio_sha256 in hashes:
            message = (
                f"duplicate recording/audio at {record.shard}:{record.row}: {record.record_id}"
            )
            raise ValueError(message)
        identifiers.add(record.record_id)
        hashes.add(record.audio_sha256)
        ownership[record.text_key.casefold()].add(speaker_split(record.speaker, recipe))
    owners = {key: max(splits, key=ranks.__getitem__) for key, splits in ownership.items()}
    assignments: dict[str, Split] = {}
    excluded: dict[str, str] = {}
    for record in records:
        split = speaker_split(record.speaker, recipe)
        owner = owners[record.text_key.casefold()]
        if split != owner:
            excluded[record.record_id] = f"script reserved for {owner}"
        else:
            assignments[record.record_id] = split
    verify_conservation(len(records), len(assignments), len(excluded))
    return Partition(assignments, excluded)
