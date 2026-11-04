"""Choose distinct same-speaker reference recordings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from agbalu.codec_tts.artifacts import Pair, Recording, Split, verify_conservation
from agbalu.codec_tts.partition import Partition, priority


@dataclass(frozen=True)
class Pairing:
    pairs: tuple[Pair, ...]
    excluded: dict[str, str]


def pair_recordings(records: Sequence[Recording], partition: Partition, seed: int) -> Pairing:
    groups: dict[tuple[Split, str], list[Recording]] = defaultdict(list)
    for record in records:
        split = partition.assignments.get(record.record_id)
        if split is not None:
            groups[split, record.speaker].append(record)
    pairs: list[Pair] = []
    excluded: dict[str, str] = {}
    for (split, speaker), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda record: priority(seed, record.record_id))
        by_script: dict[str, Recording] = {}
        for record in ordered:
            by_script.setdefault(record.text_key.casefold(), record)
        representatives = list(by_script.values())[:2]
        for target in ordered:
            reference = next(
                (
                    record
                    for record in representatives
                    if record.record_id != target.record_id
                    and record.audio_sha256 != target.audio_sha256
                    and record.text_key.casefold() != target.text_key.casefold()
                ),
                None,
            )
            if reference is None:
                excluded[target.record_id] = "no distinct same-speaker reference script/audio"
                continue
            pairs.append(
                Pair(
                    target=target.record_id,
                    reference=reference.record_id,
                    speaker=speaker,
                    split=split,
                )
            )
    verify_conservation(len(partition.assignments), len(pairs), len(excluded))
    return Pairing(tuple(pairs), excluded)
