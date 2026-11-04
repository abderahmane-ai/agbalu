"""Cache immutable codec arrays for deterministic reference-conditioned batches."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset

from agbalu.codec_tts.artifacts import (
    Pair,
    Split,
    atomic_binary,
    digest_file,
    provenance,
    read_recordings,
    run_lock,
    verify_conservation,
    write_json,
)
from agbalu.codec_tts.audio import decode_audio, embedded_bytes, raw_rows, source_files
from agbalu.codec_tts.codec import Codec
from agbalu.codec_tts.configuration import Recipe, Specification
from agbalu.codec_tts.layout import (
    CODEBOOKS,
    SPECIALS,
    SequenceLayout,
    prompt_layout,
    target_layout,
)
from agbalu.codec_tts.partition import priority
from agbalu.codec_tts.waveforms import resample_waveform
from agbalu.llm.progress import Tracker


class TextTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def get_added_vocab(self) -> dict[str, int]: ...


class TokenEntry(Specification):
    record_id: str
    file: str
    offset_frames: int = Field(ge=0)
    frames: int = Field(ge=1)
    text_ids: tuple[int, ...] = Field(min_length=1)
    audio_sha256: str
    duration_s: float = Field(gt=0)
    padding_samples: int = Field(ge=0)
    evaluation_waveform_sha256: str | None = None


def cache_codes(recipe: Recipe, codec: Codec, tokenizer: TextTokenizer) -> dict[str, object]:
    with run_lock(recipe.output):
        records = {
            record.record_id: record
            for record in read_recordings(recipe.output / "recordings.jsonl")
        }
        by_location = {(record.shard, record.row): record for record in records.values()}
        with (recipe.output / "pairs.jsonl").open() as pairs_stream:
            pairs = tuple(Pair.model_validate_json(line) for line in pairs_stream)
        protected: set[str] = set()
        for split in ("validation", "test"):
            candidates = sorted(
                (pair for pair in pairs if pair.split == split),
                key=lambda pair: priority(recipe.seed, pair.target),
            )
            for pair in candidates[: recipe.evaluation_examples]:
                protected.update((pair.target, pair.reference))
        shards = source_files(recipe)
        tracker = Tracker("codec shards", len(shards), interval_s=5)
        files: list[dict[str, object]] = []
        total = 0
        with atomic_binary(recipe.output / "tokens.jsonl") as index:
            for shard in shards:
                token_path = recipe.output / "tokens" / f"{shard.stem}.codes"
                entry_path = token_path.with_suffix(".jsonl")
                stats_path = token_path.with_suffix(".stats.json")
                identity = {
                    "config_hash": recipe.digest,
                    "source_sha256": digest_file(shard),
                    "recordings_sha256": digest_file(recipe.output / "recordings.jsonl"),
                }
                if stats_path.is_file():
                    report = json.loads(stats_path.read_bytes())
                    if (
                        any(report.get(key) != value for key, value in identity.items())
                        or report.get("token_sha256") != digest_file(token_path)
                        or report.get("index_sha256") != digest_file(entry_path)
                    ):
                        message = f"codec cache identity/checksum mismatch at {stats_path}"
                        raise ValueError(message)
                else:
                    offset = count = 0
                    with atomic_binary(token_path) as binary, atomic_binary(entry_path) as entries:
                        for row, value in raw_rows(shard):
                            recording = by_location.get((shard.name, row))
                            if recording is None:
                                continue
                            raw, text, _ = embedded_bytes(value, recipe.max_embedded_bytes)
                            waveform, rate_hz = decode_audio(raw)
                            converted = resample_waveform(waveform, rate_hz)
                            waveform_digest: str | None = None
                            if recording.record_id in protected:
                                waveform_path = (
                                    recipe.output / "evaluation" / f"{recording.record_id}.wav"
                                )
                                with atomic_binary(waveform_path) as waveform_stream:
                                    sf.write(
                                        waveform_stream,
                                        converted,
                                        24000,
                                        format="WAV",
                                        subtype="FLOAT",
                                    )
                                waveform_digest = digest_file(waveform_path)
                            codes = codec.encode(converted)
                            values = np.asarray(codes.numpy(), dtype="<u2")
                            binary.write(values.tobytes())
                            entry = TokenEntry(
                                record_id=recording.record_id,
                                file=token_path.name,
                                offset_frames=offset,
                                frames=codes.shape[0],
                                text_ids=tuple(tokenizer.encode(text, add_special_tokens=False)),
                                audio_sha256=recording.audio_sha256,
                                duration_s=recording.duration_s,
                                padding_samples=max(0, 24000 - converted.size),
                                evaluation_waveform_sha256=waveform_digest,
                            )
                            entries.write((entry.model_dump_json() + "\n").encode())
                            offset += codes.shape[0]
                            count += 1
                    report = {
                        **provenance(recipe.digest),
                        **identity,
                        "records": count,
                        "frames": offset,
                        "token_sha256": digest_file(token_path),
                        "index_sha256": digest_file(entry_path),
                    }
                    write_json(stats_path, report)
                with entry_path.open("rb") as stream:
                    while block := stream.read(1 << 20):
                        index.write(block)
                total += int(report["records"])
                files.append({"file": token_path.name, "sha256": report["token_sha256"]})
                tracker.advance()
        verify_conservation(len(records), total, 0)
        report = {
            **provenance(recipe.digest),
            "records": total,
            "files": files,
            "index_sha256": digest_file(recipe.output / "tokens.jsonl"),
        }
        write_json(recipe.output / "tokens.stats.json", report)
        return report


class SpeechDataset(Dataset[SequenceLayout]):
    def __init__(self, recipe: Recipe, specials: Mapping[str, int], split: Split) -> None:
        self.recipe = recipe
        self.specials = dict(specials)
        if set(SPECIALS) - self.specials.keys():
            message = "tokenizer is missing native TTS specials"
            raise ValueError(message)
        report = json.loads((recipe.output / "tokens.stats.json").read_bytes())
        if report.get("config_hash") != recipe.digest or report.get("index_sha256") != digest_file(
            recipe.output / "tokens.jsonl"
        ):
            message = "codec index identity/checksum mismatch"
            raise ValueError(message)
        for item in report["files"]:
            path = recipe.output / "tokens" / item["file"]
            if Path(item["file"]).name != item["file"] or digest_file(path) != item["sha256"]:
                message = f"codec binary identity/checksum mismatch: {path}"
                raise ValueError(message)
        with (recipe.output / "tokens.jsonl").open() as stream:
            self.entries = {
                entry.record_id: entry
                for line in stream
                if (entry := TokenEntry.model_validate_json(line))
            }
        with (recipe.output / "pairs.jsonl").open() as stream:
            candidates = tuple(
                pair for line in stream if (pair := Pair.model_validate_json(line)).split == split
            )
        accepted: list[Pair] = []
        excluded: list[dict[str, object]] = []
        for pair in candidates:
            target, reference = self.entries[pair.target], self.entries[pair.reference]
            positions = mixed_positions(target, reference)
            if positions > recipe.max_mixed_positions:
                excluded.append(
                    {
                        "target": pair.target,
                        "reference": pair.reference,
                        "positions": positions,
                        "reason": "mixed context exceeds maximum",
                    }
                )
            else:
                accepted.append(pair)
        self.pairs = tuple(accepted)
        verify_conservation(len(candidates), len(self.pairs), len(excluded))
        write_json(recipe.output / f"{split}-context-rejects.json", excluded)
        write_json(
            recipe.output / f"{split}-context.stats.json",
            {
                **provenance(recipe.digest),
                "input_pairs": len(candidates),
                "accepted_pairs": len(self.pairs),
                "rejected_pairs": len(excluded),
            },
        )
        if not self.pairs:
            message = f"empty codec dataset split={split}"
            raise ValueError(message)
        self.identity = digest_file(recipe.output / "tokens.stats.json") + digest_file(
            recipe.output / "pairs.jsonl"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> SequenceLayout:
        return self.layout(index)

    def positions(self, index: int) -> int:
        pair = self.pairs[index]
        return mixed_positions(self.entries[pair.target], self.entries[pair.reference])

    def codes(self, entry: TokenEntry) -> Tensor:
        path = self.recipe.output / "tokens" / entry.file
        if Path(entry.file).name != entry.file:
            message = f"invalid codec file name={entry.file!r} for {entry.record_id}"
            raise ValueError(message)
        byte_count = entry.frames * CODEBOOKS * 2
        with path.open("rb") as stream:
            stream.seek(entry.offset_frames * CODEBOOKS * 2)
            raw = stream.read(byte_count)
        if len(raw) != byte_count:
            message = (
                f"short codec read {path}:{entry.offset_frames}: {len(raw)}/{byte_count} bytes"
            )
            raise ValueError(message)
        values = np.frombuffer(raw, dtype="<u2").astype(np.int64).reshape(entry.frames, CODEBOOKS)
        return torch.from_numpy(values)

    def layout(self, index: int) -> SequenceLayout:
        pair = self.pairs[index]
        target, reference = self.entries[pair.target], self.entries[pair.reference]
        prompt = prompt_layout(
            self.specials, target.text_ids, self.codes(reference), reference.text_ids
        )
        return target_layout(prompt, self.codes(target), self.recipe.max_mixed_positions)

    def windows(self) -> Iterator[tuple[int, ...]]:
        order = sorted(
            range(len(self.pairs)),
            key=lambda index: priority(self.recipe.seed, self.pairs[index].target),
        )
        for start in range(0, len(order), self.recipe.update_examples):
            window = order[start : start + self.recipe.update_examples]
            yield tuple(
                sorted(
                    window,
                    key=self.positions,
                )
            )


def collate_layouts(
    layouts: Sequence[SequenceLayout], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not layouts:
        message = "cannot collate an empty codec microbatch"
        raise ValueError(message)
    longest = max(layout.positions for layout in layouts)
    text = torch.zeros((len(layouts), longest), dtype=torch.long)
    codes = torch.zeros((len(layouts), longest, CODEBOOKS), dtype=torch.long)
    audio = torch.zeros_like(text, dtype=torch.bool)
    valid = torch.zeros_like(text, dtype=torch.bool)
    labels = torch.full_like(codes, -100)
    for row, layout in enumerate(layouts):
        size = layout.positions
        text[row, :size], codes[row, :size] = layout.text_ids, layout.audio_codes
        audio[row, :size], labels[row, :size] = layout.audio_mask, layout.labels
        valid[row, :size] = True
    values: tuple[Tensor, ...] = (text, codes, audio, valid, labels)
    if device.type == "cuda":
        values = tuple(value.pin_memory() for value in values)
    moved = tuple(value.to(device, non_blocking=device.type == "cuda") for value in values)
    return moved[0], moved[1], moved[2], moved[3], moved[4]


def collate_cpu(layouts: Sequence[SequenceLayout]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return collate_layouts(layouts, torch.device("cpu"))


def mixed_positions(target: TokenEntry, reference: TokenEntry) -> int:
    return (
        5
        + len(target.text_ids)
        + len(reference.text_ids)
        + target.frames
        + reference.frames
        + 2 * (CODEBOOKS - 1)
    )
