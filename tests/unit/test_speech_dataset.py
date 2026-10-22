"""Unit tests for PyTorch dataset and dynamic audio-seconds batching."""

from __future__ import annotations

import torch

from agbalu.speech.dataset import (
    DynamicAudioBatchSampler,
    collate_speech_batch,
)
from agbalu.speech.manifest import AudioRecord


def _make_dummy_records() -> list[AudioRecord]:
    return [
        AudioRecord("1", "1.mp3", "cv", "spk_1", "Azul", "azul", duration=1.0),
        AudioRecord("2", "2.mp3", "cv", "spk_2", "Awal", "awal", duration=2.5),
        AudioRecord("3", "3.mp3", "cv", "spk_3", "Tamurt", "tamurt", duration=3.0),
        AudioRecord("4", "4.mp3", "cv", "spk_4", "Taqbaylit", "taqbaylit", duration=4.0),
    ]


class TestBatchingAndCollation:
    def test_dynamic_audio_batch_sampler(self) -> None:
        records = _make_dummy_records()
        # With max_audio_seconds=4.0:
        # sorted durations are 1.0, 2.5, 3.0, 4.0
        # batch 1: [1.0, 2.5] = 3.5s <= 4.0
        # batch 2: [3.0] = 3.0s <= 4.0
        # batch 3: [4.0] = 4.0s <= 4.0
        sampler = DynamicAudioBatchSampler(records, max_audio_seconds=4.0, shuffle=False)
        batches = list(sampler)

        assert len(batches) == 3
        assert batches[0] == [0, 1]
        assert batches[1] == [2]
        assert batches[2] == [3]

    def test_collate_speech_batch(self) -> None:
        # Batch of 2 items with differing time lengths and label lengths
        item1 = {
            "input_features": torch.randn(20, 160),
            "labels": torch.tensor([1, 2, 3]),
            "presence_labels": torch.ones(20),
        }
        item2 = {
            "input_features": torch.randn(30, 160),
            "labels": torch.tensor([4, 5]),
            "presence_labels": torch.ones(30),
        }

        collated = collate_speech_batch([item1, item2])

        assert collated["input_features"].shape == (2, 30, 160)
        assert collated["attention_mask"].shape == (2, 30)
        assert collated["presence_labels"].shape == (2, 30)
        assert collated["labels"].shape == (2, 3)

        # First item attention mask is 1 for 20 frames, 0 for 10 padded frames
        assert torch.sum(collated["attention_mask"][0]) == 20
        assert torch.sum(collated["attention_mask"][1]) == 30

        # Second item labels are padded with -100 at index 2
        assert collated["labels"][1, 2].item() == -100
