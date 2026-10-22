"""Unit tests for speaker-stratified disjoint partitioning."""

from __future__ import annotations

from agbalu.speech.manifest import AudioRecord
from agbalu.speech.partition import (
    SplitDataset,
    partition_records,
    verify_speaker_isolation,
)


def _make_dummy_records(num_speakers: int = 10, clips_per_speaker: int = 5) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    for s in range(num_speakers):
        spk_id = f"spk_{s:03d}"
        records.extend(
            AudioRecord(
                audio_id=f"rec_{s}_{c}",
                audio_path=f"clip_{s}_{c}.mp3",
                source="common_voice",
                speaker_id=spk_id,
                raw_text=f"Awal wis {s} {c}",
                target_text=f"awal wis {s} {c}",
                duration=2.0,
            )
            for c in range(clips_per_speaker)
        )
    return records


class TestPartition:
    def test_speaker_isolation_is_maintained(self) -> None:
        records = _make_dummy_records(num_speakers=20, clips_per_speaker=4)
        splits = partition_records(records, dev_ratio=0.10, test_ratio=0.10, seed=42)

        train_spk = {r.speaker_id for r in splits.train}
        dev_spk = {r.speaker_id for r in splits.dev}
        test_spk = {r.speaker_id for r in splits.test}

        assert len(train_spk & dev_spk) == 0
        assert len(train_spk & test_spk) == 0
        assert len(dev_spk & test_spk) == 0
        assert verify_speaker_isolation(splits) is True

    def test_report_summarizes_splits(self) -> None:
        records = _make_dummy_records(num_speakers=10, clips_per_speaker=2)
        splits = partition_records(records, dev_ratio=0.10, test_ratio=0.10, seed=42)
        report = splits.report()

        assert len(report.splits) == 4
        assert report.speaker_overlaps["train_dev"] == 0
        assert report.speaker_overlaps["train_test"] == 0
        assert report.speaker_overlaps["dev_test"] == 0

    def test_phrase_disjoint_subset(self) -> None:
        # Create speaker records where some phrases overlap and one is unique to test speaker
        records = [
            AudioRecord("1", "1.mp3", "cv", "spk_A", "Azul", "azul", 2.0),
            AudioRecord("2", "2.mp3", "cv", "spk_B", "Azul", "azul", 2.0),  # Shared phrase
            AudioRecord("3", "3.mp3", "cv", "spk_C", "Tamurt", "tamurt", 2.0),  # Unique phrase
        ]
        # Put spk_A into train, spk_B into dev, spk_C into test manually
        splits = SplitDataset(
            train=[records[0]],
            dev=[records[1]],
            test=[records[2]],
            test_phrase_disjoint=[records[2]],
        )
        assert len(splits.test_phrase_disjoint) == 1
        assert splits.test_phrase_disjoint[0].target_text == "tamurt"

    def test_mega_speakers_locked_to_train(self) -> None:
        # 1 mega-speaker with 4.0 hours (14,400s), 20 small speakers with 0.1 hours (360s) each
        mega_records = [
            AudioRecord(
                audio_id=f"mega_{i}",
                audio_path=f"mega_{i}.mp3",
                source="common_voice",
                speaker_id="spk_mega",
                raw_text="Awal n umsebrid",
                target_text="awal n umsebrid",
                duration=720.0,
            )
            for i in range(20)  # 20 * 720 = 14,400s = 4.0h
        ]
        small_records = [
            AudioRecord(
                audio_id=f"small_{s}",
                audio_path=f"small_{s}.mp3",
                source="common_voice",
                speaker_id=f"spk_small_{s:02d}",
                raw_text=f"Awal wis {s}",
                target_text=f"awal wis {s}",
                duration=360.0,  # 0.1h = 360s
            )
            for s in range(20)
        ]

        all_records = mega_records + small_records
        splits = partition_records(
            all_records,
            dev_ratio=0.05,
            test_ratio=0.05,
            seed=42,
            max_eval_speaker_hours=3.0,
        )

        train_spks = {r.speaker_id for r in splits.train}
        dev_spks = {r.speaker_id for r in splits.dev}
        test_spks = {r.speaker_id for r in splits.test}

        # Mega speaker MUST be locked into train
        assert "spk_mega" in train_spks
        assert "spk_mega" not in dev_spks
        assert "spk_mega" not in test_spks

        # Dev and test must have distinct small speakers
        assert len(dev_spks) >= 1
        assert len(test_spks) >= 1
        assert verify_speaker_isolation(splits) is True
