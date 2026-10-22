"""Unit tests for multi-source speech manifest handling and Dihya filtering."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from agbalu.speech.manifest import (
    AudioRecord,
    ingest_common_voice_tsv,
    ingest_tatoeba_tsv,
    ingest_tutlayt_jsonl,
    load_manifest,
    read_durations_tsv,
    sanitize_record,
    save_manifest,
)


class TestSanitizeRecord:
    def test_valid_record_is_created(self) -> None:
        rec = sanitize_record(
            audio_id="cv:clip_1",
            audio_path="clip_1.mp3",
            source="common_voice",
            speaker_id="spk_123",
            raw_text="Azul fell-awen ay arrac!",
            duration=3.5,
        )
        assert rec is not None
        assert rec.audio_id == "cv:clip_1"
        assert rec.target_text == "azul fell-awen ay arrac"
        assert rec.duration == 3.5

    def test_rejects_too_short(self) -> None:
        rejections: Counter[str] = Counter()
        rec = sanitize_record(
            audio_id="cv:clip_2",
            audio_path="clip_2.mp3",
            source="common_voice",
            speaker_id="spk_123",
            raw_text="Azul",
            duration=0.2,
            rejections=rejections,
        )
        assert rec is None
        assert rejections["too_short"] == 1

    def test_rejects_too_long(self) -> None:
        rejections: Counter[str] = Counter()
        rec = sanitize_record(
            audio_id="cv:clip_3",
            audio_path="clip_3.mp3",
            source="common_voice",
            speaker_id="spk_123",
            raw_text="Azul fell-awen",
            duration=35.0,
            rejections=rejections,
        )
        assert rec is None
        assert rejections["too_long"] == 1

    def test_rejects_empty_raw_text(self) -> None:
        rejections: Counter[str] = Counter()
        rec = sanitize_record(
            audio_id="cv:clip_4",
            audio_path="clip_4.mp3",
            source="common_voice",
            speaker_id="spk_123",
            raw_text="   ",
            duration=2.0,
            rejections=rejections,
        )
        assert rec is None
        assert rejections["empty_raw_text"] == 1

    def test_rejects_punctuation_only(self) -> None:
        rejections: Counter[str] = Counter()
        rec = sanitize_record(
            audio_id="cv:clip_5",
            audio_path="clip_5.mp3",
            source="common_voice",
            speaker_id="spk_123",
            raw_text="!??...,,,",
            duration=2.0,
            rejections=rejections,
        )
        assert rec is None
        assert rejections["empty_target_text"] == 1


class TestIngestTSV:
    def test_read_durations_tsv(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "clip_durations.tsv"
        tsv_file.write_text("clip_1.mp3\t3500\nclip_2.mp3\t5200\n", encoding="utf-8")
        durations = read_durations_tsv(tsv_file)
        assert durations == {"clip_1.mp3": 3.5, "clip_2.mp3": 5.2}

    def test_ingest_common_voice_tsv(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "validated.tsv"
        tsv_file.write_text(
            "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccents\tlocale\tsegment\n"
            "spk_A\tclip_1.mp3\tAɣbalu d azul\t2\t0\t\t\t\tkab\t\n"
            "spk_B\tclip_2.mp3\t\t2\t0\t\t\t\tkab\t\n",
            encoding="utf-8",
        )
        durations = {"clip_1.mp3": 4.0, "clip_2.mp3": 3.0}
        rejections: Counter[str] = Counter()
        records = list(
            ingest_common_voice_tsv(tsv_file, durations_map=durations, rejections=rejections)
        )
        assert len(records) == 1
        assert records[0].audio_id == "common_voice:clip_1"
        assert records[0].speaker_id == "spk_A"
        assert records[0].target_text == "aɣbalu d azul"
        assert rejections["empty_raw_text"] == 1

    def test_ingest_tatoeba_tsv(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "tatoeba.tsv"
        tsv_file.write_text(
            "sentence_id,user,audio_file,text,duration\n"
            "1001,massinissa,audio_1001.mp3,Iḥemmel tamurt-is,2.8\n",
            encoding="utf-8",
        )
        records = list(ingest_tatoeba_tsv(tsv_file))
        assert len(records) == 1
        assert records[0].audio_id == "tatoeba:1001"
        assert records[0].speaker_id == "tatoeba_massinissa"
        assert records[0].target_text == "iḥemmel tamurt-is"

    def test_ingest_tutlayt_jsonl(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "tutlayt.jsonl"
        items = [
            {
                "id": "t1",
                "audio": "t1.wav",
                "text": "Yal yiwen d ilelli",
                "speaker_id": "spk_1",
                "duration": 3.1,
                "language": "kab",
            },
            {
                "id": "t2",
                "audio": "t2.wav",
                "text": "Azul fell-awen",
                "speaker_id": "spk_2",
                "duration": 2.5,
                "language": "shi",  # Tashelhit sibling variety
            },
        ]
        jsonl_file.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")

        rejections: Counter[str] = Counter()
        records = list(
            ingest_tutlayt_jsonl(jsonl_file, filter_kabyle_only=True, rejections=rejections)
        )
        assert len(records) == 1
        assert records[0].audio_id == "tutlayt_kabyle:t1"
        assert rejections["non_kabyle_dialect"] == 1


class TestManifestRoundtrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        records = [
            AudioRecord(
                audio_id="cv:1",
                audio_path="1.mp3",
                source="cv",
                speaker_id="spk_1",
                raw_text="Azul",
                target_text="azul",
                duration=1.5,
            ),
            AudioRecord(
                audio_id="cv:2",
                audio_path="2.mp3",
                source="cv",
                speaker_id="spk_2",
                raw_text="Amek",
                target_text="amek",
                duration=2.0,
            ),
        ]
        manifest_file = tmp_path / "manifest.jsonl"
        report = save_manifest(records, manifest_file)
        assert report.total_records == 2
        assert report.unique_speakers == 2

        loaded = load_manifest(manifest_file)
        assert len(loaded) == 2
        assert loaded[0].audio_id == "cv:1"
        assert loaded[1].audio_id == "cv:2"
