"""Unit tests for the 20-step smoke validation drill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import Wav2Vec2BertConfig

from agbalu.speech.manifest import AudioRecord, save_manifest
from agbalu.speech.model import ConformerSpeechModel
from agbalu.speech.smoke import SmokeReport, run_smoke_drill
from agbalu.speech.vocabulary import PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER


def _setup_smoke_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        WORD_DELIMITER: 2,
        "a": 3,
        "z": 4,
        "u": 5,
        "l": 6,
    }
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab) + "\n", encoding="utf-8")

    records = [
        AudioRecord(
            audio_id=f"clip_{i}",
            audio_path=f"clip_{i}.mp3",
            source="cv",
            speaker_id=f"spk_{i % 3}",
            raw_text="Azul",
            target_text="azul",
            duration=1.0,
        )
        for i in range(25)
    ]
    manifest_path = tmp_path / "train.jsonl"
    save_manifest(records, manifest_path)

    out_dir = tmp_path / "runs" / "smoke"
    return manifest_path, vocab_path, out_dir


class TestSmokeDrill:
    def test_smoke_report_structure(self) -> None:
        rep = SmokeReport(
            steps_completed=20,
            mean_step_time_seconds=0.15,
            total_audio_seconds=120.0,
            audio_throughput_xrt=40.0,
            initial_loss=5.2,
            final_loss=2.1,
            loss_reduction_pct=59.6,
            peak_vram_gb=1.8,
            vram_headroom_gb=45.2,
            resumed_cleanly=True,
        )
        data = rep.as_dict()
        assert data["steps_completed"] == 20
        assert data["resumed_cleanly"] is True
        assert float(str(data["loss_reduction_pct"])) > 50.0

    def test_run_smoke_drill_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest_path, vocab_path, out_dir = _setup_smoke_fixtures(tmp_path)

        def _tiny_model_mock(*_args: object, **_kwargs: object) -> ConformerSpeechModel:
            config = Wav2Vec2BertConfig(
                vocab_size=7,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=32,
            )
            config.feature_projection_input_dim = 160
            return ConformerSpeechModel(config, vocab_size=7)

        monkeypatch.setattr(ConformerSpeechModel, "from_pretrained", _tiny_model_mock)

        report = run_smoke_drill(
            manifest_path,
            vocab_path,
            out_dir,
            device_str="cpu",
        )

        assert report.steps_completed == 20
        assert report.resumed_cleanly is True
        assert (out_dir / "smoke_report.json").is_file()
        assert (out_dir / "smoke_checkpoint.pt").is_file()
