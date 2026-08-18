"""The pre-flight for the baseline run (task 11.2).

The evaluation sets and FLORES+ are uploaded by different targets, so either can be the one
that was forgotten — and the failure would otherwise arrive after a 10.25 GB checkpoint has
been downloaded, five times, because the function carries retries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from modal_app import bench, llm
from modal_app.common import LLM_VOLUMES, MODELS_PATH

from agbalu.bench.mt import DIRECTIONS
from agbalu.llm.corpus import LANGUAGE_TAG


def volume(tmp_path: Path, *, languages: list[str], sets: list[str]) -> Path:
    flores = tmp_path / "bench" / "flores" / "devtest"
    flores.mkdir(parents=True)
    for language in languages:
        (flores / f"{language}.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "bench" / "flores-plus-kab_Latn-corrected.jsonl").write_text("", encoding="utf-8")

    held = tmp_path / "llm"
    held.mkdir(parents=True)
    for short in sets:
        (held / f"heldout-{short}.jsonl").write_text("{}\n", encoding="utf-8")
    return tmp_path


def patch(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(llm, "DATA_PATH", str(root))
    monkeypatch.setattr(bench, "DATA_PATH", str(root))


class TestVolumes:
    def test_the_model_cache_is_mounted(self) -> None:
        """Without it every run refetches 10.25 GB of weights."""
        assert str(MODELS_PATH) in LLM_VOLUMES


class TestCheckInputs:
    def test_a_complete_volume_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch(
            monkeypatch,
            volume(tmp_path, languages=["kab_Latn", "eng_Latn"], sets=["kab", "eng", "fra"]),
        )
        llm.check_inputs(["kab-eng", "eng-kab"])

    def test_a_missing_evaluation_set_names_both_targets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch(monkeypatch, volume(tmp_path, languages=["kab_Latn", "eng_Latn"], sets=["kab"]))
        with pytest.raises(FileNotFoundError, match="llm-holdout"):
            llm.check_inputs(["kab-eng"])

    def test_a_missing_flores_split_is_still_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch(monkeypatch, volume(tmp_path, languages=["kab_Latn"], sets=["kab", "eng", "fra"]))
        with pytest.raises(FileNotFoundError, match="modal-upload"):
            llm.check_inputs(["eng-kab"])

    def test_an_unknown_direction_is_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch(
            monkeypatch,
            volume(tmp_path, languages=["kab_Latn", "eng_Latn"], sets=["kab", "eng", "fra"]),
        )
        with pytest.raises(ValueError, match="unknown direction"):
            llm.check_inputs(["kab-zgh"])


class TestCorrectedDevtest:
    def test_only_devtest_is_read_and_it_is_ordered_by_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FLORES+ ids restart at 0 per split, so `id` alone interleaves the two."""
        (tmp_path / "bench").mkdir()
        rows = [
            {"split": "devtest", "id": 2, "text": "b"},
            {"split": "dev", "id": 0, "text": "dev"},
            {"split": "devtest", "id": 0, "text": "a"},
        ]
        (tmp_path / "bench" / "flores-plus-kab_Latn-corrected.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        monkeypatch.setattr(llm, "DATA_PATH", str(tmp_path))
        assert llm.corrected_devtest() == ["a", "b"]


class TestBaselineDirections:
    def test_the_default_scope_is_what_the_fine_tune_was_scored_on(self) -> None:
        assert llm.BASELINE_DIRECTIONS == ("kab-eng", "eng-kab", "kab-fra", "fra-kab")

    def test_every_default_direction_is_one_the_harness_knows(self) -> None:
        assert set(llm.BASELINE_DIRECTIONS) <= set(DIRECTIONS)

    def test_every_held_out_language_has_a_tag(self) -> None:
        assert dict(llm.HELDOUT_LANGUAGES) == LANGUAGE_TAG
