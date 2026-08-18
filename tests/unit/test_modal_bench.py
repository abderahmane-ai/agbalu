"""The `--weights` guard, the mount it depends on, and the two filename collisions.

`BENCH_VOLUMES` carried no checkpoint volume — correct while the function only scored
public repos, wrong the moment `--weights` was added to score a fine-tune from one. The
path then did not resolve, and `from_pretrained` reports a local path it cannot find as a
Hub repo id, so the mount defect surfaced from inside `huggingface_hub` naming neither the
path nor the volume.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from modal_app import bench
from modal_app.bench import (
    HUB_WEIGHTS,
    check_weights,
    requested_directions,
    resolve_weights,
    results_stem,
)
from modal_app.common import BENCH_VOLUMES, CHECKPOINT_PATH

from agbalu.bench.mt import DIRECTIONS


class TestBenchVolumes:
    def test_the_checkpoint_volume_is_mounted(self) -> None:
        """Without it `--weights` cannot name anything a fine-tune wrote."""
        assert str(CHECKPOINT_PATH) in BENCH_VOLUMES


class TestCheckWeights:
    def test_a_saved_checkpoint_passes(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        check_weights(tmp_path)

    def test_a_missing_directory_names_the_mount(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=str(CHECKPOINT_PATH)):
            check_weights(tmp_path / "never-written")

    def test_a_file_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "final"
        path.write_text("not a directory", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="not a directory"):
            check_weights(path)

    def test_a_directory_without_a_config_is_refused(self, tmp_path: Path) -> None:
        """A run interrupted mid-save leaves the directory and not the config."""
        with pytest.raises(FileNotFoundError, match=re.escape("config.json")):
            check_weights(tmp_path)


class TestResolveWeights:
    """Volume first, Hub second — and only when nothing was asked for by name."""

    def test_the_volume_checkpoint_wins_when_it_is_there(self, tmp_path: Path) -> None:
        default = tmp_path / "final"
        default.mkdir()
        (default / "config.json").write_text("{}", encoding="utf-8")
        assert resolve_weights("", default) == default

    def test_an_explicit_path_is_used_as_given(self, tmp_path: Path) -> None:
        asked = tmp_path / "other-run"
        asked.mkdir()
        (asked / "config.json").write_text("{}", encoding="utf-8")
        assert resolve_weights(str(asked), tmp_path / "unused") == asked

    def test_an_explicit_path_that_is_missing_raises_rather_than_falling_back(
        self, tmp_path: Path
    ) -> None:
        """The footgun this guards: a mistyped `--weights` silently translating the whole
        document with a different model, and the output looking perfectly fine."""
        with pytest.raises(FileNotFoundError):
            resolve_weights(str(tmp_path / "typo"), tmp_path / "default")

    def test_an_explicit_path_without_a_config_raises(self, tmp_path: Path) -> None:
        asked = tmp_path / "half-saved"
        asked.mkdir()
        with pytest.raises(FileNotFoundError, match=re.escape("config.json")):
            resolve_weights(str(asked), tmp_path / "default")

    def test_a_default_that_holds_no_config_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A volume directory can exist and be empty — a fresh account, or a save that never
        finished. That is a fallback case, not a failure."""
        default = tmp_path / "final"
        default.mkdir()
        fetched = tmp_path / "from-hub"
        fetched.mkdir()
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda *_a, **_k: str(fetched), raising=False
        )
        assert resolve_weights("", default) == fetched

    def test_a_missing_default_fetches_the_published_weights(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []
        fetched = tmp_path / "from-hub"
        fetched.mkdir()

        def fake(repo_id: str, *_args: object, **_kwargs: object) -> str:
            asked.append(repo_id)
            return str(fetched)

        monkeypatch.setattr("huggingface_hub.snapshot_download", fake, raising=False)
        assert resolve_weights("", tmp_path / "never-written") == fetched
        assert asked == [HUB_WEIGHTS]

    def test_the_hub_repo_is_the_published_release(self) -> None:
        """Named against the release, not a personal fork: the card's own snippet and every
        published chrF++ in `docs/cards/amrouche-1.3b.md` refer to this repo."""
        assert HUB_WEIGHTS == "agbalu/Amrouche-1.3B"


class TestRequestedDirections:
    def test_none_means_every_direction(self) -> None:
        assert requested_directions(None, DIRECTIONS) == frozenset(DIRECTIONS)

    def test_a_subset_is_kept_as_asked(self) -> None:
        assert requested_directions(["arb-kab", "deu-kab"], DIRECTIONS) == {"arb-kab", "deu-kab"}

    def test_a_repeat_collapses(self) -> None:
        assert requested_directions(["arb-kab", "arb-kab"], DIRECTIONS) == {"arb-kab"}

    @pytest.mark.parametrize("value", ["kab-arb", "arb_kab", "ARB-KAB", "arb-kab ", ""])
    def test_an_unknown_direction_is_named_rather_than_silently_dropped(self, value: str) -> None:
        """An empty intersection reads downstream as a finished run that scored nothing."""
        with pytest.raises(ValueError, match="unknown direction"):
            requested_directions([value], DIRECTIONS)

    def test_an_empty_request_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no directions requested"):
            requested_directions([], DIRECTIONS)


class TestResultsStem:
    def test_the_full_baseline_sweep_keeps_its_published_name(self) -> None:
        assert results_stem(None, frozenset(DIRECTIONS), DIRECTIONS) == "mt-baselines"

    def test_a_fine_tune_is_named_by_its_run_and_not_by_final(self) -> None:
        """`Path(weights).name` is `final` for every checkpoint, so two fine-tunes scored
        the same way wrote one file and the second erased the first."""
        assert (
            results_stem("/checkpoints/mt/nllb-kab-v2/final", frozenset(DIRECTIONS), DIRECTIONS)
            == "mt-finetuned-nllb-kab-v2-final"
        )

    def test_two_runs_do_not_collide(self) -> None:
        scope = frozenset(DIRECTIONS)
        first = results_stem("/checkpoints/mt/nllb-kab-v1/final", scope, DIRECTIONS)
        second = results_stem("/checkpoints/mt/nllb-kab-v2/final", scope, DIRECTIONS)
        assert first != second

    def test_an_intermediate_checkpoint_does_not_collide_with_final(self) -> None:
        scope = frozenset(DIRECTIONS)
        assert results_stem("/checkpoints/mt/nllb-kab-v1/checkpoint-450", scope, DIRECTIONS) != (
            results_stem("/checkpoints/mt/nllb-kab-v1/final", scope, DIRECTIONS)
        )

    def test_a_partial_sweep_cannot_overwrite_the_full_one(self) -> None:
        partial = results_stem(None, frozenset({"arb-kab"}), DIRECTIONS)
        assert partial == "mt-baselines-arb-kab"
        assert partial != results_stem(None, frozenset(DIRECTIONS), DIRECTIONS)

    def test_the_scope_suffix_is_order_independent(self) -> None:
        """The set comes from a comma-separated flag, so two operators typing the same
        directions in a different order must write one file, not two."""
        assert results_stem(None, frozenset({"deu-kab", "arb-kab"}), DIRECTIONS) == results_stem(
            None, frozenset({"arb-kab", "deu-kab"}), DIRECTIONS
        )


class TestCheckCorpora:
    """The pre-flight that turns twelve wasted GPU minutes into a stat call."""

    def _volume(self, tmp_path: Path, languages: list[str]) -> Path:
        root = tmp_path / "bench" / "flores" / "devtest"
        root.mkdir(parents=True)
        for language in languages:
            (root / f"{language}.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "bench" / "flores-plus-kab_Latn-corrected.jsonl").write_text(
            "", encoding="utf-8"
        )
        return tmp_path

    def test_a_complete_volume_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            bench, "DATA_PATH", str(self._volume(tmp_path, ["kab_Latn", "eng_Latn"]))
        )
        bench.check_corpora(["kab-eng", "eng-kab"], corrected=True)

    def test_a_missing_language_is_named_with_the_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            bench, "DATA_PATH", str(self._volume(tmp_path, ["kab_Latn", "eng_Latn"]))
        )
        with pytest.raises(FileNotFoundError, match="modal-upload"):
            bench.check_corpora(["arb-kab"], corrected=True)

    def test_the_published_reference_needs_the_target_side_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--published` reads FLORES+ `kab_Latn` instead of the corrected file, so the
        two conditions do not need the same files."""
        monkeypatch.setattr(bench, "DATA_PATH", str(self._volume(tmp_path, ["eng_Latn"])))
        with pytest.raises(FileNotFoundError, match="kab_Latn"):
            bench.check_corpora(["eng-kab"], corrected=False)
