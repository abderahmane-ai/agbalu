"""Task 12.6's container-side surgery, exercised on the laptop.

Everything here runs before a GPU is attached in the real thing, which is the point: the
symbol table is generated source that the recipe imports, and a syntax error or a silent
cleaner in it is a defective model rather than a crash.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch
import yaml
from modal_app import matoub
from modal_app.matoub import (
    COMPONENTS,
    STAGE1,
    STAGE2,
    assert_base_matches_vendored,
    convert_base,
    first_stage_for,
    latest_checkpoint,
    render_config,
    work_root,
    write_symbol_table,
)

from agbalu.tts.training import MIN_OOD_PHONEMES, read_list, voice_list
from agbalu.tts.vocabulary import Vocabulary

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def kokoro() -> Vocabulary:
    return Vocabulary.load()


def load_generated(path: Path) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def seed_first_stage(root: Path, *epochs: int) -> Path:
    """A Stage 1 log directory holding the checkpoints its epochs would have written."""
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for epoch in epochs or (5,):
        (logs / f"epoch_1st_{epoch:05d}.pth").write_bytes(b"x")
    return logs


def plan_for(root: Path, stage: str, *, voice: str = "", epochs: int = 10) -> matoub.StagePlan:
    return matoub.plan_stage(root, stage=stage, voice=voice, epochs=epochs, first_stage_epoch=-1)


def rendered(root: Path, plan: matoub.StagePlan) -> dict[str, Any]:
    path = render_config(root, plan, batch=4, workers=8)
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config


class TestSymbolTable:
    def test_the_generated_module_is_importable(self, tmp_path: Path, kokoro: Vocabulary) -> None:
        path = tmp_path / "kokoro_symbols.py"
        assert write_symbol_table(path, kokoro) == 178
        assert len(load_generated(path)["symbols"]) == 178

    def test_the_kabyle_rows_are_in_the_generated_table(
        self, tmp_path: Path, kokoro: Vocabulary
    ) -> None:
        path = tmp_path / "kokoro_symbols.py"
        write_symbol_table(path, kokoro)
        dicts = load_generated(path)["dicts"]
        assert (dicts["ħ"], dicts["ʕ"], dicts["ˤ"]) == (7, 8, 26)

    def test_the_generated_cleaner_encodes_kabyle(self, tmp_path: Path, kokoro: Vocabulary) -> None:
        path = tmp_path / "kokoro_symbols.py"
        write_symbol_table(path, kokoro)
        cleaner = load_generated(path)["TextCleaner"]()
        assert cleaner("ħʕˤ") == [7, 8, 26]

    def test_the_generated_cleaner_raises_instead_of_dropping(
        self, tmp_path: Path, kokoro: Vocabulary
    ) -> None:
        """The recipe's own cleaner returns a short list here and says nothing, which is
        how three consonants leave the training target without anything reporting it."""
        path = tmp_path / "kokoro_symbols.py"
        write_symbol_table(path, kokoro)
        cleaner = load_generated(path)["TextCleaner"]()
        with pytest.raises(KeyError, match="U\\+1E93"):
            cleaner("aẓu")

    def test_it_agrees_with_the_vocabulary_it_was_rendered_from(
        self, tmp_path: Path, kokoro: Vocabulary
    ) -> None:
        path = tmp_path / "kokoro_symbols.py"
        write_symbol_table(path, kokoro)
        cleaner = load_generated(path)["TextCleaner"]()
        assert tuple(cleaner("azul ħaʕ")) == kokoro.encode("azul ħaʕ")


class TestBaseCheck:
    def test_the_vendored_table_matches_a_config_carrying_it(
        self, tmp_path: Path, kokoro: Vocabulary
    ) -> None:
        published = {s: i for s, i in kokoro.symbols.items() if s not in kokoro.assigned}
        published.pop("$", None)
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"vocab": published}), encoding="utf-8")
        assert_base_matches_vendored(path, kokoro)

    def test_a_moved_symbol_stops_the_run(self, tmp_path: Path, kokoro: Vocabulary) -> None:
        published = {s: i for s, i in kokoro.symbols.items() if s not in kokoro.assigned}
        published.pop("$", None)
        published["a"] = 177
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"vocab": published}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not carry the vocabulary"):
            assert_base_matches_vendored(path, kokoro)

    def test_a_row_kabyle_claimed_becoming_occupied_stops_the_run(
        self, tmp_path: Path, kokoro: Vocabulary
    ) -> None:
        """Row 7 filling upstream would make the fine-tune overwrite a trained embedding,
        and nothing downstream would say so — the voice would just be worse."""
        published = {s: i for s, i in kokoro.symbols.items() if s not in kokoro.assigned}
        published.pop("$", None)
        published["ʘ"] = 7
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"vocab": published}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not carry the vocabulary"):
            assert_base_matches_vendored(path, kokoro)


class TestBaseConversion:
    def test_strips_the_module_prefix_and_wraps_in_net(self, tmp_path: Path) -> None:
        source = tmp_path / "kokoro.pth"
        torch.save({name: {f"module.{name}.weight": torch.zeros(2)} for name in COMPONENTS}, source)

        counts = convert_base(source, tmp_path / "base.pth")

        written = torch.load(tmp_path / "base.pth", weights_only=False)
        assert set(written) == {"net"}
        assert set(written["net"]) == set(COMPONENTS)
        assert list(written["net"]["decoder"]) == ["decoder.weight"]
        assert counts["decoder"] == 1

    def test_a_missing_component_stops_the_run(self, tmp_path: Path) -> None:
        """StyleTTS2 loads by component name and trains from scratch what it cannot find,
        which is not an error there and would be a silently worse model here."""
        source = tmp_path / "kokoro.pth"
        torch.save({"decoder": {"module.a": torch.zeros(1)}}, source)
        with pytest.raises(RuntimeError, match="carries no"):
            convert_base(source, tmp_path / "base.pth")


class TestConfig:
    def test_it_is_loadable_yaml_with_the_keys_the_recipe_reads(self, tmp_path: Path) -> None:
        config = rendered(tmp_path, plan_for(tmp_path, STAGE1))
        for key in ("log_dir", "batch_size", "epochs_1st", "data_params", "model_params"):
            assert key in config

    def test_the_model_parameters_match_the_published_base(self, tmp_path: Path) -> None:
        """`model_params` has to equal what Kokoro declares or the weights do not load."""
        params = rendered(tmp_path, plan_for(tmp_path, STAGE1))["model_params"]
        assert params["n_token"] == 178
        assert params["hidden_dim"] == 512
        assert params["style_dim"] == 128
        assert params["multispeaker"] is True

    def test_multispeaker_is_the_same_in_both_stages(self, tmp_path: Path) -> None:
        """The flag is architectural: flipping it would leave Stage 2 unable to load what
        Stage 1 wrote."""
        seed_first_stage(tmp_path)
        stages = [
            rendered(tmp_path, plan_for(tmp_path, STAGE1))["model_params"]["multispeaker"],
            rendered(tmp_path, plan_for(tmp_path, STAGE2, voice="kab_male"))["model_params"][
                "multispeaker"
            ],
        ]
        assert stages == [True, True]

    def test_a_fresh_stage_loads_only_parameters_and_a_resumed_one_does_not(
        self, tmp_path: Path
    ) -> None:
        """`load_only_params` off is what restores the optimizer; leaving it on resumes the
        weights with a fresh moment estimate and reports the run as continued."""
        assert rendered(tmp_path, plan_for(tmp_path, STAGE1))["load_only_params"] is True
        (tmp_path / "logs" / "epoch_1st_00002.pth").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs" / "epoch_1st_00002.pth").write_bytes(b"x")
        assert rendered(tmp_path, plan_for(tmp_path, STAGE1))["load_only_params"] is False

    def test_the_ood_floor_matches_what_the_writer_guarantees(self, tmp_path: Path) -> None:
        """`min_length` is what the recipe's sampler loops against, and the OOD file is
        written to clear it. Two copies of that number would be one hang away."""
        config = rendered(tmp_path, plan_for(tmp_path, STAGE1))
        assert config["data_params"]["min_length"] == MIN_OOD_PHONEMES

    def test_the_adversarial_branch_starts_inside_a_short_run(self, tmp_path: Path) -> None:
        """`joint_epoch` fixed at 3 would never fire on a 2-epoch smoke, so the smoke would
        not exercise the stage that decides Stage 2's memory."""
        config = rendered(tmp_path, plan_for(tmp_path, STAGE1, epochs=2))
        assert config["loss_params"]["joint_epoch"] < 2

    def test_stage_two_reads_one_voice_and_stage_one_reads_the_merge(self, tmp_path: Path) -> None:
        """The defect this stage was rebuilt for: a Stage 2 pointed at the merged list
        fine-tunes on both voices, raises nothing, and writes a checkpoint indistinguishable
        from the single-speaker one it was launched to produce."""
        seed_first_stage(tmp_path)
        merged = rendered(tmp_path, plan_for(tmp_path, STAGE1))["data_params"]
        alone = rendered(tmp_path, plan_for(tmp_path, STAGE2, voice="kab_female"))["data_params"]
        assert merged["train_data"].endswith("/train_list.txt")
        assert alone["train_data"].endswith("/train_list.kab_female.txt")
        assert alone["val_data"].endswith("/val_list.kab_female.txt")

    def test_the_two_voices_never_share_a_config_file(self, tmp_path: Path) -> None:
        seed_first_stage(tmp_path)
        written = {
            render_config(tmp_path, plan_for(tmp_path, STAGE2, voice=voice), batch=4, workers=8)
            for voice in ("kab_male", "kab_female")
        }
        assert len(written) == 2


class TestRecipeImports:
    def test_a_missing_dependency_is_reported_with_its_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`pandas` is imported by `meldataset` and absent from StyleTTS2's own
        `requirements.txt`. Building the image from that file put a container on a card and
        killed it at import, five times, once per retry."""
        (tmp_path / "train_first.py").write_text("import definitely_not_installed\n")
        monkeypatch.setattr(matoub, "STYLETTS2_PATH", tmp_path)
        monkeypatch.setattr(matoub, "RECIPE_SCRIPTS", ("train_first",))

        with pytest.raises(RuntimeError, match="cannot be imported in this image"):
            matoub.assert_recipe_imports()

    def test_an_importable_script_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "train_first.py").write_text("import json\n")
        monkeypatch.setattr(matoub, "STYLETTS2_PATH", tmp_path)
        monkeypatch.setattr(matoub, "RECIPE_SCRIPTS", ("train_first",))

        assert matoub.assert_recipe_imports() == ("train_first",)

    def test_a_syntax_error_in_the_generated_table_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing the scripts also parses the symbol table this repository generates,
        which nothing else covers."""
        (tmp_path / "kokoro_symbols.py").write_text("symbols = [\n")
        (tmp_path / "train_first.py").write_text("import kokoro_symbols\n")
        monkeypatch.setattr(matoub, "STYLETTS2_PATH", tmp_path)
        monkeypatch.setattr(matoub, "RECIPE_SCRIPTS", ("train_first",))

        with pytest.raises(RuntimeError, match="cannot be imported in this image"):
            matoub.assert_recipe_imports()


class TestRunLayout:
    def test_a_smoke_never_shares_a_directory_with_a_real_run(self) -> None:
        assert work_root("restored", limit=200) != work_root("restored", limit=0)

    def test_two_caps_never_share_a_directory(self) -> None:
        """A 4,000-row probe reading a 200-row smoke's checkpoint decided it was already
        complete and measured nothing, having first overwritten that smoke's lists."""
        assert work_root("restored", limit=200) != work_root("restored", limit=4000)

    def test_the_same_cap_resolves_to_the_same_directory(self) -> None:
        """Resume has to find what the previous attempt wrote."""
        assert work_root("restored", limit=4000) == work_root("restored", limit=4000)

    def test_the_arms_are_separate(self) -> None:
        assert work_root("restored", limit=0) != work_root("raw", limit=0)

    def test_the_newest_checkpoint_is_chosen(self, tmp_path: Path) -> None:
        for epoch in (0, 1, 2):
            (tmp_path / f"epoch_1st_{epoch:05d}.pth").write_bytes(b"x")
        found = latest_checkpoint(tmp_path, "epoch_1st")
        assert found is not None
        assert found.name == "epoch_1st_00002.pth"

    def test_no_checkpoint_reads_as_none_not_an_error(self, tmp_path: Path) -> None:
        assert latest_checkpoint(tmp_path, "epoch_1st") is None

    def test_the_two_stages_do_not_read_each_others_checkpoints(self, tmp_path: Path) -> None:
        (tmp_path / "epoch_2nd_00004.pth").write_bytes(b"x")
        assert latest_checkpoint(tmp_path, "epoch_1st") is None


class TestFirstStageSelection:
    def test_minus_one_takes_the_newest_epoch_not_the_copy(self, tmp_path: Path) -> None:
        """The copy is called `first_stage.pth` and says nothing about which epoch it is;
        Stage 2 records what it continued, so the concrete file is what gets resolved."""
        (tmp_path / "first_stage.pth").write_bytes(b"x")
        for epoch in (0, 1, 2):
            (tmp_path / f"epoch_1st_{epoch:05d}.pth").write_bytes(b"x")
        chosen, index = first_stage_for(tmp_path, epoch=-1)
        assert chosen.name == "epoch_1st_00002.pth"
        assert index == 2

    def test_a_named_epoch_is_honoured(self, tmp_path: Path) -> None:
        """Stage 1 saves every epoch and records no best, and Stage 2 overfitting from
        epoch 4 is reported on a comparable corpus — the operator has to be able to pick."""
        for epoch in (0, 1, 2):
            (tmp_path / f"epoch_1st_{epoch:05d}.pth").write_bytes(b"x")
        assert first_stage_for(tmp_path, epoch=1) == (tmp_path / "epoch_1st_00001.pth", 1)

    def test_an_absent_epoch_names_what_there_is(self, tmp_path: Path) -> None:
        (tmp_path / "epoch_1st_00000.pth").write_bytes(b"x")
        with pytest.raises(RuntimeError, match=r"epoch_1st_00000\.pth"):
            first_stage_for(tmp_path, epoch=7)

    def test_a_missing_first_stage_says_to_run_stage_one(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="run stage1 first"):
            first_stage_for(tmp_path, epoch=-1)


class TestLists:
    @staticmethod
    def _corpus(root: Path) -> None:
        for voice in ("kab_male", "kab_female"):
            for split in ("train", "dev"):
                directory = root / voice / "restored"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{voice}.{split}.txt").write_text(
                    "\n".join(f"{voice}/{split}{index}.wav|azul|{voice}" for index in range(3))
                    + "\n",
                    encoding="utf-8",
                )

    def test_every_stage_gets_a_list_and_stage_two_gets_one_per_voice(
        self, tmp_path: Path, kokoro: Vocabulary, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(matoub, "CORPUS_ROOT", tmp_path / "corpus")
        self._corpus(tmp_path / "corpus")
        root = tmp_path / "work"
        root.mkdir()
        matoub.build_lists(root, "restored", ("kab_male", "kab_female"), kokoro, limit=0)
        for name in ("train_list.txt", "val_list.txt"):
            assert (root / name).is_file()
            for voice in ("kab_male", "kab_female"):
                assert (root / voice_list(name, voice)).is_file()

    def test_a_lone_voice_keeps_the_speaker_id_stage_one_conditioned_it_on(
        self, tmp_path: Path, kokoro: Vocabulary, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Renumbering a single-voice list to 0 would address a different row of the speaker
        embedding than the eighteen hours of Stage 1 that trained it."""
        monkeypatch.setattr(matoub, "CORPUS_ROOT", tmp_path / "corpus")
        self._corpus(tmp_path / "corpus")
        root = tmp_path / "work"
        root.mkdir()
        report = matoub.build_lists(root, "restored", ("kab_male", "kab_female"), kokoro, limit=0)
        ids = report["speaker_ids"]
        assert isinstance(ids, dict)
        for voice, expected in ids.items():
            rows = read_list(root / voice_list("train_list.txt", voice))
            assert rows
            assert {row.speaker for row in rows} == {str(expected)}


class TestRecipePatch:
    def test_the_patch_ships_and_carries_both_corrections(self) -> None:
        """It is applied inside the container against a pinned commit; missing from the
        repository, `resources/` mounts without it and Stage 2 fails on a card."""
        text = matoub.STAGE2_PATCH.read_text(encoding="utf-8")
        assert text.count("--- a/train_second.py") == 2
        assert "resuming_second_stage" in text
        assert "predictor_encoder" in text
        assert "+                ref = None" in text

    def test_it_binds_ref_where_the_recipe_reads_it_with_diffusion_off(self) -> None:
        """`ref` is assigned under `epoch >= diff_epoch` and read under `epoch >=
        joint_epoch`. This project runs diffusion off, so without the patch the first joint
        step raises `UnboundLocalError` — hours into a paid run, not at step 0."""
        text = matoub.STAGE2_PATCH.read_text(encoding="utf-8")
        added = text[text.index("+                ref = None") :]
        assert "if multispeaker and epoch >= diff_epoch:" in added.splitlines()[1]

    def test_a_git_failure_is_a_precondition_and_not_a_retryable_one(self) -> None:
        """`git apply` refusing says the pin moved, identically on all five retries."""
        with pytest.raises(matoub.PreconditionError, match="git"):
            matoub._git("apply", "--check", "/nonexistent.patch")


class TestPreconditions:
    """Every one of these repeats identically on a retry, so none may reach a card twice."""

    @staticmethod
    def _prepared(root: Path, **keys: object) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "prepared.json").write_text(json.dumps(keys), encoding="utf-8")
        return root

    def _check(
        self,
        root: Path,
        *,
        stage: str = STAGE2,
        voice: str = "kab_male",
        limit: int = 0,
    ) -> matoub.StagePlan:
        return matoub.check_preconditions(
            root,
            stage=stage,
            arm="restored",
            voice=voice,
            limit=limit,
            epochs=6,
            first_stage_epoch=-1,
        )

    def test_they_are_all_one_catchable_type(self) -> None:
        """`matoub_train` catches this and returns; anything else escapes and buys five
        containers for one diagnosis, which is what a stale marker already cost."""
        assert issubclass(matoub.PreconditionError, RuntimeError)

    def test_a_smoke_is_told_to_prepare_its_own_capped_directory(self, tmp_path: Path) -> None:
        """`LIMIT` routes to `<arm>-smoke-<n>`, which an uncapped prepare never wrote — the
        message has to name the cap or it sends the operator back to the wrong command."""
        with pytest.raises(matoub.PreconditionError, match="LIMIT=200"):
            self._check(tmp_path / "restored-smoke-200", limit=200)

    def test_a_marker_predating_a_check_names_the_check(self, tmp_path: Path) -> None:
        root = self._prepared(tmp_path, recipe_imports_verified=["train_first.py"])
        with pytest.raises(matoub.PreconditionError, match="stage2_patch_verified"):
            self._check(root)

    def test_stage_one_does_not_require_the_stage_two_check(self, tmp_path: Path) -> None:
        root = self._prepared(tmp_path, recipe_imports_verified=["train_first.py"])
        assert self._check(root, stage=STAGE1, voice="").stage == STAGE1

    def test_a_missing_voice_is_refused_for_stage_two(self, tmp_path: Path) -> None:
        root = self._prepared(
            tmp_path, recipe_imports_verified=["x"], stage2_patch_verified="p.patch"
        )
        with pytest.raises(matoub.PreconditionError, match="stage2 fine-tunes one voice"):
            self._check(root, voice="")

    def test_a_voice_is_refused_for_stage_one(self, tmp_path: Path) -> None:
        root = self._prepared(tmp_path, recipe_imports_verified=["x"])
        with pytest.raises(matoub.PreconditionError, match="drop VOICE"):
            self._check(root, stage=STAGE1)

    def test_a_refusal_reads_as_a_result_and_not_as_training(self) -> None:
        payload = matoub._refuse("nothing prepared", stage=STAGE2, arm="restored")
        assert payload["trained_this_run"] is False
        assert payload["refused"] == "nothing prepared"


class TestCapacity:
    def test_an_oom_is_deterministic_and_shares_the_refusal_door(self) -> None:
        """Same allocation, same failure, five times — and the recipe raises it internally,
        so this process sees only an exit code unless it reads the output for it."""
        assert issubclass(matoub.CapacityError, matoub.DeterministicError)
        assert issubclass(matoub.PreconditionError, matoub.DeterministicError)

    def test_the_markers_match_what_torch_actually_prints(self) -> None:
        printed = (
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 126.00 MiB. "
            "GPU 0 has a total capacity of 22.06 GiB of which 150.00 MiB is free."
        )
        assert any(marker in printed for marker in matoub.OOM_MARKERS)

    def test_stage_two_gets_a_shorter_segment_than_stage_one(self, tmp_path: Path) -> None:
        """Stage 1 trained six epochs at 200; Stage 2 adds the decoder, both discriminators
        and WavLM over the decoded segment, and 200 does not fit an A10G there."""
        seed_first_stage(tmp_path)
        assert plan_for(tmp_path, STAGE1).max_len == 200
        assert plan_for(tmp_path, STAGE2, voice="kab_male").max_len == 100

    def test_an_override_beats_the_stage_default(self, tmp_path: Path) -> None:
        seed_first_stage(tmp_path)
        plan = matoub.plan_stage(
            tmp_path,
            stage=STAGE2,
            voice="kab_male",
            epochs=2,
            first_stage_epoch=-1,
            max_len=50,
        )
        assert plan.max_len == 50
        assert rendered(tmp_path, plan)["max_len"] == 50

    def test_the_batch_reaches_the_config_the_recipe_reads(self, tmp_path: Path) -> None:
        """Every reference config is above ours — upstream 16, `config_ft` 8, the Kokoro
        recipe 12 — so 4 is a floor set by an OOM at a `max_len` we no longer use."""
        seed_first_stage(tmp_path)
        plan = plan_for(tmp_path, STAGE2, voice="kab_male")
        assert render_config(tmp_path, plan, batch=8, workers=8).is_file()
        config = yaml.safe_load(
            render_config(tmp_path, plan, batch=8, workers=8).read_text(encoding="utf-8")
        )
        assert config["batch_size"] == 8

    def test_zero_means_the_stage_default_and_never_zero_frames(self, tmp_path: Path) -> None:
        """A max_len of 0 would give the decoder no frames at all."""
        seed_first_stage(tmp_path)
        plan = matoub.plan_stage(
            tmp_path, stage=STAGE2, voice="kab_male", epochs=2, first_stage_epoch=-1, max_len=0
        )
        assert plan.max_len == matoub.MAX_LEN[STAGE2]


class TestStagePlan:
    def test_the_voices_write_to_separate_directories(self, tmp_path: Path) -> None:
        """One `log_dir` for both would collide on `epoch_2nd_*.pth` and on
        `first_stage.pth`: the second voice launched would resume from the first's."""
        seed_first_stage(tmp_path)
        male = plan_for(tmp_path, STAGE2, voice="kab_male")
        female = plan_for(tmp_path, STAGE2, voice="kab_female")
        assert male.logs != female.logs
        assert male.final != female.final
        assert plan_for(tmp_path, STAGE1).logs == tmp_path / "logs"

    def test_a_fresh_stage_two_does_not_take_the_resume_branch(self, tmp_path: Path) -> None:
        seed_first_stage(tmp_path)
        plan = plan_for(tmp_path, STAGE2, voice="kab_male")
        assert plan.resuming is False
        assert plan.pretrained.name == "epoch_1st_00005.pth"

    def test_a_resumed_stage_two_opens_the_branch_the_recipe_reads(self, tmp_path: Path) -> None:
        """`second_stage_load_pretrained` is the only key that makes `train_second.py` read
        `pretrained_model` at all. Written `false`, a preempted run silently reloaded Stage 1
        and restarted from epoch zero with a fresh optimizer, five retries deep."""
        seed_first_stage(tmp_path)
        logs = matoub.stage_logs(tmp_path, stage=STAGE2, voice="kab_male")
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "epoch_2nd_00003.pth").write_bytes(b"x")
        plan = plan_for(tmp_path, STAGE2, voice="kab_male")
        assert plan.resuming is True
        assert plan.pretrained.name == "epoch_2nd_00003.pth"
        config = rendered(tmp_path, plan)
        assert config["second_stage_load_pretrained"] is True
        assert config["resuming_second_stage"] is True
        assert config["load_only_params"] is False

    def test_the_completion_guard_names_the_last_epoch_not_the_first(self, tmp_path: Path) -> None:
        seed_first_stage(tmp_path)
        plan = plan_for(tmp_path, STAGE2, voice="kab_male", epochs=6)
        assert plan.final.name == "epoch_2nd_00005.pth"
        assert plan_for(tmp_path, STAGE1, epochs=6).final.name == "epoch_1st_00005.pth"

    def test_a_contradictory_from_is_refused_rather_than_ignored(self, tmp_path: Path) -> None:
        """A resumed Stage 2 never re-reads the first stage, so a changed `FROM=` would be
        accepted and have no effect on what the run actually continues."""
        seed_first_stage(tmp_path, 3, 5)
        logs = matoub.stage_logs(tmp_path, stage=STAGE2, voice="kab_male")
        logs.mkdir(parents=True, exist_ok=True)
        (logs / matoub.PLAN_MARKER).write_text(
            json.dumps({"first_stage_epoch": 5, "voice": "kab_male"}), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="FROM=5"):
            matoub.plan_stage(
                tmp_path, stage=STAGE2, voice="kab_male", epochs=6, first_stage_epoch=3
            )

    def test_a_recorded_start_survives_a_resume_that_names_nothing(self, tmp_path: Path) -> None:
        seed_first_stage(tmp_path, 3, 5)
        logs = matoub.stage_logs(tmp_path, stage=STAGE2, voice="kab_male")
        logs.mkdir(parents=True, exist_ok=True)
        (logs / matoub.PLAN_MARKER).write_text(
            json.dumps({"first_stage_epoch": 3, "voice": "kab_male"}), encoding="utf-8"
        )
        assert plan_for(tmp_path, STAGE2, voice="kab_male").first_stage_epoch == 3
