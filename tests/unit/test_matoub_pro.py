"""The Pro training regime, asserted against the defects it exists to remove.

Every number here is one v1 got wrong (`docs/matoub_pro_design.md` §1). A config that reads
plausibly and trains the wrong thing is the failure this file is for: `lambda_diff: 0.0` and
`slmadv_params.min_len` above the corpus's own lengths both produce a healthy loss curve, a
finished checkpoint, and a model that cannot do what it was built for.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from modal_app.matoub import STAGE1, STAGE2, PreconditionError
from modal_app.matoub_pro import (
    BATCH,
    BATCH_PERCENTAGE,
    MAX_LEN,
    MIN_OOD_LINES,
    MIN_OOD_PHONEMES,
    PATCHED_FILES,
    PRO_PATCH,
    SEED,
    SLM_HIDDEN,
    SLM_LAYERS,
    SLM_MODEL,
    SLMADV_MIN_LEN,
    STAGE2_PATCH,
    SpeakerAudio,
    StagePlan,
    admissible,
    render_config,
    select,
    select_ood,
)

from agbalu.tts.corpus_pro import Source

FRAME_MS: Final = 12.5
"""One `max_len` frame is `hop_length / sr` = 300 / 24000 seconds."""


def plan(root: Path, stage: str, *, epochs: int = 10) -> StagePlan:
    return StagePlan(
        stage=stage,
        root=root,
        logs=root / "logs",
        train_list=root / "train.txt",
        val_list=root / "dev.txt",
        pretrained=root / "kokoro_base.pth",
        resuming=False,
        epochs=epochs,
        max_len=MAX_LEN[stage],
        batch=BATCH[stage],
        final=root / "final.pth",
    )


Aggregate = dict[str, float | int | None]
"""The keywords `acoustics` varies, one per measurement a speaker can be refused on."""


def acoustics(
    *,
    floor: float | int | None = 36.0,
    share: float | int | None = 0.69,
    clips: float | int | None = 1000,
    zeroed: float | int | None = 18,
) -> dict[str, object]:
    """A speaker aggregate shaped like `agbalu.tts.acoustics.summarise` writes it."""
    fields: dict[str, object] = {}
    if floor is not None:
        fields["speech_to_floor_db"] = {"measured": clips, "median": floor}
    if share is not None:
        fields["speech_frame_share"] = {"measured": clips, "median": share}
    return {"clips": clips, "clips_with_zeroed_silence": zeroed, "fields": fields}


def speaker(name: str, *, seconds: float = 3600.0, **built: float | int | None) -> SpeakerAudio:
    """A speaker worth an hour, so the hours budget is the thing under test."""
    return SpeakerAudio(
        speaker=name,
        sources=(
            Source(
                clip=f"common_voice_kab_{1}.mp3",
                text="Azul.",
                ipa="æzul.",
                order=1,
                duration_s=seconds,
                speech_start_s=0.5,
                speech_end_s=seconds - 0.3,
                speech_dbfs=-19.0,
                peak=0.5,
                cer=0.0,
            ),
        ),
        acoustics=acoustics(**built),
    )


class TestTheSchedule:
    @pytest.mark.parametrize("epochs", [2, 3, 4, 6, 10, 20, 50])
    def test_every_gate_lands_inside_the_run_and_in_order(
        self, tmp_path: Path, epochs: int
    ) -> None:
        """v1 wrote `diff_epoch: 999` on a six-epoch run, so the sampler's gate sat 993 epochs
        past the end of training."""
        built = plan(tmp_path, STAGE2, epochs=epochs)
        assert 1 <= built.diff_epoch <= epochs
        assert 1 <= built.tma_epoch <= epochs
        assert 1 <= built.joint_epoch <= epochs
        assert built.diff_epoch <= built.tma_epoch <= built.joint_epoch

    def test_half_of_stage_two_is_joint_training(self, tmp_path: Path) -> None:
        """Upstream runs `joint_epoch: 50` of 100. v1 reached it in two of four epochs."""
        built = plan(tmp_path, STAGE2, epochs=10)
        assert built.joint_epoch == 5

    def test_a_two_epoch_smoke_still_reaches_the_joint_step(self, tmp_path: Path) -> None:
        """The smoke is worth nothing if it stops before the branch where the waveform
        decoder, MPD, MSD and the SLM switch on together."""
        assert plan(tmp_path, STAGE2, epochs=2).joint_epoch == 1


class TestTheCard:
    def test_stage_one_keeps_the_point_v1_measured_to_fit(self) -> None:
        """v1 ran six Stage 1 epochs at (200, 4) on an A10G."""
        assert (MAX_LEN[STAGE1], BATCH[STAGE1]) == (200, 4)

    def test_stage_two_keeps_the_product_that_ran_and_doubles_the_window(self) -> None:
        """(200, 4) asked 21.41 GiB of 22.06 at the first joint step and died; (100, 4) ran.
        The joint step is dominated by waveform length times batch, so Pro takes the same 400
        units at twice the F0 and energy supervision window."""
        assert MAX_LEN[STAGE2] * BATCH[STAGE2] == 100 * 4
        assert MAX_LEN[STAGE2] > 100

    def test_the_decoder_segment_is_reported_in_seconds_it_can_be_read_in(self) -> None:
        assert MAX_LEN[STAGE2] * FRAME_MS / 1000.0 == pytest.approx(2.5)

    @pytest.mark.parametrize("stage", [STAGE1, STAGE2])
    def test_the_adversary_can_always_collect_two_samples(self, stage: str) -> None:
        """`slmadv` breaks out at `batch_percentage * len(waves)` and then returns `None` if
        it collected one or fewer — which `train_second.py` answers with a bare `continue`.
        Below this floor the joint epochs run with no adversary and a clean log."""
        assert BATCH_PERCENTAGE[stage] * BATCH[stage] >= 2


def section(config: Mapping[str, object], *keys: str) -> dict[str, object]:
    """One nested block of the rendered config. The recipe reads YAML, so the assertions
    navigate it rather than trusting a shape."""
    node: object = config
    for key in keys:
        assert isinstance(node, dict), f"{key} is not under a mapping"
        node = node[key]
    assert isinstance(node, dict)
    return node


class TestTheConfig:
    @pytest.fixture
    def config(self, tmp_path: Path) -> dict[str, object]:
        written = render_config(plan(tmp_path, STAGE2), workers=8)
        loaded = yaml.safe_load(written.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return dict(loaded)

    def test_style_diffusion_is_trained(self, config: dict[str, object]) -> None:
        """The mechanism that turns one style vector per utterance into a distribution, and
        the only source of `alpha`, `beta` and `diffusion_steps`. v1 set both to 0.0."""
        losses = section(config, "loss_params")
        assert losses["lambda_diff"] == 1.0
        assert losses["lambda_sty"] == 1.0

    def test_the_slm_has_heard_kabyle_and_keeps_the_discriminator_shape(
        self, config: dict[str, object]
    ) -> None:
        slm = section(config, "model_params", "slm")
        assert slm["model"] == SLM_MODEL
        assert slm["hidden"] == SLM_HIDDEN
        assert slm["nlayers"] == SLM_LAYERS + 1

    def test_the_adversarial_branch_asks_for_lengths_the_joined_corpus_has(
        self, config: dict[str, object]
    ) -> None:
        """`mel_input_length > min_len`, so 300 frames asks for utterances past 3.75 s. v1's
        median was 2.87 s, which is why it had to be lowered to 100."""
        assert section(config, "slmadv_params")["min_len"] == SLMADV_MIN_LEN
        assert SLMADV_MIN_LEN * FRAME_MS / 1000.0 < 5.0

    def test_the_ood_floor_and_the_adversary_agree(self, config: dict[str, object]) -> None:
        assert section(config, "data_params")["min_length"] == MIN_OOD_PHONEMES

    def test_the_weights_are_seeded(self, config: dict[str, object]) -> None:
        """`meldataset` seeds numpy and `random` at import and nothing seeds torch, so two
        runs of one arm differ by more than the arm does."""
        assert config["seed"] == SEED

    def test_the_reconstruction_mel_is_left_where_the_aligner_was_pretrained(
        self, config: dict[str, object]
    ) -> None:
        """`meldataset` ignores `spect_params` entirely; the value is carried so the config
        records the filterbank the style encoder and the pitch extractor actually read."""
        assert section(config, "preprocess_params", "spect_params")["fmax"] == 8000

    def test_stage_one_and_stage_two_differ_only_where_they_must(self, tmp_path: Path) -> None:
        one, two = tmp_path / "a", tmp_path / "b"
        one.mkdir()
        two.mkdir()
        first = dict(yaml.safe_load(render_config(plan(one, STAGE1), workers=8).read_text("utf-8")))
        second = dict(
            yaml.safe_load(render_config(plan(two, STAGE2), workers=8).read_text("utf-8"))
        )
        assert section(first, "loss_params") == section(second, "loss_params")
        assert section(first, "model_params") == section(second, "model_params")
        assert (
            section(first, "slmadv_params")["batch_percentage"]
            != section(second, "slmadv_params")["batch_percentage"]
        )


class TestTheSpeakerFilter:
    def test_it_admits_the_voice_this_project_shipped(self) -> None:
        """12.3 measured rank 2 at 36.05 dB speech-to-floor, a 0.687 speech share and 1.8% of
        its clips gated to digital zero."""
        assert admissible(acoustics(), hours=11.2) is None

    def test_it_refuses_the_voice_12_3_recommended_against(self) -> None:
        """Rank 1 has its silence gated to digital zero in 59.7% of its clips, which takes the
        low-energy onsets where Kabyle's emphatics and spirantised stops live."""
        assert admissible(acoustics(zeroed=597), hours=9.2) == "silence-gated-to-digital-zero"

    @pytest.mark.parametrize(
        ("built", "hours", "expected"),
        [
            ({}, 0.1, "under-the-hours-floor"),
            ({"floor": 12.0}, 5.0, "noisy"),
            ({"floor": None}, 5.0, "noisy"),
            ({"share": 0.2}, 5.0, "mostly-silence"),
            ({"share": None}, 5.0, "mostly-silence"),
        ],
    )
    def test_it_names_the_reason_rather_than_returning_a_bare_false(
        self, built: Aggregate, hours: float, expected: str
    ) -> None:
        assert admissible(acoustics(**built), hours=hours) == expected

    def test_the_voice_is_taken_whether_or_not_the_budget_has_room(self) -> None:
        """A Stage 1 that never heard the speaker Stage 2 fine-tunes is the wrong backbone,
        so the voice is not subject to the hours budget the rest of the pool is."""
        rows = [speaker("first"), speaker("second"), speaker("voice")]
        taken, refused = select(rows, hours=1.5, keep="voice")
        assert "voice" in [item.speaker for item in taken]
        assert refused.get("over-the-hours-budget") == 1

    def test_a_refused_speaker_is_counted_by_reason(self) -> None:
        taken, refused = select([speaker("n", floor=5.0)], hours=100.0, keep="voice")
        assert not taken
        assert refused == {"noisy": 1}


class TestTheOodFile:
    def test_it_keeps_only_lines_the_adversary_can_use(self, tmp_path: Path) -> None:
        source = tmp_path / "ood.txt"
        source.write_text(
            "\n".join(["a" * (MIN_OOD_PHONEMES + 1)] * MIN_OOD_LINES + ["short"]) + "\n",
            encoding="utf-8",
        )
        destination = tmp_path / "OOD_texts.txt"
        assert select_ood(source, destination) == MIN_OOD_LINES
        assert all(
            len(line) >= MIN_OOD_PHONEMES
            for line in destination.read_text(encoding="utf-8").splitlines()
        )

    def test_too_few_long_lines_is_a_refusal_and_not_a_warning(self, tmp_path: Path) -> None:
        source = tmp_path / "ood.txt"
        source.write_text("\n".join(["a" * (MIN_OOD_PHONEMES + 1)] * 3) + "\n", encoding="utf-8")
        with pytest.raises(PreconditionError):
            select_ood(source, tmp_path / "out.txt")

    def test_a_missing_source_is_a_refusal(self, tmp_path: Path) -> None:
        with pytest.raises(PreconditionError):
            select_ood(tmp_path / "absent.txt", tmp_path / "out.txt")


class TestThePatches:
    def test_both_are_on_disk_where_the_container_mounts_them(self) -> None:
        assert STAGE2_PATCH.is_file()
        assert PRO_PATCH.is_file()

    def test_every_file_either_patch_touches_is_checked_out_before_they_apply(self) -> None:
        """`git apply` refuses a diff that is already applied and a preempted container is
        retried, so a file one patch touches and the revert list misses breaks the retry."""
        touched = {
            line.removeprefix("--- a/").strip()
            for patch in (STAGE2_PATCH, PRO_PATCH)
            for line in patch.read_text(encoding="utf-8").splitlines()
            if line.startswith("--- a/")
        }
        assert touched <= set(PATCHED_FILES)

    def test_the_pro_patch_carries_the_hunks_the_regime_depends_on(self) -> None:
        text = PRO_PATCH.read_text(encoding="utf-8")
        assert "LengthBucketSampler" in text
        assert "torch.manual_seed" in text
        assert "AGBALU_SLM_LAYERS" in text
        assert "AGBALU_SLM_MIN_SHARE" in text
        assert "AGBALU_MEL_LEN_ST" in text
        assert "slm_skipped" in text
