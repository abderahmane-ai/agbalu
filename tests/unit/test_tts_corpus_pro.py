"""The joining decisions, taken apart from the audio they act on.

Three of them are the whole of Pro's corpus and each is a defect in v1's. Punctuation reaches
the target. Consecutive Common Voice ids are one session, so clips join. And the seam between
two joined clips is *not* trimmed, which is where the speaker's real pause — and the breath
in it — lives. The cases here are the ones the real corpus has: a broken id run, a contributor
who paused four seconds between recordings, a clip whose silence is longer than its speech,
and a transcript Fadhma reads as a different sentence.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Final

import pytest

from agbalu.speech.corpus import Clip
from agbalu.tts.acoustics import ClipProfile
from agbalu.tts.corpus import (
    CLIPPED,
    MISMATCHED,
    NO_RULE,
    SILENT,
    TOO_LONG,
    CorpusError,
    level_gain,
)
from agbalu.tts.corpus_pro import (
    LEAD_KEEP_MS,
    MAX_SECONDS,
    PACK_SECONDS,
    RESERVED,
    SEAM_MAX_S,
    TRAIL_KEEP_MS,
    UNSEATED,
    Source,
    assign_dev,
    build,
    clip_order,
    held_out,
    join,
    pack,
    report,
    seam,
    sessions,
    span,
    target_level,
    training_list,
    verdict,
)

if TYPE_CHECKING:
    from pathlib import Path

RATE: Final = 48_000
NOTHING: Final = frozenset[str]()


Varied = dict[str, float | int | None]
"""The keywords `profile` varies, one per measurement a clip can be dropped on."""


def profile(
    *,
    seconds: float | int | None = 3.0,
    lead_ms: float | int | None = 480.0,
    trail_ms: float | int | None = 320.0,
    speech_dbfs: float | int | None = -19.0,
    peak: float | int | None = 0.5,
    clipped: float | int | None = 0,
    share: float | int | None = 0.7,
) -> ClipProfile:
    """A clip's measurements, shaped like the male voice's own: 12.3 measured a median 480 ms
    of lead and 320 ms of trail on it, which is what v1 threw away."""
    return ClipProfile(
        samples=round(float(seconds or 0.0) * RATE),
        sample_rate=RATE,
        peak=float(peak or 0.0),
        rms_dbfs=-22.0,
        speech_dbfs=None if speech_dbfs is None else float(speech_dbfs),
        dc_offset=0.0,
        rolloff_hz=3400.0,
        band_edge_hz=7860.0,
        top_octave_db=-40.0,
        floor_dbfs=-55.0,
        speech_to_floor_db=36.0,
        speech_frame_share=float(share or 0.0),
        lead_silence_ms=None if lead_ms is None else float(lead_ms),
        trail_silence_ms=None if trail_ms is None else float(trail_ms),
        decay_time_s=0.2,
        decay_offsets=1,
        clipped_runs=int(clipped or 0),
        clipped_sample_share=0.0,
    )


def clip(index: int, *, text: str = "Azul fell-awen.", speaker: str = "s") -> Clip:
    return Clip(
        clip=f"common_voice_kab_{23405507 + index}.mp3",
        speaker=speaker,
        split="train",
        duration_ms=3000,
        text=text,
        target=text.lower().rstrip("."),
        repaired=False,
    )


def source(index: int, *, seconds: float = 3.0, lead: float = 0.48, trail: float = 0.32) -> Source:
    return Source(
        clip=f"common_voice_kab_{23405507 + index}.mp3",
        text="Azul fell-awen.",
        ipa="æzul fəlː æwən.",
        order=23405507 + index,
        duration_s=seconds,
        speech_start_s=lead,
        speech_end_s=seconds - trail,
        speech_dbfs=-19.0,
        peak=0.5,
        cer=0.0,
    )


class TestTheVerdict:
    def test_a_kept_clip_carries_its_punctuated_reading(self) -> None:
        kept = verdict(clip(0), profile(), 0.0, reserved=NOTHING, cer_limit=0.5)
        assert isinstance(kept, Source)
        assert kept.ipa.endswith(".")

    def test_it_reads_text_and_not_the_ASR_target(self) -> None:
        """`corpus.py:207` phonemises `clip.target`, which never carried a mark. The correct
        string was one field away in the same dataclass."""
        kept = verdict(clip(0), profile(), None, reserved=NOTHING, cer_limit=0.5)
        assert isinstance(kept, Source)
        assert "." in kept.ipa

    @pytest.mark.parametrize(
        ("built", "expected"),
        [
            ({"clipped": 3}, CLIPPED),
            ({"speech_dbfs": None}, SILENT),
            ({"share": 0.0}, SILENT),
        ],
    )
    def test_an_unusable_clip_is_dropped_by_name(self, built: Varied, expected: str) -> None:
        assert verdict(clip(0), profile(**built), None, reserved=NOTHING, cer_limit=0.5) == expected

    def test_a_transcript_of_a_different_sentence_is_dropped(self) -> None:
        assert verdict(clip(0), profile(), 0.9, reserved=NOTHING, cer_limit=0.5) == MISMATCHED

    def test_a_word_with_no_rule_is_dropped_rather_than_mispronounced(self) -> None:
        assert (
            verdict(clip(0, text="Zénith"), profile(), None, reserved=NOTHING, cer_limit=0.5)
            == NO_RULE
        )

    def test_a_clip_held_out_by_12_4_never_enters(self) -> None:
        """Pro inherits Matoub-82M's evaluation set whole, so the two are scored on the same
        prompts and Pro's own Cycle-CER is not a memorisation score."""
        target = clip(0)
        assert (
            verdict(target, profile(), None, reserved=frozenset({target.clip}), cer_limit=0.5)
            == RESERVED
        )

    def test_a_clip_whose_name_carries_no_session_id_is_dropped(self) -> None:
        odd = Clip(
            clip="something-else.wav",
            speaker="s",
            split="train",
            duration_ms=3000,
            text="Azul.",
            target="azul",
            repaired=False,
        )
        assert verdict(odd, profile(), None, reserved=NOTHING, cer_limit=0.5) == UNSEATED

    def test_the_id_is_read_from_the_name_and_nothing_else(self) -> None:
        assert clip_order("common_voice_kab_23405507.mp3") == 23405507
        assert clip_order("clip.wav") is None


class TestSessions:
    def test_consecutive_ids_are_one_run(self) -> None:
        runs = list(sessions([source(2), source(0), source(1)]))
        assert len(runs) == 1
        assert [item.order for item in runs[0]] == sorted(item.order for item in runs[0])

    def test_a_gap_in_the_ids_starts_a_new_session(self) -> None:
        assert len(list(sessions([source(0), source(1), source(9)]))) == 2

    def test_a_long_silence_splits_a_run_rather_than_being_clamped(self) -> None:
        """A contributor who paused four seconds between two recordings would otherwise teach
        a four-second full stop. A clamped gap is a fabricated pause."""
        long_trail = source(0, trail=SEAM_MAX_S + 1.0)
        assert len(list(sessions([long_trail, source(1)]))) == 2

    def test_the_seam_is_what_the_two_clips_actually_carry(self) -> None:
        assert seam(source(0), source(1)) == pytest.approx(0.32 + 0.48)


class TestTheCut:
    def test_a_lone_clip_is_trimmed_on_both_sides(self) -> None:
        start, stop = span([source(0)], 0)
        assert start == pytest.approx(0.48 - LEAD_KEEP_MS / 1000.0)
        assert stop == pytest.approx(3.0 - 0.32 + TRAIL_KEEP_MS / 1000.0)

    def test_an_interior_edge_keeps_the_whole_clip(self) -> None:
        """The seam is the point: the silence between two sentences is the speaker's, not a
        constant chosen here, and any breath taken in it is in the training target."""
        members = [source(0), source(1), source(2)]
        assert span(members, 1) == (0.0, members[1].duration_s)

    def test_the_first_clip_of_a_run_keeps_its_trailing_silence(self) -> None:
        members = [source(0), source(1)]
        assert span(members, 0)[1] == members[0].duration_s

    def test_the_last_clip_of_a_run_keeps_its_leading_silence(self) -> None:
        members = [source(0), source(1)]
        assert span(members, 1)[0] == 0.0

    def test_a_clip_with_less_lead_than_the_cap_keeps_what_it_has(self) -> None:
        assert span([source(0, lead=0.05)], 0)[0] == 0.0


class TestPacking:
    def test_a_run_is_cut_at_the_target(self) -> None:
        run = [source(index) for index in range(8)]
        for members in pack(run):
            assert sum(span(members, i)[1] - span(members, i)[0] for i in range(len(members))) <= (
                PACK_SECONDS
            )

    def test_every_clip_lands_in_exactly_one_utterance(self) -> None:
        run = [source(index) for index in range(11)]
        placed = [item.clip for members in pack(run) for item in members]
        assert sorted(placed) == sorted(item.clip for item in run)

    def test_a_clip_longer_than_the_target_is_still_kept_alone(self) -> None:
        run = [source(0, seconds=PACK_SECONDS + 2.0)]
        assert [len(members) for members in pack(run)] == [1]

    def test_an_utterance_past_the_ceiling_is_dropped_by_name(self) -> None:
        assert join([source(0, seconds=MAX_SECONDS + 5.0)], "kab_male", -19.0) == TOO_LONG


class TestJoining:
    def test_the_reading_is_the_members_joined_in_order(self) -> None:
        built = join([source(0), source(1)], "kab_male", -19.0)
        assert not isinstance(built, str)
        assert built.ipa == "æzul fəlː æwən. æzul fəlː æwən."
        assert built.clips == 2

    def test_the_name_identifies_its_first_clip_and_its_length(self) -> None:
        built = join([source(0), source(1)], "kab_male", -19.0)
        assert not isinstance(built, str)
        assert built.name == "common_voice_kab_23405507-2"

    def test_the_gain_is_per_clip_and_is_the_shared_one(self) -> None:
        """Two pieces levelled differently meet at an audible step, so each member is brought
        to the corpus's own level by `agbalu.tts.corpus.level_gain` — one definition, used
        here and by 12.4."""
        quiet = replace(source(0), speech_dbfs=-30.0)
        built = join([quiet, source(1)], "kab_male", -19.0)
        assert not isinstance(built, str)
        gains = [member.gain for member in built.members]
        assert gains[0] > gains[1]
        assert gains[0] == pytest.approx(level_gain(-30.0, 0.5, -19.0))
        assert gains[1] == pytest.approx(level_gain(-19.0, 0.5, -19.0))

    def test_it_refuses_an_empty_run(self) -> None:
        with pytest.raises(CorpusError):
            join([], "kab_male", -19.0)


class TestBuildAndSplits:
    def test_a_speaker_builds_utterances_and_reports_what_packing_dropped(self) -> None:
        built, dropped, target = build([source(index) for index in range(20)], "kab_male")
        assert built
        assert target == pytest.approx(-19.0)
        assert sum(item.clips for item in built) + sum(dropped.values()) == 20

    def test_the_validation_draw_is_seeded_and_disjoint(self) -> None:
        built, _, _ = build([source(index) for index in range(60)], "kab_male")
        once = assign_dev(built, dev=5)
        twice = assign_dev(built, dev=5)
        assert [item.split for item in once] == [item.split for item in twice]
        assert sum(item.split == "dev" for item in once) == 5

    def test_it_refuses_a_validation_list_that_leaves_nothing_to_train_on(self) -> None:
        built, _, _ = build([source(index) for index in range(4)], "kab_male")
        with pytest.raises(CorpusError):
            assign_dev(built, dev=len(built))

    def test_the_level_target_is_the_corpus_median_and_not_a_constant(self) -> None:
        assert target_level([source(0), source(1)]) == pytest.approx(-19.0)
        with pytest.raises(CorpusError):
            target_level([])


class TestTheRecord:
    def test_the_filelist_names_the_wav_the_render_writes(self) -> None:
        built, _, _ = build([source(index) for index in range(4)], "kab_male")
        rows = list(training_list(built, audio_dir="/data/tts_pro/corpus/audio"))
        assert rows[0].startswith("/data/tts_pro/corpus/audio/common_voice_kab_23405507-")
        assert rows[0].count("|") == 2

    def test_the_report_carries_the_distributions_the_joins_rest_on(self) -> None:
        built, dropped, target = build([source(index) for index in range(30)], "kab_male")
        payload = report("kab_male", "abc", built, dropped, target, considered=30)
        assert json.dumps(payload)
        rendered = json.loads(json.dumps(payload))
        assert rendered["with_internal_boundary"] > 0
        assert rendered["over_slmadv_floor"] > 0
        assert rendered["seam_s"]["measured"] > 0
        assert rendered["pack"]["pack_seconds"] == PACK_SECONDS

    def test_the_report_survives_a_corpus_of_one(self) -> None:
        """A distribution over nothing reports `None`, never `0.0`: a clamp turns "no data"
        into a plausible number, and a seam that does not exist has no length."""
        built, dropped, target = build([source(0)], "kab_male")
        rendered = json.loads(
            json.dumps(report("kab_male", "abc", built, dropped, target, considered=1))
        )
        assert rendered["seam_s"]["measured"] == 0
        assert rendered["seam_s"]["median"] is None


class TestInheritingTheHeldOutSet:
    def test_it_reads_dev_and_cycle_and_nothing_else(self, tmp_path: Path) -> None:
        manifest = tmp_path / "kab_male.manifest.jsonl"
        manifest.write_text(
            "\n".join(
                json.dumps({"clip": name, "split": split})
                for name, split in (("a.mp3", "train"), ("b.mp3", "dev"), ("c.mp3", "cycle"))
            )
            + "\n",
            encoding="utf-8",
        )
        assert held_out(manifest) == frozenset({"b.mp3", "c.mp3"})

    def test_a_missing_manifest_raises_rather_than_holding_nothing_out(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(CorpusError):
            held_out(tmp_path / "absent.jsonl")

    def test_a_manifest_that_reserves_nothing_raises(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text(
            json.dumps({"clip": "a.mp3", "split": "train"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(CorpusError):
            held_out(manifest)
