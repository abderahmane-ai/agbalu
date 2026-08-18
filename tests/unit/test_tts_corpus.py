"""The decisions that make a voice corpus, taken apart from the audio they act on.

Everything here runs once on the raw clip and is applied to both arms, so a defect in it
is a defect in the ablation rather than in one arm: a bound computed differently per arm
would make 12.7 measure its own cropping. The cases are the ones the real corpus has —
clips whose silence is longer than their speech, a level spread of 17.8 dB, transcripts
Fadhma reads as a different sentence, and words the rule table has no reading for.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from agbalu.speech.corpus import Clip
from agbalu.tts.acoustics import CLIP_LEVEL, ClipProfile
from agbalu.tts.corpus import (
    ARMS,
    CLIPPED,
    CYCLE,
    CYCLE_CLIPS,
    DEV,
    DEV_CLIPS,
    MAX_SECONDS,
    MIN_SECONDS,
    MISMATCHED,
    NO_RULE,
    OUTPUT_RATE,
    PEAK_CEILING_DBFS,
    SILENT,
    SMOKE_CYCLE,
    SMOKE_DEV,
    TOO_LONG,
    TOO_SHORT,
    TRAIN,
    CorpusError,
    Utterance,
    apply,
    assign_splits,
    bounds,
    consider,
    gain_for,
    levelled,
    name_for,
    report,
    split_sizes,
    target_level,
    training_list,
    write_lists,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

RATE: Final = 48_000
TEXT: Final = "Azul fell-ak a gma, amek i tettiliḍ?"
TARGET: Final = "azul fell-ak a gma amek i tettiliḍ"


def profile_for(
    *,
    duration_s: float = 3.0,
    lead_ms: float | None = 400.0,
    trail_ms: float | None = 300.0,
    speech_dbfs: float | None = -20.0,
    peak: float = 0.5,
    clipped_runs: int = 0,
    speech_frame_share: float = 0.6,
    rate: int = RATE,
) -> ClipProfile:
    return ClipProfile(
        samples=round(duration_s * rate),
        sample_rate=rate,
        peak=peak,
        rms_dbfs=-25.0,
        speech_dbfs=speech_dbfs,
        dc_offset=0.0,
        rolloff_hz=5_000.0,
        band_edge_hz=7_900.0,
        top_octave_db=-110.0,
        floor_dbfs=-80.0,
        speech_to_floor_db=60.0,
        speech_frame_share=speech_frame_share,
        lead_silence_ms=lead_ms,
        trail_silence_ms=trail_ms,
        decay_time_s=0.2,
        decay_offsets=1,
        clipped_runs=clipped_runs,
        clipped_sample_share=0.0,
    )


def clip_for(name: str = "common_voice_kab_1.mp3", target: str = TARGET) -> Clip:
    return Clip(
        clip=name,
        speaker="speaker",
        split="train",
        duration_ms=3_000,
        text=TEXT,
        target=target,
        repaired=False,
    )


def utterance_for(
    name: str,
    *,
    seconds: float = 3.0,
    speech_dbfs: float = -20.0,
    peak: float = 0.5,
    words: int = 6,
) -> Utterance:
    target = " ".join(["azul"] * words)
    return Utterance(
        clip=name,
        text=target,
        target=target,
        ipa="æzul",
        start_s=0.0,
        stop_s=seconds,
        speech_dbfs=speech_dbfs,
        peak=peak,
        cer=0.05,
    )


class TestNames:
    def test_each_measured_gender_names_a_corpus(self) -> None:
        assert name_for("female_feminine") == "kab_female"
        assert name_for("male_masculine") == "kab_male"

    def test_an_unlabelled_voice_is_refused_rather_than_numbered(self) -> None:
        with pytest.raises(CorpusError, match="names no corpus"):
            name_for("")

    def test_an_unknown_label_is_refused(self) -> None:
        with pytest.raises(CorpusError, match="names no corpus"):
            name_for("other")


class TestBounds:
    def test_the_cut_keeps_a_fixed_pad_either_side_of_the_speech(self) -> None:
        start, stop = bounds(profile_for(duration_s=3.0, lead_ms=400.0, trail_ms=300.0))
        assert start == pytest.approx(0.35)
        assert stop == pytest.approx(2.8)

    def test_the_pad_never_runs_past_the_clip(self) -> None:
        start, stop = bounds(profile_for(duration_s=2.0, lead_ms=0.0, trail_ms=0.0))
        assert start == 0.0
        assert stop == pytest.approx(2.0)

    def test_a_clip_with_no_speech_bound_returns_its_whole_length(self) -> None:
        assert bounds(profile_for(duration_s=4.0, lead_ms=None, trail_ms=None)) == (0.0, 4.0)

    def test_silence_longer_than_the_clip_falls_back_to_the_whole_length(self) -> None:
        assert bounds(profile_for(duration_s=1.0, lead_ms=900.0, trail_ms=900.0)) == (0.0, 1.0)

    def test_the_same_bounds_come_back_whatever_the_stored_rate_was(self) -> None:
        """Bounds are seconds precisely so both arms cut at the same place — one resampled
        down from the stored mp3, the other resynthesised at 48 kHz."""
        at_32k = bounds(profile_for(duration_s=3.0, rate=32_000))
        at_48k = bounds(profile_for(duration_s=3.0, rate=48_000))
        assert at_32k == pytest.approx(at_48k)


class TestConsider:
    def test_a_clean_clip_becomes_an_utterance_carrying_its_phonemes(self) -> None:
        verdict = consider(clip_for(), profile_for(), 0.05)
        assert isinstance(verdict, Utterance)
        assert verdict.ipa
        assert "͡" not in verdict.ipa, "the reading must already be folded onto the base's symbols"

    def test_a_clipped_recording_is_dropped(self) -> None:
        assert consider(clip_for(), profile_for(clipped_runs=2), 0.05) == CLIPPED

    def test_a_clip_whose_gate_found_no_speech_is_dropped(self) -> None:
        assert consider(clip_for(), profile_for(speech_dbfs=None), 0.05) == SILENT

    def test_a_clip_with_no_speech_frames_is_dropped(self) -> None:
        assert consider(clip_for(), profile_for(speech_frame_share=0.0), 0.05) == SILENT

    def test_a_transcript_the_decoder_reads_as_another_sentence_is_dropped(self) -> None:
        assert consider(clip_for(), profile_for(), 0.9) == MISMATCHED

    def test_an_unmeasured_transcript_does_not_drop_the_clip(self) -> None:
        assert isinstance(consider(clip_for(), profile_for(), None), Utterance)

    def test_a_clip_trimmed_below_the_floor_is_dropped(self) -> None:
        verdict = consider(clip_for(), profile_for(duration_s=2.5, lead_ms=800, trail_ms=800), 0.05)
        assert verdict == TOO_SHORT

    def test_a_clip_past_the_ceiling_is_dropped(self) -> None:
        verdict = consider(
            clip_for(), profile_for(duration_s=MAX_SECONDS + 5, lead_ms=0.0, trail_ms=0.0), 0.05
        )
        assert verdict == TOO_LONG

    def test_a_transcript_the_table_has_no_reading_for_is_dropped(self) -> None:
        assert consider(clip_for(target="azul 3 gma"), profile_for(), 0.05) == NO_RULE

    def test_kabyle_specific_letters_survive_into_the_reading(self) -> None:
        verdict = consider(clip_for(target="aţan ɛemmi yeḥwaǧ"), profile_for(), 0.05)
        assert isinstance(verdict, Utterance)
        assert "ʕ" in verdict.ipa
        assert "ħ" in verdict.ipa

    def test_the_utterance_keeps_the_bounds_the_profile_gave(self) -> None:
        verdict = consider(clip_for(), profile_for(duration_s=3.0), 0.05)
        assert isinstance(verdict, Utterance)
        assert (verdict.start_s, verdict.stop_s) == bounds(profile_for(duration_s=3.0))
        assert verdict.seconds >= MIN_SECONDS


class TestLevels:
    def test_the_target_is_the_median_gated_speech_level(self) -> None:
        clips = [
            utterance_for(f"c{i}", speech_dbfs=level) for i, level in enumerate((-30, -20, -5))
        ]
        assert target_level(clips) == pytest.approx(-20.0)

    def test_an_empty_corpus_has_no_level_and_raises(self) -> None:
        with pytest.raises(CorpusError, match="no surviving clips"):
            target_level([])

    def test_a_single_clip_is_its_own_target_and_takes_unit_gain(self) -> None:
        only = utterance_for("one", speech_dbfs=-14.0, peak=0.1)
        resolved, target = levelled([only])
        assert target == pytest.approx(-14.0)
        assert resolved[0].gain == pytest.approx(1.0)

    def test_a_quiet_clip_is_brought_up_and_a_loud_one_brought_down(self) -> None:
        quiet = utterance_for("q", speech_dbfs=-40.0, peak=0.01)
        loud = utterance_for("l", speech_dbfs=-10.0, peak=0.2)
        assert gain_for(quiet, -20.0) > 1.0
        assert gain_for(loud, -20.0) < 1.0

    def test_the_gain_reaches_the_target_when_the_peak_allows_it(self) -> None:
        clip = utterance_for("c", speech_dbfs=-30.0, peak=0.01)
        moved = 20 * np.log10(gain_for(clip, -20.0))
        assert moved == pytest.approx(10.0, abs=1e-6)

    def test_a_clip_that_would_clip_is_held_at_the_ceiling_instead(self) -> None:
        clip = utterance_for("c", speech_dbfs=-30.0, peak=0.9)
        gain = gain_for(clip, 0.0)
        assert 20 * np.log10(gain * clip.peak) == pytest.approx(PEAK_CEILING_DBFS, abs=1e-6)

    def test_a_silent_peak_does_not_divide_by_zero(self) -> None:
        assert gain_for(utterance_for("c", speech_dbfs=-30.0, peak=0.0), -20.0) > 0


class TestSplits:
    def corpus(self, count: int = 40) -> list[Utterance]:
        return [utterance_for(f"clip{index}") for index in range(count)]

    def test_the_three_splits_partition_the_corpus(self) -> None:
        assigned = assign_splits(self.corpus(), dev=5, cycle=7)
        counts = {
            split: sum(1 for u in assigned if u.split == split) for split in (TRAIN, DEV, CYCLE)
        }
        assert counts[DEV] == 5
        assert counts[CYCLE] == 7
        assert sum(counts.values()) == 40

    def test_no_clip_lands_in_two_splits(self) -> None:
        assigned = assign_splits(self.corpus(), dev=5, cycle=7)
        assert len({u.clip for u in assigned}) == len(assigned)

    def test_the_draw_reproduces_for_one_seed_and_moves_for_another(self) -> None:
        def held(seed: int) -> set[str]:
            assigned = assign_splits(self.corpus(), dev=5, cycle=7, seed=seed)
            return {u.clip for u in assigned if u.split != TRAIN}

        assert held(12) == held(12)
        assert held(12) != held(99)

    def test_the_cycle_set_only_takes_utterances_long_enough_to_score(self) -> None:
        short = [utterance_for(f"s{i}", words=2) for i in range(20)]
        long = [utterance_for(f"l{i}", words=9) for i in range(10)]
        assigned = assign_splits([*short, *long], dev=3, cycle=6, min_words=4)
        assert all(u.words >= 4 for u in assigned if u.split == CYCLE)

    def test_too_few_long_utterances_raises_rather_than_shrinking_the_cycle_set(self) -> None:
        with pytest.raises(CorpusError, match="reach"):
            assign_splits([utterance_for(f"s{i}", words=1) for i in range(30)], dev=2, cycle=5)

    def test_a_corpus_that_would_leave_nothing_to_train_on_raises(self) -> None:
        with pytest.raises(CorpusError, match="nothing to train on"):
            assign_splits(self.corpus(10), dev=5, cycle=5)

    def test_negative_sizes_raise(self) -> None:
        with pytest.raises(CorpusError, match="must not be negative"):
            assign_splits(self.corpus(), dev=-1, cycle=2)

    def test_asking_for_no_held_out_sets_leaves_everything_in_train(self) -> None:
        assigned = assign_splits(self.corpus(), dev=0, cycle=0)
        assert all(u.split == TRAIN for u in assigned)

    def test_a_capped_build_holds_out_the_smoke_sizes(self) -> None:
        assert split_sizes(20) == (SMOKE_DEV, SMOKE_CYCLE)
        assert split_sizes(0) == (DEV_CLIPS, CYCLE_CLIPS)

    def test_the_smoke_sizes_keep_a_capped_corpus_splittable(self) -> None:
        """The defect: a 20-clip cap leaves about a dozen utterances, and the recipe's
        200/300 held-out sizes refused to split it — so the smoke was the one call that
        could not run. The smoke sizes must leave every split non-empty at that size."""
        utterances = [utterance_for(f"c{i}") for i in range(12)]
        dev, cycle = split_sizes(20)
        assigned = assign_splits(utterances, dev=dev, cycle=cycle)
        counts = {
            split: sum(1 for u in assigned if u.split == split) for split in (TRAIN, DEV, CYCLE)
        }
        assert counts[DEV] == dev
        assert counts[CYCLE] == cycle
        assert counts[TRAIN] > 0


class TestApply:
    def tone(self, seconds: float, rate: int = OUTPUT_RATE) -> NDArray[np.float32]:
        return np.full(round(seconds * rate), 0.25, dtype=np.float32)

    def test_the_cut_takes_exactly_the_bounded_samples(self) -> None:
        clip = utterance_for("c")
        cut = apply(self.tone(3.0), OUTPUT_RATE, clip)
        assert cut.size == round(clip.seconds * OUTPUT_RATE)

    def test_both_arms_cut_the_same_samples_at_the_same_rate(self) -> None:
        clip = utterance_for("c")
        first = apply(self.tone(3.0), OUTPUT_RATE, clip)
        second = apply(self.tone(3.0) * np.float32(0.5), OUTPUT_RATE, clip)
        assert first.size == second.size

    def test_the_gain_is_applied(self) -> None:
        clip = replace(utterance_for("c"), gain=2.0)
        assert apply(self.tone(3.0), OUTPUT_RATE, clip)[0] == pytest.approx(0.5)

    def test_a_gain_that_would_overflow_lands_on_the_ceiling(self) -> None:
        clip = replace(utterance_for("c"), gain=10.0)
        peak = float(np.abs(apply(self.tone(3.0), OUTPUT_RATE, clip)).max())
        assert 20 * np.log10(peak) == pytest.approx(PEAK_CEILING_DBFS, abs=1e-4)

    def test_an_arm_louder_than_the_gain_assumed_is_scaled_and_never_flat_topped(self) -> None:
        # What restoration does: the gain was resolved against the raw peak and the arm
        # written here comes back hotter, so the ceiling has to hold against this waveform.
        clip = replace(utterance_for("c"), gain=1.0)
        restored = self.tone(3.0) * np.float32(4.0)
        cut = apply(restored, OUTPUT_RATE, clip)
        assert float(np.abs(cut).max()) < 1.0
        assert not bool((np.abs(cut) >= CLIP_LEVEL).any())

    def test_scaling_preserves_the_waveform(self) -> None:
        clip = replace(utterance_for("c"), gain=8.0)
        shape = np.linspace(-1.0, 1.0, OUTPUT_RATE * 3, dtype=np.float32)
        cut = apply(shape, OUTPUT_RATE, clip)
        ratios = cut[np.abs(cut) > 1e-6] / shape[: cut.size][np.abs(cut) > 1e-6]
        assert float(ratios.std()) == pytest.approx(0.0, abs=1e-6)

    def test_non_finite_samples_raise(self) -> None:
        broken = self.tone(3.0).copy()
        broken[7] = np.float32("inf")
        with pytest.raises(CorpusError, match="non-finite"):
            apply(broken, OUTPUT_RATE, utterance_for("c"))

    def test_audio_shorter_than_the_bounds_is_cut_to_what_exists(self) -> None:
        clip = utterance_for("c", seconds=3.0)
        assert apply(self.tone(1.0), OUTPUT_RATE, clip).size == OUTPUT_RATE

    def test_a_cut_that_would_be_empty_raises(self) -> None:
        with pytest.raises(CorpusError, match="cuts to nothing"):
            apply(np.zeros(0, dtype=np.float32), OUTPUT_RATE, utterance_for("c"))

    def test_a_non_positive_rate_raises(self) -> None:
        with pytest.raises(CorpusError, match="sample rate"):
            apply(self.tone(3.0), 0, utterance_for("c"))


class TestLists:
    def test_the_list_carries_the_path_the_phonemes_and_the_speaker(self) -> None:
        rows = list(
            training_list([utterance_for("common_voice_kab_9.mp3")], "kab_male", audio_dir="/a")
        )
        assert rows == ["/a/common_voice_kab_9.wav|æzul|kab_male"]

    def test_an_empty_split_writes_an_empty_file_rather_than_none(self, tmp_path: Path) -> None:
        write_lists(tmp_path, "kab_male", [], audio_dir="/a")
        assert (tmp_path / "kab_male.train.txt").read_text(encoding="utf-8") == ""

    def test_each_split_gets_its_own_list(self, tmp_path: Path) -> None:
        assigned = assign_splits([utterance_for(f"c{i}") for i in range(30)], dev=3, cycle=4)
        write_lists(tmp_path, "kab_male", assigned, audio_dir="/a")
        counts = [
            len((tmp_path / f"kab_male.{split}.txt").read_text(encoding="utf-8").splitlines())
            for split in (TRAIN, DEV, CYCLE)
        ]
        assert counts == [23, 3, 4]

    def test_the_manifest_round_trips_every_decision(self, tmp_path: Path) -> None:
        clips, _ = levelled([utterance_for("c1"), utterance_for("c2", speech_dbfs=-30.0)])
        path = write_manifest(tmp_path, "kab_male", clips)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [row["clip"] for row in rows] == ["c1", "c2"]
        assert all("gain" in row and "start_s" in row and "ipa" in row for row in rows)

    def test_a_reading_with_a_pipe_would_break_the_format_and_none_is_produced(self) -> None:
        """The list is pipe-delimited, so a phoneme string carrying one would silently gain
        a field. The rule table emits no such symbol, asserted over real readings rather
        than over the table that was used to build them."""
        readings = [
            consider(clip_for(target=word), profile_for(), 0.05)
            for word in ("azul fell-ak", "aţan ɛemmi yeḥwaǧ", "ččuṛ tameṭṭut")
        ]
        for reading in readings:
            assert isinstance(reading, Utterance)
            assert "|" not in reading.ipa


class TestReport:
    def test_the_report_counts_what_was_kept_and_what_was_dropped(self) -> None:
        clips, target = levelled([utterance_for(f"c{i}") for i in range(10)])
        summary = report(
            "kab_male", "abc", assign_splits(clips, dev=2, cycle=3), {CLIPPED: 4}, target
        )
        assert summary["clips"] == 10
        assert summary["considered"] == 14
        assert summary["dropped"] == {CLIPPED: 4}

    def test_the_report_names_the_rate_the_audio_was_written_at(self) -> None:
        clips, target = levelled([utterance_for("c")])
        assert report("kab_male", "abc", clips, {}, target)["output_rate"] == OUTPUT_RATE

    def test_the_report_records_the_inventory_the_base_has_to_represent(self) -> None:
        """The token table lives with the checkpoint and drops what it cannot hold without
        raising, so what the corpus actually emits is recorded rather than assumed."""
        kept = [
            verdict
            for word in ("azul fell-ak", "aţan ɛemmi yeḥwaǧ")
            if isinstance(
                verdict := consider(clip_for(target=word), profile_for(), 0.05), Utterance
            )
        ]
        clips, target = levelled(kept)
        summary = report("kab_male", "abc", clips, {}, target)
        inventory = summary["inventory"]
        assert isinstance(inventory, str)
        assert {"ʕ", "ħ"} <= set(inventory)
        assert "͡" not in inventory
        assert summary["inventory_size"] == len(set(inventory))

    def test_the_restored_arm_is_named_first(self) -> None:
        assert ARMS[0] == "restored"
