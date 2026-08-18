"""The parts of the ASR entrypoint that run without a GPU (task 5.5).

Three things here decide whether a twenty-hour run produces a usable model, and none of
them needs a GPU to check: how batches are formed, what the learning rate does over the
schedule, and which clips validation looks at. A moving validation subset alone would
make `best.pt` a coin toss between checkpoints.
"""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from modal_app.asr import (
    EXPECTED_SECONDS_PER_STEP,
    FINAL_LR_FRACTION,
    FLOOR_CHECK_AFTER,
    FLOOR_FACTOR,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    LOCAL_SPEECH,
    LOCAL_VOCABULARY,
    PACK_SCALE,
    REMOTE_LM_FILE,
    REMOTE_SPEECH,
    SAMPLE_RATE,
    SPLITS,
    VOCABULARY_FILE,
    WARMUP_CAP,
    ThroughputGuard,
    _quantise,
    _summary,
    _write_shard,
    buckets,
    learning_rate,
    load_pack,
    next_shard,
    pack_gaps,
    pack_order,
    read_shard,
    shard_paths,
    speech_uploads,
    steps_per_epoch,
    stream,
    subsample,
    throughput_breach,
    warmup_steps,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

from agbalu.model.checkpoint import TrainingState
from agbalu.speech.corpus import Clip


def clip(name: str, duration_ms: int) -> Clip:
    return Clip(
        clip=name,
        speaker="spk",
        split="train",
        duration_ms=duration_ms,
        text="Azul",
        target="azul",
        repaired=False,
    )


class TestBuckets:
    def test_no_clips(self) -> None:
        assert buckets([], 320) == []

    def test_one_clip(self) -> None:
        assert [len(b) for b in buckets([clip("a", 3000)], 320)] == [1]

    def test_every_clip_appears_exactly_once(self) -> None:
        clips = [clip(f"{i}.mp3", 1000 + 137 * i) for i in range(200)]
        names = [c.clip for batch in buckets(clips, 30) for c in batch]
        assert sorted(names) == sorted(c.clip for c in clips)

    def test_no_batch_exceeds_the_budget_once_padded(self) -> None:
        """The cost is `longest x rows`, not the sum: every row is padded to the longest."""
        clips = [clip(f"{i}.mp3", 400 + 100 * i) for i in range(120)]
        for batch in buckets(clips, 30):
            padded = max(c.duration_ms for c in batch) * len(batch)
            assert padded <= 30_000 or len(batch) == 1

    def test_a_clip_longer_than_the_whole_budget_still_gets_a_batch(self) -> None:
        clips = [clip("long.mp3", 20_000), clip("short.mp3", 1_000)]
        assert sum(len(b) for b in buckets(clips, 5)) == 2

    def test_batches_are_duration_sorted_so_padding_is_cheap(self) -> None:
        clips = [clip(f"{i}.mp3", ms) for i, ms in enumerate([9000, 1000, 5000, 2000])]
        flat = [c.duration_ms for batch in buckets(clips, 320) for c in batch]
        assert flat == sorted(flat)

    def test_a_uniform_set_packs_to_the_budget(self) -> None:
        batches = buckets([clip(f"{i}.mp3", 1000) for i in range(100)], 10)
        assert all(len(b) == 10 for b in batches)
        assert len(batches) == 10

    @pytest.mark.parametrize("seconds", [1, 5, 30, 160])
    def test_a_bigger_budget_never_makes_more_batches(self, seconds: int) -> None:
        clips = [clip(f"{i}.mp3", 2000) for i in range(60)]
        assert len(buckets(clips, seconds)) >= len(buckets(clips, seconds * 2))

    def test_the_result_is_deterministic(self) -> None:
        """Two runs meant to be compared must see identical batches."""
        clips = [clip(f"{i}.mp3", 500 + 311 * (i % 17)) for i in range(80)]
        first = [[c.clip for c in batch] for batch in buckets(clips, 20)]
        second = [[c.clip for c in batch] for batch in buckets(list(reversed(clips)), 20)]
        assert first == second


class TestLearningRate:
    TOTAL = 10_000

    def test_warms_up_from_near_zero(self) -> None:
        warmup = warmup_steps(self.TOTAL)
        assert learning_rate(0, self.TOTAL) == pytest.approx(LEARNING_RATE / warmup)

    def test_reaches_the_peak_at_the_end_of_warmup(self) -> None:
        warmup = warmup_steps(self.TOTAL)
        assert learning_rate(warmup - 1, self.TOTAL) == pytest.approx(LEARNING_RATE)

    def test_decays_to_the_floor_at_the_last_step(self) -> None:
        floor = LEARNING_RATE * FINAL_LR_FRACTION
        assert learning_rate(self.TOTAL, self.TOTAL) == pytest.approx(floor)

    def test_never_exceeds_the_peak_and_stays_positive(self) -> None:
        rates = [learning_rate(step, self.TOTAL) for step in range(0, self.TOTAL + 1, 7)]
        assert all(0 < rate <= LEARNING_RATE + 1e-12 for rate in rates)

    def test_the_decay_phase_never_falls_below_the_floor(self) -> None:
        warmup = warmup_steps(self.TOTAL)
        floor = LEARNING_RATE * FINAL_LR_FRACTION
        rates = [learning_rate(step, self.TOTAL) for step in range(warmup, self.TOTAL + 1)]
        assert min(rates) >= floor - 1e-12

    def test_decreases_monotonically_after_warmup(self) -> None:
        warmup = warmup_steps(self.TOTAL)
        rates = [learning_rate(step, self.TOTAL) for step in range(warmup, self.TOTAL + 1, 13)]
        assert rates == sorted(rates, reverse=True)

    @pytest.mark.parametrize("total", [1, 2, 20, 50])
    def test_a_short_run_still_reaches_the_peak(self, total: int) -> None:
        """A fixed 500-step warmup left a 20-step smoke at 4% of the peak — it verified
        the plumbing and nothing about whether the model learns."""
        rates = [learning_rate(step, total) for step in range(total)]
        assert max(rates) == pytest.approx(LEARNING_RATE)
        assert all(math.isfinite(rate) for rate in rates)

    def test_a_long_run_caps_the_warmup(self) -> None:
        assert warmup_steps(1_000_000) == WARMUP_CAP

    def test_an_unknown_total_holds_at_the_peak_after_warmup(self) -> None:
        assert learning_rate(WARMUP_CAP + 100, 0) == pytest.approx(LEARNING_RATE)


class TestSubsample:
    def test_a_split_under_the_cap_is_returned_whole(self) -> None:
        clips = [clip(f"{i}.mp3", 1000 + i) for i in range(10)]
        assert len(subsample(clips, 100)) == 10

    def test_the_cap_is_respected(self) -> None:
        clips = [clip(f"{i}.mp3", 1000 + i) for i in range(15_002)]
        assert len(subsample(clips, 1500)) == 1500

    def test_it_is_deterministic_regardless_of_input_order(self) -> None:
        """A moving validation subset makes two checkpoints incomparable, and `best.pt`
        is chosen by exactly that comparison."""
        clips = [clip(f"{i}.mp3", 1000 + (i * 37) % 900) for i in range(500)]
        first = [c.clip for c in subsample(clips, 50)]
        second = [c.clip for c in subsample(list(reversed(clips)), 50)]
        assert first == second

    def test_it_spans_the_duration_range(self) -> None:
        clips = [clip(f"{i}.mp3", 500 + 10 * i) for i in range(1000)]
        taken = subsample(clips, 20)
        assert taken[0].duration_ms == 500
        assert taken[-1].duration_ms > 9000

    @pytest.mark.parametrize("cap", [0, -1])
    def test_a_non_positive_cap_disables_the_subset(self, cap: int) -> None:
        clips = [clip(f"{i}.mp3", 1000 + i) for i in range(40)]
        assert len(subsample(clips, cap)) == 40

    def test_no_duplicates(self) -> None:
        clips = [clip(f"{i}.mp3", 1000) for i in range(900)]
        taken = subsample(clips, 100)
        assert len({c.clip for c in taken}) == len(taken)


class TestStream:
    """The loader's contract: in order, at most `depth` ahead, and never swallowing.

    The first fine-tune spent 304 ms per clip decoding audio on the training thread while
    the GPU held at 1% of its peak, so this is the difference between a 13-hour epoch and
    an hour of one. Order matters because a batch's waveforms are zipped back against the
    clips they were loaded for; getting it wrong pairs audio with another clip's labels
    and trains on noise with nothing in the loss to show it.
    """

    def test_no_items(self) -> None:
        calls: list[int] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(stream([], calls.append, pool, 4)) == []
        assert calls == []

    def test_yields_in_source_order_when_loads_finish_out_of_order(self) -> None:
        gate = threading.Event()

        def load(value: int) -> int:
            if value == 0:
                gate.wait(timeout=10)
            else:
                gate.set()
            return value

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(stream(range(6), load, pool, 4)) == [0, 1, 2, 3, 4, 5]

    def test_loads_each_item_exactly_once(self) -> None:
        guard = threading.Lock()
        seen: list[int] = []

        def load(value: int) -> int:
            with guard:
                seen.append(value)
            return value

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(stream(range(20), load, pool, 4)) == list(range(20))
        assert sorted(seen) == list(range(20))

    def test_depth_beyond_the_source_is_not_an_error(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(stream(range(3), lambda v: v, pool, 32)) == [0, 1, 2]

    def test_runs_no_further_ahead_than_depth(self) -> None:
        """The bound is the whole point: an unbounded map would decode the split into
        memory, and a depth of zero would put decoding back in front of the GPU."""
        guard = threading.Lock()
        seen: list[int] = []

        def load(value: int) -> int:
            with guard:
                seen.append(value)
            return value

        with ThreadPoolExecutor(max_workers=2) as pool:
            batches = stream(range(1000), load, pool, 4)
            assert next(batches) == 0
            with guard:
                assert len(seen) <= 5

    def test_a_failed_load_reaches_the_consumer(self) -> None:
        def load(value: int) -> int:
            if value == 3:
                message = "unreadable clip"
                raise RuntimeError(message)
            return value

        with ThreadPoolExecutor(max_workers=2) as pool:
            batches = stream(range(6), load, pool, 2)
            assert [next(batches), next(batches), next(batches)] == [0, 1, 2]
            with pytest.raises(RuntimeError, match="unreadable clip"):
                next(batches)


class TestStepsPerEpoch:
    """Validation once an epoch. A fixed 2,000 gave four points across ten epochs, which
    cannot show where the curve flattens — and `best.pt` is chosen by exactly that curve."""

    def test_the_real_corpus_validates_ten_times_over_ten_epochs(self) -> None:
        assert steps_per_epoch(3287) == 821

    def test_a_schedule_shorter_than_one_step_still_validates(self) -> None:
        assert steps_per_epoch(1) == 1
        assert steps_per_epoch(0) == 1

    def test_it_tracks_the_accumulation_rather_than_a_constant(self) -> None:
        assert steps_per_epoch(3287) * GRADIENT_ACCUMULATION <= 3287


class TestSummary:
    """A resumed run inherits every cumulative counter, so the result must say what this
    call did. The encoder run learned it the same way: a smoke resumed at step 20 of 20,
    trained nothing, and printed the checkpoint's numbers as a measurement."""

    def test_a_completed_resume_reports_no_steps_of_its_own(self) -> None:
        state = TrainingState(step=20, epoch=1, best_validation_loss=3.1953)
        summary = _summary("fadhma-v1", state, audio_seconds=0.0, trained=0)
        assert summary["steps_this_run"] == 0
        assert summary["steps"] == 20
        assert summary["audio_hours"] == 0.0

    def test_a_run_that_trained_reports_its_own_share(self) -> None:
        state = TrainingState(step=50, epoch=2)
        summary = _summary("fadhma-v1", state, audio_seconds=7200.0, trained=30)
        assert summary["steps_this_run"] == 30
        assert summary["steps"] == 50
        assert summary["audio_hours"] == 2.0


class TestSpeechUploads:
    """`fetch` brings the audio and nothing else.

    Everything else the run opens on the volume has to be put there by `upload_speech`, and
    the place a missing file surfaces otherwise is minutes into a GPU container.
    """

    def test_every_file_the_container_opens_is_uploaded(self) -> None:
        uploads = speech_uploads()
        opened = {f"/{REMOTE_SPEECH.name}/{split}.jsonl" for split in SPLITS}
        opened.add(f"/{REMOTE_SPEECH.name}/{VOCABULARY_FILE}")
        opened.add(f"/{REMOTE_SPEECH.name}/{REMOTE_LM_FILE}")
        assert set(uploads) == opened

    def test_the_map_does_not_depend_on_what_is_on_this_disk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The language model was listed only when it already existed locally, so on a
        machine that had not built it `upload_speech` uploaded everything else, reported
        nothing missing, and the evaluation died inside kenlm on the volume."""
        monkeypatch.setattr("modal_app.asr.LOCAL_LM_BINARY", Path("artifacts/asr/absent.klm"))
        assert f"/{REMOTE_SPEECH.name}/{REMOTE_LM_FILE}" in speech_uploads()

    def test_the_sources_are_where_the_build_targets_write(self) -> None:
        uploads = speech_uploads()
        assert uploads[f"/{REMOTE_SPEECH.name}/train.jsonl"] == LOCAL_SPEECH / "train.jsonl"
        assert uploads[f"/{REMOTE_SPEECH.name}/{VOCABULARY_FILE}"] == LOCAL_VOCABULARY

    def test_no_source_is_uploaded_twice_under_two_names(self) -> None:
        uploads = speech_uploads()
        assert len(set(uploads.values())) == len(uploads)

    def test_the_remote_directory_is_the_one_the_container_reads(self) -> None:
        """A leading slash and the volume's own directory name: the mount point is
        `/data/speech`, and `batch_upload` paths are relative to the volume root."""
        assert all(remote.startswith(f"/{REMOTE_SPEECH.name}/") for remote in speech_uploads())
        assert REMOTE_SPEECH.name == "speech"


def tone(samples: int, *, amplitude: float = 0.5) -> NDArray[np.float32]:
    """A deterministic waveform whose values span the range, so quantisation is visible."""
    curve = amplitude * np.sin(np.linspace(0.0, 8.0, samples, dtype=np.float32))
    shaped: NDArray[np.float32] = curve.astype(np.float32)
    return shaped


def pack(
    root: Path, split: str, waves: dict[str, NDArray[np.float32]], number: int = 0
) -> list[Clip]:
    """Write one shard and return the clips it holds, in the order it holds them."""
    (root / split).mkdir(parents=True, exist_ok=True)
    clips = [clip(name, len(wave) * 1000 // SAMPLE_RATE) for name, wave in waves.items()]
    _write_shard(root / split, number, clips, [_quantise(w) for w in waves.values()])
    return clips


class TestQuantise:
    def test_round_trips_within_one_quantisation_step(self) -> None:
        wave = tone(2048)
        restored = np.asarray(_quantise(wave), dtype=np.float32) / PACK_SCALE
        assert np.max(np.abs(restored - wave)) <= 1.0 / PACK_SCALE

    @pytest.mark.parametrize("value", [1.0, 2.0, -1.0, -2.0])
    def test_full_scale_and_beyond_stay_in_range(self, value: float) -> None:
        """`+1.0 * 32768` is one past int16, and mp3 decoding does overshoot ±1.0."""
        packed = _quantise(np.full(8, value, dtype=np.float32))
        assert packed.dtype == np.int16
        assert np.all(packed >= -32768)
        assert np.all(packed <= 32767)

    def test_an_empty_waveform_is_not_an_error(self) -> None:
        assert _quantise(np.zeros(0, dtype=np.float32)).size == 0

    def test_silence_stays_silent(self) -> None:
        assert not np.any(_quantise(np.zeros(64, dtype=np.float32)))


class TestPackRoundTrip:
    """The pack replaced a per-clip mp3 open, so its only defensible property is that it
    returns what the decode put in — for every clip, not for the first one."""

    def test_every_clip_comes_back_at_its_own_samples(self, tmp_path: Path) -> None:
        waves = {f"{n}.mp3": tone(100 + 37 * n) for n in range(5)}
        clips = pack(tmp_path, "train", waves)
        loaded = load_pack("train", clips, tmp_path)
        for name, wave in waves.items():
            assert np.max(np.abs(loaded.waveform(name) - wave)) <= 1.0 / PACK_SCALE

    def test_lengths_are_preserved_exactly(self, tmp_path: Path) -> None:
        waves = {f"{n}.mp3": tone(100 + 37 * n) for n in range(5)}
        clips = pack(tmp_path, "train", waves)
        loaded = load_pack("train", clips, tmp_path)
        assert [loaded.waveform(c.clip).size for c in clips] == [len(w) for w in waves.values()]

    def test_spans_are_contiguous_and_ordered(self, tmp_path: Path) -> None:
        """Contiguity is the point: it is what makes the read sequential rather than 186
        separate seeks, which is the whole reason the pack exists."""
        waves = {f"{n}.mp3": tone(64 * (n + 1)) for n in range(6)}
        clips = pack(tmp_path, "train", waves)
        rows = read_shard(shard_paths("train", tmp_path)[0])
        offset = 0
        for (name, at, length), expected in zip(rows, waves.items(), strict=True):
            assert (name, at) == (expected[0], offset)
            offset += length
        assert offset == sum(len(w) for w in waves.values())
        del clips

    def test_a_single_clip_split_works(self, tmp_path: Path) -> None:
        clips = pack(tmp_path, "dev", {"only.mp3": tone(320)})
        assert load_pack("dev", clips, tmp_path).waveform("only.mp3").size == 320

    def test_clips_spread_over_several_shards_all_resolve(self, tmp_path: Path) -> None:
        first = pack(tmp_path, "train", {"a.mp3": tone(96)}, number=0)
        second = pack(tmp_path, "train", {"b.mp3": tone(160)}, number=1)
        loaded = load_pack("train", [*first, *second], tmp_path)
        assert loaded.waveform("a.mp3").size == 96
        assert loaded.waveform("b.mp3").size == 160
        assert len(loaded.descriptors) == 2
        assert {shard for shard, _, _ in loaded.spans.values()} == {0, 1}

    def test_a_zero_length_clip_survives_the_round_trip(self, tmp_path: Path) -> None:
        clips = pack(tmp_path, "train", {"empty.mp3": tone(0), "real.mp3": tone(48)})
        loaded = load_pack("train", clips, tmp_path)
        assert loaded.waveform("empty.mp3").size == 0
        assert loaded.waveform("real.mp3").size == 48


class TestPackLocality:
    """The pack's whole purpose. Round-tripping the bytes is necessary and not sufficient:
    written in the wrong order it returns every sample correctly and still leaves a batch as
    ~46 reads scattered across 16 GB, which is the access pattern being removed."""

    def test_a_batch_is_one_contiguous_span_of_the_pack(self, tmp_path: Path) -> None:
        clips = [clip(f"{n}.mp3", 400 + 97 * (n % 23)) for n in range(60)]
        waves = {c.clip: tone(c.duration_ms * SAMPLE_RATE // 1000) for c in pack_order(clips)}
        pack(tmp_path, "train", waves)
        loaded = load_pack("train", clips, tmp_path)

        for batch in buckets(clips, 5):
            spans = [loaded.spans[c.clip] for c in batch]
            assert len({shard for shard, _, _ in spans}) == 1
            offset = spans[0][1]
            for _, at, length in spans:
                assert at == offset
                offset += length

    def test_pack_order_is_the_order_buckets_groups_in(self) -> None:
        """One key, one function. Two sorts that must agree are the defect this avoids."""
        clips = [clip(f"{n}.mp3", 500 + 311 * (n % 17)) for n in range(80)]
        flat = [c.clip for batch in buckets(clips, 20) for c in batch]
        assert flat == [c.clip for c in pack_order(clips)]

    def test_pack_order_is_stable_against_input_order(self) -> None:
        clips = [clip(f"{n}.mp3", 500 + (n * 37) % 900) for n in range(50)]
        assert pack_order(clips) == pack_order(list(reversed(clips)))


class TestPackIntegrity:
    def test_an_unpacked_clip_is_named_rather_than_reaching_a_loader_thread(
        self, tmp_path: Path
    ) -> None:
        clips = pack(tmp_path, "train", {"a.mp3": tone(64)})
        with pytest.raises(RuntimeError, match="not in the pack"):
            load_pack("train", [*clips, clip("missing.mp3", 1000)], tmp_path)

    def test_nothing_packed_at_all_still_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="modal-asr-repack"):
            load_pack("train", [clip("a.mp3", 1000)], tmp_path)

    def test_a_truncated_shard_is_rejected_not_half_read(self, tmp_path: Path) -> None:
        """A container killed mid-write is the failure this guards. The index declares the
        byte count, so a short `.pcm` is detectable without a checksum of its own."""
        clips = pack(tmp_path, "train", {"a.mp3": tone(256)})
        samples = shard_paths("train", tmp_path)[0].with_suffix(".pcm")
        samples.write_bytes(samples.read_bytes()[:-64])
        assert read_shard(shard_paths("train", tmp_path)[0]) == []
        with pytest.raises(RuntimeError, match="not in the pack"):
            load_pack("train", clips, tmp_path)

    def test_a_shard_whose_samples_vanished_is_rejected(self, tmp_path: Path) -> None:
        pack(tmp_path, "train", {"a.mp3": tone(64)})
        index = shard_paths("train", tmp_path)[0]
        index.with_suffix(".pcm").unlink()
        assert read_shard(index) == []

    def test_unreadable_json_is_rejected_rather_than_raising(self, tmp_path: Path) -> None:
        pack(tmp_path, "train", {"a.mp3": tone(64)})
        index = shard_paths("train", tmp_path)[0]
        index.write_text("{ not json", encoding="utf-8")
        assert read_shard(index) == []

    def test_a_shard_at_the_wrong_sample_rate_is_rejected(self, tmp_path: Path) -> None:
        """The pack carries no audio metadata of its own, so the rate it was written at is
        the only thing standing between a resample change and silently wrong training."""
        pack(tmp_path, "train", {"a.mp3": tone(64)})
        index = shard_paths("train", tmp_path)[0]
        payload = json.loads(index.read_text(encoding="utf-8"))
        payload["sample_rate"] = 8000
        index.write_text(json.dumps(payload), encoding="utf-8")
        assert read_shard(index) == []

    def test_a_partial_write_leaves_no_index_for_the_reader_to_trust(self, tmp_path: Path) -> None:
        """`.partial` files are what a killed writer leaves; `shard_paths` globs `*.json`
        and must not pick them up."""
        (tmp_path / "train").mkdir(parents=True)
        (tmp_path / "train" / "000.json.partial").write_text("{}", encoding="utf-8")
        assert shard_paths("train", tmp_path) == []


class TestNextShard:
    """Numbering from the maximum, not the count. Two concurrent repacks exposed this: a
    count reuses a number the moment any shard in the middle is missing, and reusing a number
    overwrites a good shard."""

    def test_an_empty_pack_starts_at_zero(self, tmp_path: Path) -> None:
        assert next_shard("train", tmp_path) == 0

    def test_it_follows_the_highest_present(self, tmp_path: Path) -> None:
        pack(tmp_path, "train", {"a.mp3": tone(32)}, number=0)
        pack(tmp_path, "train", {"b.mp3": tone(32)}, number=1)
        assert next_shard("train", tmp_path) == 2

    def test_a_hole_does_not_make_it_reuse_a_live_number(self, tmp_path: Path) -> None:
        """000 and 002 present, 001 gone: a count would answer 2 and destroy 002."""
        pack(tmp_path, "train", {"a.mp3": tone(32)}, number=0)
        pack(tmp_path, "train", {"c.mp3": tone(32)}, number=2)
        assert next_shard("train", tmp_path) == 3

    def test_a_rejected_shard_still_holds_its_number(self, tmp_path: Path) -> None:
        pack(tmp_path, "train", {"a.mp3": tone(256)}, number=0)
        samples = shard_paths("train", tmp_path)[0].with_suffix(".pcm")
        samples.write_bytes(samples.read_bytes()[:-64])
        assert read_shard(shard_paths("train", tmp_path)[0]) == []
        assert next_shard("train", tmp_path) == 1

    def test_a_non_numeric_index_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir(parents=True)
        (tmp_path / "train" / "notes.json").write_text("{}", encoding="utf-8")
        assert next_shard("train", tmp_path) == 0

    def test_splits_number_independently(self, tmp_path: Path) -> None:
        pack(tmp_path, "train", {"a.mp3": tone(32)}, number=7)
        assert next_shard("dev", tmp_path) == 0


class TestPackGaps:
    def test_everything_is_a_gap_before_anything_is_packed(self, tmp_path: Path) -> None:
        clips = [clip(f"{n}.mp3", 1000) for n in range(4)]
        assert pack_gaps("train", clips, tmp_path) == clips

    def test_a_packed_clip_is_not_a_gap(self, tmp_path: Path) -> None:
        clips = pack(tmp_path, "train", {"a.mp3": tone(64), "b.mp3": tone(64)})
        assert pack_gaps("train", clips, tmp_path) == []

    def test_only_the_unpacked_remainder_comes_back_in_manifest_order(self, tmp_path: Path) -> None:
        packed = pack(tmp_path, "train", {"a.mp3": tone(64)})
        rest = [clip("b.mp3", 1000), clip("c.mp3", 1000)]
        assert pack_gaps("train", [*packed, *rest], tmp_path) == rest

    def test_a_truncated_shard_puts_its_clips_back_in_the_queue(self, tmp_path: Path) -> None:
        """Resumption and integrity are the same predicate: a shard `read_shard` rejects is
        a shard whose clips must be packed again, and nothing else has to notice."""
        clips = pack(tmp_path, "train", {"a.mp3": tone(256)})
        samples = shard_paths("train", tmp_path)[0].with_suffix(".pcm")
        samples.write_bytes(samples.read_bytes()[:-64])
        assert pack_gaps("train", clips, tmp_path) == clips

    def test_a_split_is_independent_of_its_siblings(self, tmp_path: Path) -> None:
        pack(tmp_path, "train", {"a.mp3": tone(64)})
        assert pack_gaps("dev", [clip("a.mp3", 1000)], tmp_path) == [clip("a.mp3", 1000)]


class TestThroughputBreach:
    """The guard that turns a $14 lesson into a $0.42 one. It bounds spend, not quality."""

    def test_the_measured_warm_rate_does_not_fire(self) -> None:
        assert (
            throughput_breach(
                seconds=EXPECTED_SECONDS_PER_STEP * 20,
                steps=20,
                expected=EXPECTED_SECONDS_PER_STEP,
                factor=FLOOR_FACTOR,
            )
            is None
        )

    def test_the_cold_volume_rate_fires(self) -> None:
        """50.7 s/step is what the cancelled run actually held."""
        breach = throughput_breach(
            seconds=50.7 * 20, steps=20, expected=EXPECTED_SECONDS_PER_STEP, factor=FLOOR_FACTOR
        )
        assert breach == pytest.approx(50.7)

    def test_it_reports_the_rate_it_measured_not_the_threshold(self) -> None:
        breach = throughput_breach(seconds=1000.0, steps=10, expected=1.0, factor=2.0)
        assert breach == pytest.approx(100.0)

    def test_exactly_at_the_threshold_is_allowed(self) -> None:
        assert throughput_breach(seconds=20.0, steps=10, expected=1.0, factor=2.0) is None

    def test_a_zero_factor_disables_it(self) -> None:
        assert throughput_breach(seconds=1e6, steps=1, expected=1.0, factor=0.0) is None

    @pytest.mark.parametrize("steps", [0, -1])
    def test_no_steps_measured_is_not_a_breach(self, steps: int) -> None:
        assert throughput_breach(seconds=1e6, steps=steps, expected=1.0, factor=3.0) is None

    def test_an_unset_expectation_disables_it(self) -> None:
        assert throughput_breach(seconds=1e6, steps=10, expected=0.0, factor=3.0) is None

    def test_a_run_faster_than_expected_never_fires(self) -> None:
        assert (
            throughput_breach(
                seconds=1.0, steps=100, expected=EXPECTED_SECONDS_PER_STEP, factor=1.0
            )
            is None
        )


class TestThroughputGuard:
    """The stateful half: when the clock starts, when it answers, and that it answers once."""

    def test_the_first_observation_only_starts_the_clock(self) -> None:
        """A run's first step carries cudnn autotune, so it is excluded rather than averaged
        in — otherwise one expensive warmup step hides a genuinely slow volume."""
        guard = ThroughputGuard(factor=FLOOR_FACTOR, after=2)
        assert guard.observe() is None
        assert guard.steps == 0
        assert guard.started is not None

    def test_it_does_not_answer_before_the_check_point(self) -> None:
        guard = ThroughputGuard(factor=FLOOR_FACTOR, after=5, expected=1e-9)
        assert [guard.observe() for _ in range(5)] == [None] * 5

    def test_a_slow_loop_breaches_on_the_nth_measured_step(self) -> None:
        guard = ThroughputGuard(factor=1.0, after=3, expected=1e-9)
        guard.observe()
        assert guard.observe() is None
        assert guard.observe() is None
        breach = guard.observe()
        assert breach is not None
        assert guard.breached == breach

    def test_a_fast_loop_never_breaches(self) -> None:
        guard = ThroughputGuard(factor=FLOOR_FACTOR, after=3, expected=1e6)
        for _ in range(10):
            guard.observe()
        assert guard.breached is None

    def test_it_checks_once_and_then_goes_inert(self) -> None:
        """A slowdown after the first minutes is a preemption, not a misconfiguration, and
        the budget it would protect has already been committed."""
        guard = ThroughputGuard(factor=FLOOR_FACTOR, after=2, expected=1e6)
        for _ in range(3):
            guard.observe()
        guard.expected = 1e-9
        assert [guard.observe() for _ in range(20)] == [None] * 20
        assert guard.breached is None

    def test_a_zero_factor_disables_it_end_to_end(self) -> None:
        guard = ThroughputGuard(factor=0.0, after=2, expected=1e-9)
        for _ in range(5):
            guard.observe()
        assert guard.breached is None

    def test_the_default_factor_and_expectation_are_the_module_constants(self) -> None:
        """The loop constructs it with `factor` alone, so the other two must default to the
        figures the docstrings justify, not to whatever a test last passed."""
        guard = ThroughputGuard(factor=FLOOR_FACTOR)
        assert guard.expected == EXPECTED_SECONDS_PER_STEP
        assert guard.after == FLOOR_CHECK_AFTER


class TestAbortedSummary:
    def test_a_healthy_run_carries_no_abort_keys(self) -> None:
        summary = _summary("r", TrainingState(step=10), audio_seconds=0.0, trained=10)
        assert "aborted" not in summary
        assert "seconds_per_step" not in summary

    def test_a_breach_is_visible_in_the_result_a_spawned_call_returns(self) -> None:
        """The result is all anyone sees of a spawned call, and a guard that fired must not
        be indistinguishable from a schedule that finished."""
        summary = _summary(
            "r",
            TrainingState(step=20),
            audio_seconds=0.0,
            trained=20,
            aborted="throughput",
            seconds_per_step=50.712,
        )
        assert summary["aborted"] == "throughput"
        assert summary["seconds_per_step"] == 50.71
