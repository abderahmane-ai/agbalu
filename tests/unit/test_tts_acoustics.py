"""The blind measurements of task 12.3, held to signals whose answers are known.

A profiler cannot be checked against the corpus it profiles — the corpus is the unknown.
Every case here builds a waveform with the property already decided: a lowpass at a stated
frequency, a level ratio of a stated size, a decay of a stated time. What the module must
not do is as important: return zero for something it could not measure, or let background
noise set the band edge.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from agbalu.tts.acoustics import (
    AcousticsError,
    ClipProfile,
    distribution,
    frames,
    profile,
    summarise,
)

RATE: Final = 24_000
TOLERANCE_HZ: Final = 600.0


def band_limited(
    cutoff_hz: float, seconds: float = 1.0, rate: int = RATE, seed: int = 3
) -> NDArray[np.float64]:
    """Noise with every component above `cutoff_hz` removed, built in the spectral domain."""
    generator = np.random.default_rng(seed)
    length = int(seconds * rate)
    spectrum = generator.normal(size=length // 2 + 1) + 1j * generator.normal(size=length // 2 + 1)
    frequencies = np.fft.rfftfreq(length, 1 / rate)
    spectrum[frequencies > cutoff_hz] = 0.0
    wave: NDArray[np.float64] = np.fft.irfft(spectrum, n=length)
    return np.asarray(wave / np.abs(wave).max() * 0.5, dtype=np.float64)


def tilted(
    tilt_db_per_octave: float = -12.0, rate: int = RATE, seed: int = 5
) -> NDArray[np.float64]:
    """Full-band noise falling like a voice does, flat below 100 Hz.

    Flat noise is the one shape speech never has, and it is what let a rolloff pass as a
    band edge: 99% of a tilted spectrum's energy sits far below wherever it ends.
    """
    generator = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(rate, 1 / rate)
    octaves = np.log2(np.maximum(frequencies, 100.0) / 100.0)
    gain = 10 ** (tilt_db_per_octave * octaves / 20)
    spectrum = (
        generator.normal(size=frequencies.size) + 1j * generator.normal(size=frequencies.size)
    ) * gain
    wave: NDArray[np.float64] = np.fft.irfft(spectrum, n=rate)
    return np.asarray(wave / np.abs(wave).max() * 0.5, dtype=np.float64)


def coded(wave: NDArray[np.float64], cutoff_hz: float, rate: int = RATE) -> NDArray[np.float64]:
    """Everything above `cutoff_hz` removed, silence included.

    A codec lowpasses the whole signal, which is why the stop band of a real clip reads
    at the arithmetic floor rather than at the room's noise level. Padding a band-limited
    signal with broadband noise instead leaves no cliff to find — correctly, and it is
    not the case Common Voice presents.
    """
    spectrum = np.fft.rfft(wave)
    spectrum[np.fft.rfftfreq(wave.size, 1 / rate) > cutoff_hz] = 0.0
    limited: NDArray[np.float64] = np.fft.irfft(spectrum, n=wave.size)
    return limited


def gated(
    loud: NDArray[np.float64], ratio_db: float, silence_s: float = 0.3, rate: int = RATE
) -> NDArray[np.float64]:
    """`loud` between two stretches of noise `ratio_db` below it."""
    quiet_rms = float(np.sqrt(np.square(loud).mean())) * 10 ** (-ratio_db / 20)
    generator = np.random.default_rng(11)
    pad = generator.normal(scale=quiet_rms, size=int(silence_s * rate))
    return np.concatenate([pad, loud, pad])


class TestBand:
    @pytest.mark.parametrize("limit", [4_000.0, 6_000.0, 9_000.0])
    def test_a_band_limit_is_found_at_its_own_frequency(self, limit: float) -> None:
        measured = profile(gated(band_limited(limit), 45.0), RATE).rolloff_hz
        assert measured is not None
        assert abs(measured - limit) < TOLERANCE_HZ

    def test_broadband_background_noise_does_not_set_the_answer(self) -> None:
        """The defect this replaced a threshold estimator for: noise 45 dB down sits above
        any fixed spectral floor, and read a 4 kHz limit as 5.5 kHz."""
        quiet = profile(gated(band_limited(4_000.0), 60.0), RATE).rolloff_hz
        noisy = profile(gated(band_limited(4_000.0), 30.0), RATE).rolloff_hz
        assert quiet is not None
        assert noisy is not None
        assert abs(quiet - noisy) < 400.0

    def test_full_band_audio_rolls_off_near_nyquist(self) -> None:
        measured = profile(gated(band_limited(11_999.0), 45.0), RATE)
        assert measured.rolloff_hz is not None
        assert measured.nyquist_hz == 12_000.0
        assert measured.rolloff_hz > 11_000.0

    def test_the_rolloff_is_reported_in_hertz_not_bins(self) -> None:
        wave = gated(band_limited(6_000.0, rate=48_000), 45.0, rate=48_000)
        measured = profile(wave, 48_000).rolloff_hz
        assert measured is not None
        assert abs(measured - 6_000.0) < TOLERANCE_HZ

    def test_the_edge_is_found_on_a_spectrum_that_falls_like_speech(self) -> None:
        """What the corpus caught: two voices with identically empty stop bands measured
        7,000 Hz and 3,140 Hz of rolloff, which is timbre and not bandwidth."""
        measured = profile(coded(gated(tilted(), 45.0), 8_000.0), RATE)
        assert measured.band_edge_hz is not None
        assert abs(measured.band_edge_hz - 8_000.0) < TOLERANCE_HZ
        assert measured.rolloff_hz is not None
        assert measured.rolloff_hz < measured.band_edge_hz - 1_000.0

    def test_tilt_alone_is_not_reported_as_an_edge(self) -> None:
        assert profile(gated(tilted(), 45.0), RATE).band_edge_hz is None

    def test_the_edge_is_the_limit_and_not_the_tilt(self) -> None:
        steep = profile(coded(gated(tilted(-18.0), 45.0), 6_000.0), RATE)
        shallow = profile(coded(gated(tilted(-6.0), 45.0), 6_000.0), RATE)
        assert steep.band_edge_hz is not None
        assert shallow.band_edge_hz is not None
        assert abs(steep.band_edge_hz - shallow.band_edge_hz) < 400.0
        assert steep.rolloff_hz is not None
        assert shallow.rolloff_hz is not None
        assert shallow.rolloff_hz > steep.rolloff_hz + 500.0

    def test_the_top_octave_separates_a_wide_band_from_a_narrow_one(self) -> None:
        narrow = profile(gated(band_limited(4_000.0), 45.0), RATE).top_octave_db
        wide = profile(gated(band_limited(11_999.0), 45.0), RATE).top_octave_db
        assert narrow is not None
        assert wide is not None
        assert narrow < -40.0
        assert wide > -10.0


class TestSpeechToFloor:
    @pytest.mark.parametrize("ratio", [25.0, 40.0, 55.0])
    def test_a_known_level_ratio_is_recovered(self, ratio: float) -> None:
        measured = profile(gated(band_limited(6_000.0), ratio), RATE).speech_to_floor_db
        assert measured is not None
        assert abs(measured - ratio) < 4.0

    def test_a_zeroed_silence_reports_no_floor_rather_than_the_clamp(self) -> None:
        """Voice 1's encoder writes digital zero, so `_decibels` returns its own clamp and
        the median `floor_dbfs` over 19,426 clips came back as exactly -120.0."""
        loud = band_limited(6_000.0, seconds=0.5)
        wave = np.concatenate([np.zeros(int(0.3 * RATE)), loud, np.zeros(int(0.3 * RATE))])
        measured = profile(wave, RATE)
        assert measured.floor_dbfs is None
        assert measured.speech_to_floor_db is None
        assert measured.speech_frame_share > 0.0

    def test_a_real_floor_is_reported_as_a_level(self) -> None:
        measured = profile(gated(band_limited(6_000.0, seconds=0.5), 45.0), RATE)
        assert measured.floor_dbfs is not None
        assert measured.floor_dbfs < 0.0

    def test_the_gate_is_the_same_criterion_at_two_recording_levels(self) -> None:
        """The two voices differ by 13 dB of RMS. A gate defined only against the floor
        would measure their speech shares 43 dB apart, and speech seconds is the number
        this phase's premise rests on."""
        wave = gated(band_limited(6_000.0, seconds=0.5), 45.0)
        loud = profile(wave, RATE).speech_frame_share
        quiet = profile(wave * 10 ** (-13 / 20), RATE).speech_frame_share
        assert loud == pytest.approx(quiet, abs=0.01)

    def test_a_clip_of_pure_noise_measures_no_speech(self) -> None:
        generator = np.random.default_rng(5)
        measured = profile(generator.normal(scale=0.01, size=RATE), RATE)
        assert measured.speech_to_floor_db is None
        assert measured.speech_frame_share == 0.0
        assert measured.rolloff_hz is None


class TestSpeechLevel:
    """`speech_dbfs` is what 12.4 normalises on, so it has to ignore the padding that
    `rms_dbfs` averages in. Common Voice pads both ends of every clip, and on rank 1 the
    padding is 45% of the file."""

    def test_the_speech_level_ignores_the_silence_the_whole_signal_average_includes(
        self,
    ) -> None:
        speech = band_limited(6_000.0, seconds=0.5)
        padded = np.concatenate([np.zeros(int(RATE)), speech, np.zeros(int(RATE))])
        measured = profile(padded, RATE)
        assert measured.speech_dbfs is not None
        assert measured.rms_dbfs is not None
        assert measured.speech_dbfs > measured.rms_dbfs

    def test_padding_a_clip_leaves_the_speech_level_where_it_was(self) -> None:
        speech = gated(band_limited(6_000.0, seconds=0.5), 45.0)
        bare = profile(speech, RATE).speech_dbfs
        padded = profile(np.concatenate([np.zeros(int(0.8 * RATE)), speech]), RATE).speech_dbfs
        assert bare is not None
        assert padded is not None
        assert bare == pytest.approx(padded, abs=1.0)

    def test_halving_the_amplitude_moves_the_speech_level_by_six_decibels(self) -> None:
        speech = gated(band_limited(6_000.0, seconds=0.5), 45.0)
        loud = profile(speech, RATE).speech_dbfs
        quiet = profile(speech * 0.5, RATE).speech_dbfs
        assert loud is not None
        assert quiet is not None
        assert loud - quiet == pytest.approx(6.02, abs=0.2)

    def test_a_clip_with_no_speech_reports_no_level_rather_than_a_floor(self) -> None:
        generator = np.random.default_rng(5)
        assert profile(generator.normal(scale=0.01, size=RATE), RATE).speech_dbfs is None

    def test_digital_silence_reports_no_speech_level(self) -> None:
        assert profile(np.zeros(RATE), RATE).speech_dbfs is None

    def test_the_level_is_carried_into_the_record_and_the_summary(self) -> None:
        measured = profile(gated(band_limited(6_000.0, seconds=0.5), 45.0), RATE)
        assert measured.as_dict()["speech_dbfs"] == pytest.approx(measured.speech_dbfs, abs=1e-3)
        fields = summarise([measured])["fields"]
        assert isinstance(fields, dict)
        assert "speech_dbfs" in fields


class TestDecay:
    def test_a_known_decay_time_is_recovered(self) -> None:
        rate, seconds = RATE, 1.5
        generator = np.random.default_rng(7)
        tone = generator.normal(size=int(seconds * rate)) * 0.4
        target = 0.5
        times = np.arange(tone.size) / rate
        envelope = np.where(times < 0.5, 1.0, 10 ** (-3 * (times - 0.5) / target))
        wave = np.concatenate([tone * envelope, generator.normal(scale=1e-5, size=rate // 4)])
        measured = profile(wave, rate)
        assert measured.decay_offsets >= 1
        assert measured.decay_time_s is not None
        assert abs(measured.decay_time_s - target) < 0.15

    def test_no_offset_leaves_the_estimate_absent_rather_than_zero(self) -> None:
        measured = profile(band_limited(6_000.0), RATE)
        assert measured.decay_offsets == 0
        assert measured.decay_time_s is None


class TestClipping:
    def test_a_flat_top_is_a_run(self) -> None:
        wave = band_limited(6_000.0).copy()
        wave[100:106] = 1.0
        assert profile(wave, RATE).clipped_runs == 1

    def test_a_single_full_scale_sample_is_not_clipping(self) -> None:
        wave = band_limited(6_000.0).copy()
        wave[100] = 1.0
        measured = profile(wave, RATE)
        assert measured.clipped_runs == 0
        assert measured.clipped_sample_share == 0.0

    def test_the_share_is_over_samples(self) -> None:
        wave = np.zeros(1_000)
        wave[:100] = 1.0
        measured = profile(wave, RATE)
        assert measured.clipped_runs == 1
        assert measured.clipped_sample_share == pytest.approx(0.1)


class TestSilenceMargins:
    def test_the_margins_are_measured_in_milliseconds(self) -> None:
        loud = band_limited(6_000.0, seconds=0.5)
        generator = np.random.default_rng(2)
        quiet = float(np.sqrt(np.square(loud).mean())) * 10 ** (-45 / 20)
        wave = np.concatenate(
            [
                generator.normal(scale=quiet, size=int(0.4 * RATE)),
                loud,
                generator.normal(scale=quiet, size=int(0.2 * RATE)),
            ]
        )
        measured = profile(wave, RATE)
        assert measured.lead_silence_ms is not None
        assert measured.trail_silence_ms is not None
        assert abs(measured.lead_silence_ms - 400.0) < 40.0
        assert abs(measured.trail_silence_ms - 200.0) < 40.0


class TestExtremes:
    def test_an_empty_waveform_is_refused(self) -> None:
        with pytest.raises(AcousticsError, match="empty"):
            profile(np.zeros(0), RATE)

    def test_a_non_finite_sample_is_refused(self) -> None:
        wave = band_limited(6_000.0).copy()
        wave[10] = np.nan
        with pytest.raises(AcousticsError, match="non-finite"):
            profile(wave, RATE)

    def test_a_zero_sample_rate_is_refused(self) -> None:
        with pytest.raises(AcousticsError, match="sample rate"):
            profile(band_limited(6_000.0), 0)

    def test_digital_silence_measures_nothing_rather_than_zero(self) -> None:
        measured = profile(np.zeros(RATE), RATE)
        assert measured.rms_dbfs is None
        assert measured.rolloff_hz is None
        assert measured.speech_to_floor_db is None
        assert measured.peak == 0.0

    def test_a_clip_shorter_than_one_frame_still_profiles(self) -> None:
        measured = profile(np.full(8, 0.2), RATE)
        assert measured.samples == 8
        assert measured.speech_frame_share == 0.0
        assert measured.rms_dbfs is not None

    def test_a_single_sample_is_admissible(self) -> None:
        measured = profile(np.array([0.5]), RATE)
        assert measured.samples == 1
        assert measured.peak == 0.5

    def test_a_dc_offset_is_reported_with_its_sign(self) -> None:
        wave = band_limited(6_000.0)
        shifted = profile(wave + 0.1, RATE).dc_offset - profile(wave, RATE).dc_offset
        assert shifted == pytest.approx(0.1, abs=1e-9)
        assert profile(wave - 0.1, RATE).dc_offset < 0.0

    def test_float32_input_is_accepted(self) -> None:
        wave = band_limited(6_000.0).astype(np.float32)
        assert profile(wave, RATE).samples == wave.size

    def test_frames_of_a_short_signal_are_empty_not_ragged(self) -> None:
        assert frames(np.zeros(10), 25, 10).shape == (0, 25)

    def test_a_non_positive_hop_is_refused(self) -> None:
        with pytest.raises(AcousticsError, match="positive"):
            frames(np.zeros(100), 25, 0)


def stub(
    *,
    sample_rate: int = RATE,
    decay_time_s: float | None = 0.3,
    clipped_runs: int = 0,
    samples: int = RATE,
    speech_frame_share: float = 0.8,
) -> ClipProfile:
    """A measured clip. `decay_offsets` follows `decay_time_s`, which is the module's own
    invariant: no offset was fitted exactly when there is no estimate."""
    return ClipProfile(
        samples=samples,
        sample_rate=sample_rate,
        peak=0.5,
        rms_dbfs=-20.0,
        speech_dbfs=-17.0,
        dc_offset=0.0,
        rolloff_hz=7_000.0,
        band_edge_hz=8_000.0,
        top_octave_db=-45.0,
        floor_dbfs=-70.0,
        speech_to_floor_db=30.0,
        speech_frame_share=speech_frame_share,
        lead_silence_ms=100.0,
        trail_silence_ms=100.0,
        decay_time_s=decay_time_s,
        decay_offsets=0 if decay_time_s is None else 2,
        clipped_runs=clipped_runs,
        clipped_sample_share=0.0,
    )


class TestSummary:
    def test_the_denominator_is_the_clips_that_measured_the_field(self) -> None:
        profiles = [stub(), stub(decay_time_s=None), stub(decay_time_s=0.5)]
        summary = summarise(profiles)
        fields = summary["fields"]
        assert isinstance(fields, dict)
        assert fields["decay_time_s"]["measured"] == 2
        assert fields["rolloff_hz"]["measured"] == 3
        assert summary["clips_without_decay_estimate"] == 1

    def test_speech_seconds_weights_by_duration_not_by_clip(self) -> None:
        """A mean of the per-clip shares would call this 0.55; the long clip is silent."""
        profiles = [
            stub(samples=RATE, speech_frame_share=1.0),
            stub(samples=9 * RATE, speech_frame_share=0.1),
        ]
        summary = summarise(profiles)
        assert summary["seconds"] == 10.0
        assert summary["speech_seconds"] == 1.9

    def test_a_field_no_clip_measured_has_no_median(self) -> None:
        assert distribution([None, None]) == {
            "measured": 0,
            "p10": None,
            "median": None,
            "p90": None,
        }

    def test_quantiles_ignore_the_absent_values(self) -> None:
        assert distribution([None, 1.0, 3.0])["median"] == 2.0

    def test_zeroed_silence_is_counted_rather_than_averaged(self) -> None:
        loud = band_limited(6_000.0, seconds=0.5)
        silent = np.concatenate([np.zeros(int(0.3 * RATE)), loud, np.zeros(int(0.3 * RATE))])
        summary = summarise([profile(silent, RATE), profile(gated(loud, 45.0), RATE)])
        assert summary["clips_with_zeroed_silence"] == 1
        fields = summary["fields"]
        assert isinstance(fields, dict)
        assert fields["floor_dbfs"]["measured"] == 1

    def test_mixed_sample_rates_are_reported_not_collapsed(self) -> None:
        summary = summarise([stub(), stub(sample_rate=48_000)])
        assert summary["sample_rates"] == {"24000": 1, "48000": 1}

    def test_a_mixed_rate_voice_is_summarised_per_rate(self) -> None:
        """Voice 2 is 11,751 clips at 32 kHz and 4,251 at 48 kHz, which is two setups."""
        summary = summarise([stub(), stub(), stub(sample_rate=48_000)])
        per_rate = summary["by_sample_rate"]
        assert isinstance(per_rate, dict)
        assert per_rate["24000"]["clips"] == 2
        assert per_rate["48000"]["clips"] == 1
        assert "by_sample_rate" not in per_rate["24000"]

    def test_a_single_rate_voice_carries_no_split(self) -> None:
        assert "by_sample_rate" not in summarise([stub(), stub()])

    def test_an_empty_voice_is_refused(self) -> None:
        with pytest.raises(AcousticsError, match="no clips"):
            summarise([])

    def test_clipping_is_counted_per_clip(self) -> None:
        summary = summarise([stub(), stub(clipped_runs=4), stub(clipped_runs=1)])
        assert summary["clips_with_clipping"] == 2
