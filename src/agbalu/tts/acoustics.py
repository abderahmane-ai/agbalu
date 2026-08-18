"""What a recording is, measured from the waveform: task 12.3's instrument.

Every figure here is blind — there is no clean reference and no impulse response — so each
is named for what it computes rather than for the quantity it approximates.
`speech_to_floor_db` is the active speech level over the tenth-percentile frame level, not
SNR; `decay_time_s` extrapolates a fitted offset slope to 60 dB, and is not RT60.

Two consequences, both load-bearing when the numbers are read:

- A clip with no silence has no frames to estimate a floor from, so `speech_to_floor_db`
  is a *lower* bound there. `lead_silence_ms` beside it is what says whether that is so.
- A decay reaching the noise floor before it has fallen `MIN_DECAY_DB` is discarded rather
  than fitted, so `decay_offsets` can be zero and `decay_time_s` `None`. Anything not
  measured is `None` and never a floor value: a clamp reads as a measurement.

Nothing is resampled. The cutoff is exactly the property a resample destroys — the ASR
pack is 16 kHz, so its Nyquist is 8 kHz and it cannot answer this question at any
threshold.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FRAME_MS: Final = 25.0
HOP_MS: Final = 10.0

GATE_DB: Final = 20.0
"""Decibels above the estimated floor at which a frame counts as speech."""

SPEECH_RANGE_DB: Final = 35.0
"""The other half of the gate: decibels below the clip's own 95th-percentile frame level.

A gate defined only against the floor is not the same criterion in two clips. Common Voice
voice 1 has its silence zeroed by the encoder, so its floor is the arithmetic clamp and
`floor + GATE_DB` lands at -100 dBFS, where voice 2 carries real room tone and the same
expression lands at -57 — thresholds 43 dB apart deciding what counts as speech, and speech
seconds is the number this phase's premise rests on. Whichever gate is higher wins, so a
noisy clip is still measured against its noise and a gated one against its speech."""

NOISE_PERCENTILE: Final = 10.0
"""The frame level taken as the floor. A tenth of a Common Voice clip is silence in the
usual case and the percentile degrades into a low speech frame when it is not, which is
why the result is reported as a floor ratio and not as SNR."""

ROLLOFF_SHARE: Final = 0.99
"""Share of the speech-band energy below `rolloff_hz`. This is a property of the voice,
not of the encoding: speech falls about 12 dB an octave, so 99% of its energy sits far
below wherever the band actually ends. Two Common Voice speakers with identically empty
stop bands measured 7,000 Hz and 3,140 Hz. Read it as timbre; `band_edge_hz` is the
bandwidth."""

EDGE_SPAN_HZ: Final = 500.0
EDGE_DROP_DB: Final = 30.0
SMOOTH_HZ: Final = 100.0
"""A codec's lowpass is a cliff and a voice's tilt is a slope, so the edge is found as
the largest drop across `EDGE_SPAN_HZ` rather than by any absolute threshold. Below
`EDGE_DROP_DB` there is no edge and the answer is `None` — which is the honest reading
of audio that was never band-limited."""

CLIP_LEVEL: Final = 0.99
MIN_CLIP_RUN: Final = 3
"""One sample at full scale is a peak; three consecutive are a flat top. mp3 decodes
overshoot unity, so the run length is what distinguishes clipping from a loud sample."""

DECAY_WINDOW_MS: Final = 200.0
MIN_DECAY_DB: Final = 10.0
MIN_DECAY_FRAMES: Final = 4
DECAY_DEPTH_DB: Final = 30.0

EPSILON: Final = 1e-12
CLAMP_DB: Final = -119.0
"""Just above `10·log10(EPSILON)`. A frame of digital silence reports the clamp, and a
level equal to it is the arithmetic showing through rather than a measurement."""


class AcousticsError(Exception):
    """Audio that cannot be profiled."""


@dataclass(frozen=True, slots=True)
class ClipProfile:
    """One clip's measurements. `None` marks what this clip could not establish."""

    samples: int
    sample_rate: int
    peak: float
    rms_dbfs: float | None
    speech_dbfs: float | None
    dc_offset: float
    rolloff_hz: float | None
    band_edge_hz: float | None
    top_octave_db: float | None
    floor_dbfs: float | None
    speech_to_floor_db: float | None
    speech_frame_share: float
    lead_silence_ms: float | None
    trail_silence_ms: float | None
    decay_time_s: float | None
    decay_offsets: int
    clipped_runs: int
    clipped_sample_share: float

    @property
    def duration_s(self) -> float:
        return self.samples / self.sample_rate

    @property
    def nyquist_hz(self) -> float:
        """What `rolloff_hz` is read against: the same rolloff means opposite things at
        16 kHz and at 48 kHz."""
        return self.sample_rate / 2

    def as_dict(self) -> dict[str, object]:
        def rounded(value: float | None, places: int) -> float | None:
            return None if value is None else round(value, places)

        return {
            "samples": self.samples,
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 4),
            "peak": round(self.peak, 6),
            "rms_dbfs": rounded(self.rms_dbfs, 3),
            "speech_dbfs": rounded(self.speech_dbfs, 3),
            "dc_offset": round(self.dc_offset, 8),
            "rolloff_hz": rounded(self.rolloff_hz, 1),
            "band_edge_hz": rounded(self.band_edge_hz, 1),
            "top_octave_db": rounded(self.top_octave_db, 2),
            "nyquist_hz": self.nyquist_hz,
            "floor_dbfs": rounded(self.floor_dbfs, 3),
            "speech_to_floor_db": rounded(self.speech_to_floor_db, 3),
            "speech_frame_share": round(self.speech_frame_share, 5),
            "lead_silence_ms": rounded(self.lead_silence_ms, 1),
            "trail_silence_ms": rounded(self.trail_silence_ms, 1),
            "decay_time_s": rounded(self.decay_time_s, 4),
            "decay_offsets": self.decay_offsets,
            "clipped_runs": self.clipped_runs,
            "clipped_sample_share": round(self.clipped_sample_share, 8),
        }


SUMMARY_FIELDS: Final[Mapping[str, Callable[[ClipProfile], float | None]]] = {
    "duration_s": lambda clip: clip.duration_s,
    "peak": lambda clip: clip.peak,
    "rms_dbfs": lambda clip: clip.rms_dbfs,
    "speech_dbfs": lambda clip: clip.speech_dbfs,
    "dc_offset": lambda clip: clip.dc_offset,
    "rolloff_hz": lambda clip: clip.rolloff_hz,
    "band_edge_hz": lambda clip: clip.band_edge_hz,
    "top_octave_db": lambda clip: clip.top_octave_db,
    "floor_dbfs": lambda clip: clip.floor_dbfs,
    "speech_to_floor_db": lambda clip: clip.speech_to_floor_db,
    "speech_frame_share": lambda clip: clip.speech_frame_share,
    "lead_silence_ms": lambda clip: clip.lead_silence_ms,
    "trail_silence_ms": lambda clip: clip.trail_silence_ms,
    "decay_time_s": lambda clip: clip.decay_time_s,
}


def _decibels(power: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(10.0 * np.log10(np.maximum(power, EPSILON)), dtype=np.float64)


def distribution(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Quantiles over the clips that measured this field, with that count beside them.

    `measured` is the denominator and is never the clip count: a field `None` on nine
    tenths of a voice has a median, and reading it without the count reads it as the
    voice's.
    """
    present = [value for value in values if value is not None]
    if not present:
        return {"measured": 0, "p10": None, "median": None, "p90": None}
    quantiles = np.percentile(np.asarray(present, dtype=np.float64), (10, 50, 90))
    return {
        "measured": len(present),
        "p10": round(float(quantiles[0]), 4),
        "median": round(float(quantiles[1]), 4),
        "p90": round(float(quantiles[2]), 4),
    }


def summarise(profiles: Sequence[ClipProfile], *, split_rates: bool = True) -> dict[str, object]:
    """One voice's distributions, and the counts a reader needs to weigh them.

    A voice carrying more than one sample rate is summarised again per rate, because it is
    then two recording setups and not one: `top_octave_db` is defined against each file's
    own Nyquist, so its median over a mixed voice averages two different bands, and whether
    the subsets share a band edge is what decides if they can be trained as one voice.
    """
    if not profiles:
        message = "no clips to summarise"
        raise AcousticsError(message)
    rates = Counter(clip.sample_rate for clip in profiles)
    by_rate = (
        {
            str(rate): summarise(
                [clip for clip in profiles if clip.sample_rate == rate], split_rates=False
            )
            for rate in sorted(rates)
        }
        if split_rates and len(rates) > 1
        else None
    )
    return {
        "clips": len(profiles),
        "seconds": round(sum(clip.duration_s for clip in profiles), 2),
        # Duration-weighted, never the mean of the per-clip shares: the shares belong to
        # clips of different lengths, and what a voice can supervise is seconds of speech.
        # Common Voice pads both ends, so this is well below the audio total.
        "speech_seconds": round(
            sum(clip.duration_s * clip.speech_frame_share for clip in profiles), 2
        ),
        "sample_rates": {str(rate): count for rate, count in sorted(rates.items())},
        **({"by_sample_rate": by_rate} if by_rate else {}),
        "clips_with_clipping": sum(1 for clip in profiles if clip.clipped_runs),
        "clips_without_decay_estimate": sum(1 for clip in profiles if not clip.decay_offsets),
        # No level was representable in this clip's quietest frames, so its silence was
        # written as digital zero. A property of the encoder, and the reason `floor_dbfs`
        # is absent rather than -120.
        "clips_with_zeroed_silence": sum(
            1 for clip in profiles if clip.floor_dbfs is None and clip.speech_frame_share
        ),
        "fields": {
            name: distribution([read(clip) for clip in profiles])
            for name, read in SUMMARY_FIELDS.items()
        },
    }


def frames(audio: NDArray[np.float64], length: int, hop: int) -> NDArray[np.float64]:
    """Overlapping frames. Empty when the signal is shorter than one frame."""
    if length < 1 or hop < 1:
        message = f"frame length and hop must be positive, got {length} and {hop}"
        raise AcousticsError(message)
    if audio.size < length:
        return np.empty((0, length), dtype=np.float64)
    windows: NDArray[np.float64] = np.lib.stride_tricks.sliding_window_view(audio, length)
    return windows[::hop]


def _clipped(audio: NDArray[np.float64]) -> tuple[int, float]:
    """Runs of `MIN_CLIP_RUN` or more consecutive samples at or above `CLIP_LEVEL`."""
    hot = np.abs(audio) >= CLIP_LEVEL
    if not bool(hot.any()):
        return 0, 0.0
    padded = np.concatenate(([False], hot, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[0::2]
    long_runs = lengths[lengths >= MIN_CLIP_RUN]
    return int(long_runs.size), float(long_runs.sum()) / float(audio.size)


def _band(spectrum: NDArray[np.float64], sample_rate: int, length: int) -> tuple[float, float]:
    """Where the energy stops, and how far down the octave below Nyquist sits.

    The pair is what separates a genuinely wide band from an upsampled narrow one: a file
    stored at 24 kHz but synthesised at 16 kHz keeps its rolloff near 7 kHz and its top
    octave 60 dB or more below the spectral peak.
    """
    cumulative = np.cumsum(spectrum)
    index = int(np.searchsorted(cumulative, ROLLOFF_SHARE * float(cumulative[-1])))
    rolloff = float(min(index, spectrum.size - 1)) * sample_rate / length
    top = spectrum[spectrum.size // 2 :]
    top_db = float(_decibels(top.mean(keepdims=True))[0])
    peak_db = float(_decibels(spectrum.max(keepdims=True))[0])
    return rolloff, top_db - peak_db


def _band_edge(spectrum: NDArray[np.float64], sample_rate: int, length: int) -> float | None:
    """The frequency of the sharpest fall in the spectrum, when there is one.

    A lowpass drops tens of decibels within a few hundred hertz and a voice's own tilt
    does not, so the discriminator is the size of the step and not its height above any
    floor. Returns `None` when nothing falls by `EDGE_DROP_DB`, which is what audio that
    was never band-limited should report.

    The highest edge it can return is the last position `EDGE_SPAN_HZ` fits into — 11,500 Hz
    at 24 kHz — so a distribution whose quantiles all sit there is censored at the
    instrument and reads as "at or above", never as a measured band edge.
    """
    per_bin = sample_rate / length
    smooth = max(round(SMOOTH_HZ / per_bin), 1)
    span = max(round(EDGE_SPAN_HZ / per_bin), 1)
    kernel = np.full(smooth, 1.0 / smooth)
    smoothed = np.convolve(_decibels(spectrum), kernel, mode="valid")
    if smoothed.size <= span:
        return None
    drops = smoothed[:-span] - smoothed[span:]
    # The highest such fall, not the largest: a rumble filter puts a cliff at the bottom
    # of the spectrum too, and the band edge is by definition the last one.
    cliffs = np.flatnonzero(drops >= EDGE_DROP_DB)
    if cliffs.size == 0:
        return None
    return float(int(cliffs[-1]) + (smooth - 1) / 2) * per_bin


def _decay(
    levels: NDArray[np.float64], speech: NDArray[np.bool_], hop_s: float
) -> tuple[float | None, int]:
    """Median 60 dB extrapolation of the log-energy slope after each speech offset.

    Each fit starts at the last speech frame and stops once the level has fallen
    `DECAY_DEPTH_DB`, so both ends are referenced to the decay itself. Truncating against
    the noise floor instead makes the fit a function of the encoder: a clip whose silence
    is digital zero never lands, and one truncated just below the gate yields two frames,
    because an offset is *defined* by crossing that gate.
    """
    if speech.size < MIN_DECAY_FRAMES:
        return None, 0
    offsets = np.flatnonzero(speech[:-1] & ~speech[1:])
    window = max(round(DECAY_WINDOW_MS / 1000.0 / hop_s), MIN_DECAY_FRAMES)
    slopes: list[float] = []

    for start in offsets:
        segment = levels[start : start + window]
        landed = np.flatnonzero(segment <= segment[0] - DECAY_DEPTH_DB)
        segment = segment[: int(landed[0]) + 1] if landed.size else segment
        if segment.size < MIN_DECAY_FRAMES or segment[0] - segment[-1] < MIN_DECAY_DB:
            continue
        times = np.arange(segment.size, dtype=np.float64) * hop_s
        slope = float(np.polyfit(times, segment, 1)[0])
        if slope < 0.0:
            slopes.append(slope)

    if not slopes:
        return None, 0
    return 60.0 / abs(float(np.median(slopes))), len(slopes)


def profile(audio: NDArray[np.float32] | NDArray[np.float64], sample_rate: int) -> ClipProfile:
    """Measure one clip at its own rate. The waveform is never resampled or filtered."""
    if sample_rate < 1:
        message = f"sample rate must be positive, got {sample_rate}"
        raise AcousticsError(message)
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        message = "empty waveform"
        raise AcousticsError(message)
    if not bool(np.isfinite(signal).all()):
        message = "waveform carries non-finite samples"
        raise AcousticsError(message)

    length = max(round(FRAME_MS / 1000.0 * sample_rate), 2)
    hop = max(round(HOP_MS / 1000.0 * sample_rate), 1)
    hop_s = hop / sample_rate

    windows = frames(signal, length, hop)
    power = np.square(windows).mean(axis=1) if windows.size else np.empty(0, dtype=np.float64)
    levels = _decibels(power)

    rolloff: float | None = None
    edge: float | None = None
    top_octave: float | None = None
    noise_floor: float | None = None
    speech_level: float | None = None
    speech_to_floor: float | None = None
    speech_share = 0.0
    lead: float | None = None
    trail: float | None = None
    decay_time: float | None = None
    decay_offsets = 0

    if levels.size:
        floor = float(np.percentile(levels, NOISE_PERCENTILE))
        measured_floor = floor > CLAMP_DB
        gate = max(floor + GATE_DB, float(np.percentile(levels, 95.0)) - SPEECH_RANGE_DB)
        speech = levels >= gate
        voiced = np.flatnonzero(speech)
        if voiced.size:
            taper = np.hanning(length)
            spectrum = np.square(np.abs(np.fft.rfft(windows[speech] * taper, axis=1))).mean(axis=0)
            active = float(_decibels(power[speech].mean(keepdims=True))[0])
            speech_level = active
            rolloff, top_octave = _band(spectrum, sample_rate, length)
            edge = _band_edge(spectrum, sample_rate, length)
            # The floor is a level only where one was representable. An encoder that writes
            # digital zero leaves `_decibels` reporting its own clamp, and a ratio against
            # a clamp is arithmetic rather than a measurement.
            noise_floor = floor if measured_floor else None
            speech_to_floor = active - floor if measured_floor else None
            speech_share = float(voiced.size) / float(levels.size)
            lead = float(voiced[0]) * hop_s * 1000.0
            trail = float(levels.size - 1 - voiced[-1]) * hop_s * 1000.0
            decay_time, decay_offsets = _decay(levels, speech, hop_s)

    mean_power = float(np.square(signal).mean())
    runs, clipped_share = _clipped(signal)

    return ClipProfile(
        samples=int(signal.size),
        sample_rate=sample_rate,
        peak=float(np.abs(signal).max()),
        rms_dbfs=None if mean_power <= EPSILON else float(10.0 * np.log10(mean_power)),
        speech_dbfs=speech_level,
        dc_offset=float(signal.mean()),
        rolloff_hz=rolloff,
        band_edge_hz=edge,
        top_octave_db=top_octave,
        floor_dbfs=noise_floor,
        speech_to_floor_db=speech_to_floor,
        speech_frame_share=speech_share,
        lead_silence_ms=lead,
        trail_silence_ms=trail,
        decay_time_s=decay_time,
        decay_offsets=decay_offsets,
        clipped_runs=runs,
        clipped_sample_share=clipped_share,
    )
