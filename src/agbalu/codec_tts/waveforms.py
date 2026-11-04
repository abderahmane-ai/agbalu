"""Convert waveforms without changing their declared duration."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

Waveform = NDArray[np.float32]
CLIPPING_THRESHOLD = 0.999


@dataclass(frozen=True)
class Signal:
    samples: int
    rate_hz: int
    peak: float
    rms_dbfs: float | None
    clipping_fraction: float


def text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def validate_waveform(waveform: Waveform, rate_hz: int) -> None:
    if waveform.ndim != 1 or waveform.size == 0 or rate_hz <= 0:
        message = f"invalid waveform shape={waveform.shape}, rate_hz={rate_hz}"
        raise ValueError(message)
    if not np.isfinite(waveform).all():
        offset = int(np.flatnonzero(~np.isfinite(waveform))[0])
        message = f"non-finite waveform value={waveform[offset]} at sample={offset}"
        raise ValueError(message)


def measure_signal(waveform: Waveform, rate_hz: int) -> Signal:
    validate_waveform(waveform, rate_hz)
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    return Signal(
        waveform.size,
        rate_hz,
        float(np.abs(waveform).max()),
        20 * math.log10(rms) if rms > 0 else None,
        float(np.mean(np.abs(waveform) >= CLIPPING_THRESHOLD)),
    )


def resample_waveform(waveform: Waveform, rate_hz: int, target_rate_hz: int = 24000) -> Waveform:
    validate_waveform(waveform, rate_hz)
    if target_rate_hz <= 0:
        message = f"invalid target_rate_hz={target_rate_hz}"
        raise ValueError(message)
    divisor = math.gcd(rate_hz, target_rate_hz)
    converted = np.asarray(
        resample_poly(
            waveform, target_rate_hz // divisor, rate_hz // divisor, window=("kaiser", 8.6)
        ),
        dtype=np.float32,
    )
    expected = (waveform.size * target_rate_hz + rate_hz - 1) // rate_hz
    if converted.size != expected:
        message = f"resampling returned {converted.size} samples, expected {expected}"
        raise ValueError(message)
    validate_waveform(converted, target_rate_hz)
    return converted
