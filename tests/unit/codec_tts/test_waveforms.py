"""Verify signal measurements and exact resampling lengths."""

import math

import numpy as np
import pytest

from agbalu.codec_tts.waveforms import measure_signal, resample_waveform


@pytest.mark.parametrize("rate_hz", [16000, 24000, 32000, 44100, 48000])
@pytest.mark.parametrize("samples", [1, 13, 1001])
def test_exact_converted_size(rate_hz: int, samples: int) -> None:
    result = resample_waveform(np.ones(samples, dtype=np.float32), rate_hz)
    assert result.size == math.ceil(samples * 24000 / rate_hz)
    assert np.isfinite(result).all()


def test_silence_is_not_a_fabricated_noise_floor() -> None:
    signal = measure_signal(np.zeros(4, dtype=np.float32), 24000)
    assert signal.rms_dbfs is None
    assert signal.peak == signal.clipping_fraction == 0


def test_level_and_clip_measurement() -> None:
    signal = measure_signal(np.array([1, -1, 0, 0], dtype=np.float32), 24000)
    assert signal.peak == 1
    assert signal.clipping_fraction == 0.5
    assert signal.rms_dbfs == pytest.approx(-3.0102999566)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_invalid_samples(values: list[float]) -> None:
    with pytest.raises(ValueError, match="waveform"):
        resample_waveform(np.array(values, dtype=np.float32), 48000)


def test_invalid_shape_and_rate() -> None:
    with pytest.raises(ValueError, match="shape"):
        resample_waveform(np.ones((2, 2), dtype=np.float32), 24000)
    with pytest.raises(ValueError, match="target_rate_hz=0"):
        resample_waveform(np.ones(4, dtype=np.float32), 48000, 0)
