"""Detect aliasing with pass-band and stop-band controls."""

import numpy as np

from agbalu.codec_tts.waveforms import resample_waveform


def test_filter_rejects_known_alias() -> None:
    time_s = np.arange(48000) / 48000
    high = np.sin(2 * np.pi * 16000 * time_s).astype(np.float32)
    low = np.sin(2 * np.pi * 1000 * time_s).astype(np.float32)
    filtered = resample_waveform(high, 48000)[128:-128]
    positive_control = high[::2][128:-128]
    passed = resample_waveform(low, 48000)[128:-128]
    assert np.sqrt(np.mean(filtered**2)) < 0.001
    assert np.sqrt(np.mean(positive_control**2)) > 0.6
    assert np.sqrt(np.mean(passed**2)) > 0.6
