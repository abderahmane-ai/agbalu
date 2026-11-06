"""Verify non-overlapping video timing and signal-derived player bars."""

import numpy as np
import pytest
from tools.speech_showcase import Word
from tools.video_showcase import caption_groups, ease, parse_loudness, schedule, waveform_bars


def test_four_audio_clips_play_sequentially_with_transition_space() -> None:
    intervals = schedule([2, 3, 4, 5])
    np.testing.assert_allclose(intervals, [(1.2, 3.2), (3.55, 6.55), (6.9, 10.9), (11.25, 16.25)])
    assert sum(end - start for start, end in intervals) == pytest.approx(14)


@pytest.mark.parametrize("durations", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_audio_durations_are_rejected(durations: list[float]) -> None:
    with pytest.raises(ValueError, match="invalid clip durations"):
        schedule(durations)


def test_bars_measure_real_energy_instead_of_random_animation() -> None:
    wave = np.repeat(np.asarray([0.1, 0.5, 1.0], dtype=np.float32), 100)
    bars = waveform_bars(wave, 3)
    np.testing.assert_allclose(bars, np.asarray([0.1, 0.5, 1]) ** 0.65, rtol=1e-6)


def test_silent_and_short_audio_fail_instead_of_fake_bars() -> None:
    with pytest.raises(ValueError, match="silent"):
        waveform_bars(np.zeros(300, dtype=np.float32))
    with pytest.raises(ValueError, match="invalid waveform"):
        waveform_bars(np.ones(3, dtype=np.float32))


def test_transition_easing_has_exact_endpoints() -> None:
    assert [ease(t) for t in [-1, 0, 0.5, 1, 2]] == [0, 0, 0.5, 1, 1]


def test_captions_preserve_words_and_sentence_boundaries() -> None:
    text = "Azul fell-awen! Taqbaylit d tayri, d timucuha, d tudert."
    words = tuple(Word(text=w, start_s=i, end_s=i + 1) for i, w in enumerate(text.split()))
    groups = caption_groups(words)
    assert [" ".join(w.text for w in group) for group in groups] == [
        "Azul fell-awen!",
        "Taqbaylit d tayri, d timucuha, d tudert.",
    ]
    assert tuple(w for group in groups for w in group) == words


def test_loudness_report_allows_ffmpeg_trailing_encoder_statistics() -> None:
    log = '[Parsed_loudnorm] {"input_i":"-18.3","output_i":"-16.0"}\n[out] muxing overhead\n'
    assert parse_loudness(log) == {"input_i": "-18.3", "output_i": "-16.0"}
    with pytest.raises(ValueError, match="no loudness report"):
        parse_loudness("no audio output")
