"""The restoration path's contracts, which are all about length and independence.

Two properties decide whether the restored arm is comparable with the raw one. A clip
must come back exactly as long as its duration implies, or every trim bound computed on
the raw audio lands somewhere else; and one clip's features must not depend on another's,
or a batch boundary becomes an audible edit. The upstream script truncates over-long
input with a warning, so the refusal is asserted rather than the truncation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
import torch

from agbalu.tts.restore import (
    MEL_BINS,
    OUTPUT_RATE,
    SOURCE_RATE,
    STRIDE,
    RestoreError,
    Restorer,
    _fit,
    download,
    filterbank,
    resample,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

RATE: Final = 48_000


def tone(seconds: float, rate: int = RATE, hertz: float = 220.0) -> NDArray[np.float32]:
    times = np.arange(round(seconds * rate), dtype=np.float32) / rate
    return np.ascontiguousarray(0.4 * np.sin(2 * np.pi * hertz * times), dtype=np.float32)


def fake_restorer(*, chunk_seconds: float = 11.0, output_samples: int | None = None) -> Restorer:
    """A `Restorer` whose traced modules are replaced by shape-preserving stand-ins.

    The checkpoints are 400 MB of TorchScript that only a container downloads, and what
    is under test here is the padding, the refusal and the length restoration around
    them, not the vocoder.
    """
    built = object.__new__(Restorer)
    built._device = torch.device("cpu")
    built._chunk_samples = round(chunk_seconds * SOURCE_RATE)

    def _features(features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"last_hidden_state": features}

    def _decoder(hidden: torch.Tensor) -> torch.Tensor:
        samples = (
            output_samples if output_samples is not None else round(chunk_seconds * OUTPUT_RATE)
        )
        return torch.zeros(int(hidden.shape[0]), 1, samples)

    built._features = _features
    built._decoder = _decoder
    return built


class TestResample:
    def test_the_same_rate_is_a_copy_not_a_conversion(self) -> None:
        audio = tone(0.1)
        assert np.array_equal(resample(audio, RATE, RATE), audio)

    def test_halving_the_rate_halves_the_length(self) -> None:
        audio = tone(1.0, rate=48_000)
        assert len(resample(audio, 48_000, 24_000)) == pytest.approx(len(audio) / 2, abs=2)

    def test_the_result_is_contiguous_float32(self) -> None:
        converted = resample(tone(0.1), RATE, SOURCE_RATE)
        assert converted.dtype == np.float32
        assert converted.flags["C_CONTIGUOUS"]

    def test_an_empty_waveform_stays_empty(self) -> None:
        assert resample(np.zeros(0, dtype=np.float32), RATE, RATE).size == 0


class TestFilterbank:
    def test_the_features_fold_two_frames_into_one_row(self) -> None:
        batch = torch.from_numpy(np.stack([tone(1.0, rate=SOURCE_RATE) for _ in range(3)]))
        features = filterbank(batch)
        assert features.shape[0] == 3
        assert features.shape[2] == MEL_BINS * STRIDE

    def test_one_clips_gain_does_not_move_another_clips_features(self) -> None:
        """Normalisation is per clip. Over the batch, a loud neighbour would rescale
        every other row and a batch boundary would become an audible edit."""
        quiet = tone(0.5, rate=SOURCE_RATE)
        alone = filterbank(torch.from_numpy(np.stack([quiet, quiet])))
        beside = filterbank(torch.from_numpy(np.stack([quiet, quiet * np.float32(20.0)])))
        assert torch.allclose(alone[0], beside[0], atol=1e-5)

    def test_each_clip_is_centred_over_its_own_time_axis(self) -> None:
        batch = torch.from_numpy(
            np.stack([tone(0.5, rate=SOURCE_RATE), tone(0.5, rate=SOURCE_RATE, hertz=440.0)])
        )
        features = filterbank(batch)
        assert features[0].mean().abs() < 0.05
        assert features[1].mean().abs() < 0.05

    def test_the_front_end_is_not_scale_invariant_so_gain_is_applied_after_restoration(
        self,
    ) -> None:
        """`energy_floor` is absolute, so scaling a clip moves which mel bins clamp. It is
        why levelling happens on the restorer's output and never on its input: pre-gained
        audio would restore differently from the raw arm's own source."""
        audio = tone(0.5, rate=SOURCE_RATE)
        loud = filterbank(torch.from_numpy(np.stack([audio * np.float32(4.0)])))
        quiet = filterbank(torch.from_numpy(np.stack([audio * np.float32(0.05)])))
        assert not torch.allclose(loud[0], quiet[0], atol=1e-2)

    def test_a_one_dimensional_tensor_raises(self) -> None:
        with pytest.raises(RestoreError, match="batch, samples"):
            filterbank(torch.from_numpy(tone(0.5, rate=SOURCE_RATE)))

    def test_a_three_dimensional_tensor_raises(self) -> None:
        with pytest.raises(RestoreError, match="batch, samples"):
            filterbank(torch.zeros(2, 1, 16_000))


class TestPrepare:
    def test_a_clip_prepared_alone_gives_the_rows_it_gives_inside_a_batch(self) -> None:
        """What licenses moving the filterbank onto the loader threads.

        The normalisation is per clip over its own time axis, so preparing each clip
        separately has to reproduce the batch exactly — bit for bit, not approximately.
        Were it over the batch, this would be the test that said so.
        """
        restorer = fake_restorer()
        waves = [tone(1.0), tone(2.5, hertz=330.0), tone(0.75)]
        alone = torch.stack([restorer.prepare(wave, RATE)[0] for wave in waves])

        chunk = round(restorer.chunk_seconds * SOURCE_RATE)
        padded = torch.zeros(len(waves), chunk, dtype=torch.float32)
        for index, wave in enumerate(waves):
            narrowed = resample(wave, RATE, SOURCE_RATE)
            padded[index, : narrowed.size] = torch.from_numpy(narrowed)

        assert torch.equal(alone, filterbank(padded))

    def test_the_reported_length_is_the_duration_not_the_chunk(self) -> None:
        _, samples = fake_restorer().prepare(tone(3.0), RATE)
        assert samples == 3 * OUTPUT_RATE

    def test_an_over_long_clip_raises_here_rather_than_at_the_device(self) -> None:
        with pytest.raises(RestoreError, match="rather than truncating"):
            fake_restorer(chunk_seconds=2.0).prepare(tone(5.0), RATE)


class TestRestorePrepared:
    def test_mismatched_features_and_lengths_raise(self) -> None:
        restorer = fake_restorer()
        rows, samples = restorer.prepare(tone(1.0), RATE)
        with pytest.raises(RestoreError, match="against"):
            restorer.restore_prepared([rows], [samples, samples])

    def test_an_empty_batch_touches_no_model(self) -> None:
        assert fake_restorer().restore_prepared([], []) == []


class TestFit:
    def test_a_short_output_is_zero_extended(self) -> None:
        assert _fit(torch.ones(10), 16).tolist() == [1.0] * 10 + [0.0] * 6

    def test_a_long_output_is_trimmed(self) -> None:
        assert _fit(torch.ones(20), 4).tolist() == [1.0] * 4

    def test_an_exact_output_is_returned_unchanged(self) -> None:
        exact = torch.ones(7)
        assert _fit(exact, 7) is exact

    def test_a_zero_target_returns_nothing(self) -> None:
        assert _fit(torch.ones(5), 0).numel() == 0


class TestDownload:
    def test_an_unknown_device_raises_before_any_network_call(self) -> None:
        with pytest.raises(RestoreError, match="no Sidon trace"):
            download("mps")


def restore(
    restorer: Restorer, waveforms: Sequence[NDArray[np.float32]], rates: Sequence[int]
) -> list[NDArray[np.float32]]:
    """The two halves composed the way `modal_app.tts` composes them, one thread apart."""
    prepared = [restorer.prepare(audio, rate) for audio, rate in zip(waveforms, rates, strict=True)]
    return restorer.restore_prepared(
        [rows for rows, _ in prepared], [samples for _, samples in prepared]
    )


class TestRestorer:
    def test_a_non_positive_chunk_raises_before_the_checkpoints_are_read(self) -> None:
        with pytest.raises(RestoreError, match="positive"):
            Restorer(Path("absent.pt"), Path("absent.pt"), torch.device("cpu"), chunk_seconds=0.0)

    def test_the_chunk_is_reported_in_seconds(self) -> None:
        assert fake_restorer(chunk_seconds=11.0).chunk_seconds == pytest.approx(11.0)

    def test_a_clip_comes_back_at_the_length_its_duration_implies(self) -> None:
        audio = tone(3.0)
        (restored,) = restore(fake_restorer(), [audio], [RATE])
        assert restored.size == round(audio.size / RATE * OUTPUT_RATE)

    def test_a_short_vocoder_output_is_padded_to_that_length(self) -> None:
        (restored,) = restore(fake_restorer(output_samples=100), [tone(2.0)], [RATE])
        assert restored.size == 2 * OUTPUT_RATE
        assert restored[-1] == 0.0

    def test_every_clip_of_a_batch_keeps_its_own_length(self) -> None:
        waves = [tone(1.0), tone(2.5), tone(0.75)]
        restored = restore(fake_restorer(), waves, [RATE] * 3)
        assert [wave.size for wave in restored] == [
            round(wave.size / RATE * OUTPUT_RATE) for wave in waves
        ]

    def test_a_non_positive_rate_raises(self) -> None:
        with pytest.raises(RestoreError, match="sample rate"):
            fake_restorer().prepare(tone(1.0), 0)

    def test_the_output_is_contiguous_float32(self) -> None:
        (restored,) = restore(fake_restorer(), [tone(1.0)], [RATE])
        assert restored.dtype == np.float32
        assert restored.flags["C_CONTIGUOUS"]
