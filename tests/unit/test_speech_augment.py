"""Unit tests for acoustic augmentation operations."""

from __future__ import annotations

import torch

from agbalu.speech.augment import (
    AcousticAugmenter,
    add_noise,
    apply_telephony_filter,
    compute_audio_energy,
    convolve_rir,
    spec_augment,
)


class TestAcousticAugmentations:
    def test_compute_audio_energy(self) -> None:
        wav = torch.ones(1000)
        energy = compute_audio_energy(wav)
        assert abs(energy - 1.0) < 1e-4

    def test_add_noise_preserves_shape(self) -> None:
        wav = torch.randn(16000)
        noise = torch.randn(8000)  # Shorter noise repeated
        mixed = add_noise(wav, noise, snr_db=10.0)
        assert mixed.shape == wav.shape
        assert not torch.allclose(mixed, wav)

    def test_convolve_rir_preserves_shape(self) -> None:
        wav = torch.randn(16000)
        rir = torch.randn(1600)  # 100ms RIR
        reverbed = convolve_rir(wav, rir)
        assert reverbed.shape == wav.shape

    def test_apply_telephony_filter_preserves_shape(self) -> None:
        wav = torch.randn(16000)
        filtered = apply_telephony_filter(wav, sample_rate=16000)
        assert filtered.shape == wav.shape

    def test_spec_augment_masks_features(self) -> None:
        # Mel features: batch=2, time=50, freq=80
        features = torch.ones(2, 50, 80)
        masked = spec_augment(
            features,
            freq_mask_param=10,
            time_mask_param=0.1,
            num_freq_masks=2,
            num_time_masks=2,
        )
        assert masked.shape == features.shape
        # Some values should be zeroed out
        assert torch.sum(masked == 0.0) > 0

    def test_augmenter_clean_mode(self) -> None:
        augmenter = AcousticAugmenter(p_clean=1.0)
        wav = torch.randn(16000)
        out = augmenter.augment(wav)
        assert torch.allclose(out, wav)
