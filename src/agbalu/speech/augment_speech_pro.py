"""Fadhma-Pro: Online Acoustic Augmentation Pipeline (Native PyTorch Resampling & Masking)."""

from __future__ import annotations

import random
from typing import Final

import torch

NOISE_PROBABILITY: Final[float] = 0.3
SPEED_PROBABILITY: Final[float] = 0.5
AUDIO_1D: Final[int] = 1
AUDIO_2D: Final[int] = 2


class SpeechAugmentationPro:
    """Acoustic augmentation for robust ASR training using 100% pure PyTorch."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        speed_perturbation: bool = True,
        spec_augment: bool = True,
        noise_snr_db: tuple[float, float] = (15.0, 30.0),
    ) -> None:
        self.sample_rate = sample_rate
        self.speed_perturbation = speed_perturbation
        self.spec_augment = spec_augment
        self.noise_snr_db = noise_snr_db

    def apply_speed_perturbation(self, waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
        """Randomly perturb speed (0.9x, 1.0x, 1.1x) using pure PyTorch linear interpolation."""
        factor = rng.choice([0.9, 1.0, 1.1])
        if factor == 1.0:
            return waveform

        orig_ndim = waveform.ndim
        if orig_ndim == AUDIO_1D:
            inp = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif orig_ndim == AUDIO_2D:
            inp = waveform.unsqueeze(0)  # (1, C, T)
        else:
            inp = waveform

        out = torch.nn.functional.interpolate(
            inp, scale_factor=factor, mode="linear", align_corners=False
        )

        if orig_ndim == AUDIO_1D:
            return out.squeeze(0).squeeze(0)
        if orig_ndim == AUDIO_2D:
            return out.squeeze(0)
        return out

    def apply_noise(self, waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
        """Add subtle white/pink noise at specified SNR."""
        if rng.random() < NOISE_PROBABILITY:
            snr_db = rng.uniform(self.noise_snr_db[0], self.noise_snr_db[1])
            signal_power: torch.Tensor = waveform.norm(p=2)
            noise: torch.Tensor = torch.randn_like(waveform)
            noise_power: torch.Tensor = noise.norm(p=2)
            if noise_power > 0 and signal_power > 0:
                scale: torch.Tensor = signal_power / (noise_power * (10.0 ** (snr_db / 20.0)))
                scaled: torch.Tensor = scale * noise
                return waveform + scaled
        return waveform

    def apply_spec_augment(
        self, features: torch.Tensor, max_time_mask: int = 20, max_freq_mask: int = 15
    ) -> torch.Tensor:
        """Apply SpecAugment masking directly on 2D/3D feature representations."""
        if not self.spec_augment:
            return features

        out = features.clone()
        t_len = out.shape[-2]
        t_mask = random.randint(0, min(max_time_mask, t_len // 4))  # noqa: S311
        if t_mask > 0 and t_len > t_mask:
            t0 = random.randint(0, t_len - t_mask)  # noqa: S311
            out[..., t0 : t0 + t_mask, :] = 0.0

        f_len = out.shape[-1]
        f_mask = random.randint(0, min(max_freq_mask, f_len // 4))  # noqa: S311
        if f_mask > 0 and f_len > f_mask:
            f0 = random.randint(0, f_len - f_mask)  # noqa: S311
            out[..., f0 : f0 + f_mask] = 0.0

        return out

    def __call__(self, waveform: torch.Tensor, rng: random.Random | None = None) -> torch.Tensor:
        """Apply acoustic augmentations to raw 16kHz audio tensor (1, T) or (T,)."""
        r = rng or random.Random()  # noqa: S311
        out = waveform
        if self.speed_perturbation and r.random() < SPEED_PROBABILITY:
            out = self.apply_speed_perturbation(out, r)
        return self.apply_noise(out, r)
