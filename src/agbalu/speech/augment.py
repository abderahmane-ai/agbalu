"""Multi-condition acoustic augmentation pipeline for Fadhma 2.

Implements online training augmentations:
1. Clean baseline (30%)
2. Room impulse response (RIR) reverberation convolution (20%)
3. Additive environmental noise mixing from MUSAN (30%, SNR 0-30 dB)
4. Telephony bandpass & codec degradation (10%)
5. Combined acoustic stress (10%)
6. SpecAugment (frequency and time masking on mel representations)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import soundfile as sf
import torch
import torch.nn.functional as f

DEFAULT_SAMPLE_RATE: Final[int] = 16000
TELEPHONY_LOW_HZ: Final[float] = 300.0
TELEPHONY_HIGH_HZ: Final[float] = 3400.0
DEFAULT_P_CLEAN: Final[float] = 0.30
DEFAULT_P_REVERB: Final[float] = 0.20
DEFAULT_P_NOISE: Final[float] = 0.30
DEFAULT_P_TELEPHONY: Final[float] = 0.10
DEFAULT_P_COMBINED: Final[float] = 0.10


def compute_audio_energy(waveform: torch.Tensor) -> float:
    """Compute root mean square energy of a 1D or 2D audio tensor."""
    return float(torch.sqrt(torch.mean(waveform**2) + 1e-9).item())


def add_noise(waveform: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix additive noise into waveform at a specified SNR in dB.

    waveform: Tensor of shape (channels, samples) or (samples,)
    noise: Tensor of shape matching waveform (or trimmed/repeated to match)
    snr_db: Signal-to-noise ratio in decibels (e.g. 0.0 to 30.0)
    """
    if waveform.numel() == 0 or noise.numel() == 0:
        return waveform

    orig_shape = waveform.shape
    wav_flat = waveform.view(-1)
    noise_flat = noise.view(-1)

    if noise_flat.size(0) < wav_flat.size(0):
        repeats = math.ceil(wav_flat.size(0) / noise_flat.size(0))
        noise_flat = noise_flat.repeat(repeats)[: wav_flat.size(0)]
    elif noise_flat.size(0) > wav_flat.size(0):
        max_start = noise_flat.size(0) - wav_flat.size(0)
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        noise_flat = noise_flat[start : start + wav_flat.size(0)]

    signal_energy = torch.mean(wav_flat**2) + 1e-9
    noise_energy = torch.mean(noise_flat**2) + 1e-9

    target_noise_energy = signal_energy / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(target_noise_energy / noise_energy)
    mixed = wav_flat + (scale * noise_flat)

    max_val = torch.max(torch.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed.view(orig_shape)


def convolve_rir(
    waveform: torch.Tensor, rir: torch.Tensor, max_rir_len: int = 16000
) -> torch.Tensor:
    """Convolve waveform with a room impulse response (RIR) using FFT.

    Normalizes RIR energy and preserves original waveform length. Trims RIR to
    max_rir_len (default 1.0s at 16kHz) and performs O(N log N) FFT convolution,
    avoiding catastrophic O(N * K) direct time-domain convolution latency.
    """
    if waveform.numel() == 0 or rir.numel() == 0:
        return waveform

    orig_shape = waveform.shape
    wav = waveform.view(-1)
    rir_trim = rir.view(-1)[:max_rir_len]
    rir_norm = rir_trim / (torch.norm(rir_trim, p=2) + 1e-9)

    n = wav.size(0) + rir_norm.size(0) - 1
    wav_fft = torch.fft.rfft(wav, n=n)
    rir_fft = torch.fft.rfft(rir_norm, n=n)
    convolved = torch.fft.irfft(wav_fft * rir_fft, n=n)
    result = convolved[: wav.size(0)]

    max_val = torch.max(torch.abs(result))
    if max_val > 1.0:
        result = result / max_val
    res: torch.Tensor = result.view(orig_shape)
    return res


def apply_telephony_filter(
    waveform: torch.Tensor, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> torch.Tensor:
    """Simulate G.711 / telephony bandpass filtering (300 Hz - 3400 Hz) using FFT."""
    if waveform.numel() == 0:
        return waveform

    orig_shape = waveform.shape
    wav = waveform.view(-1)
    fft_vals = torch.fft.rfft(wav)
    freqs = torch.fft.rfftfreq(wav.size(0), 1.0 / sample_rate)

    mask = (freqs >= TELEPHONY_LOW_HZ) & (freqs <= TELEPHONY_HIGH_HZ)
    fft_vals = fft_vals * mask.to(fft_vals.device)
    filtered = torch.fft.irfft(fft_vals, n=wav.size(0))

    result: torch.Tensor = filtered.view(orig_shape)
    return result


def spec_augment(
    mel_features: torch.Tensor,
    *,
    freq_mask_param: int = 27,
    time_mask_param: float = 0.05,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """Apply SpecAugment frequency and time masking to 2D/3D mel features.

    mel_features: shape (batch, time, freq) or (time, freq)
    """
    features = mel_features.clone()
    is_2d = len(features.shape) == 2
    if is_2d:
        features = features.unsqueeze(0)

    _batch_size, time_steps, num_freqs = features.shape

    for b in range(_batch_size):
        for _ in range(num_freq_masks):
            max_f = min(freq_mask_param, num_freqs)
            f_size = int(torch.randint(0, max_f + 1, (1,)).item())
            if f_size > 0:
                f_start = int(torch.randint(0, num_freqs - f_size + 1, (1,)).item())
                features[b, :, f_start : f_start + f_size] = 0.0

        max_time_mask = int(time_steps * time_mask_param)
        for _ in range(num_time_masks):
            t_size = int(torch.randint(0, max(1, max_time_mask) + 1, (1,)).item())
            if t_size > 0 and time_steps > t_size:
                t_start = int(torch.randint(0, time_steps - t_size + 1, (1,)).item())
                features[b, t_start : t_start + t_size, :] = 0.0

    if is_2d:
        features = features.squeeze(0)
    return features


class AcousticAugmenter:
    """Manages multi-condition sampling and online augmentation during training."""

    def __init__(
        self,
        rir_tensors: Sequence[torch.Tensor] | None = None,
        noise_tensors: Sequence[torch.Tensor] | None = None,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        p_clean: float = DEFAULT_P_CLEAN,
    ) -> None:
        self.rirs = list(rir_tensors) if rir_tensors else []
        self.noises = list(noise_tensors) if noise_tensors else []
        self.sample_rate = sample_rate
        self.p_clean = p_clean

    @classmethod
    def from_directories(
        cls,
        rir_dir: Path | None = None,
        noise_dir: Path | None = None,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_rirs: int = 150,
        max_noises: int = 150,
    ) -> AcousticAugmenter:
        """Create augmenter scanning RIR and noise audio files if available."""
        rirs: list[torch.Tensor] = []
        noises: list[torch.Tensor] = []

        if rir_dir and rir_dir.is_dir():
            wav_paths = sorted(rir_dir.rglob("*.wav"))
            stride = max(1, len(wav_paths) // max_rirs)
            for path in wav_paths[::stride][:max_rirs]:
                try:
                    data, sr = sf.read(str(path), dtype="float32")
                    tensor = torch.from_numpy(data)
                    if tensor.ndim > 1:
                        tensor = torch.mean(tensor, dim=-1)
                    if sr != sample_rate:
                        num_samples = int(tensor.size(0) * (sample_rate / sr))
                        tensor = f.interpolate(
                            tensor.view(1, 1, -1),
                            size=num_samples,
                            mode="linear",
                            align_corners=False,
                        ).view(-1)
                    # Limit RIR to 1.0s (16,000 samples at 16kHz) to keep memory & convolution light
                    rirs.append(tensor[:sample_rate])
                except (OSError, RuntimeError):
                    pass

        if noise_dir and noise_dir.is_dir():
            wav_paths = sorted(noise_dir.rglob("*.wav"))
            stride = max(1, len(wav_paths) // max_noises)
            for path in wav_paths[::stride][:max_noises]:
                try:
                    data, sr = sf.read(str(path), dtype="float32")
                    tensor = torch.from_numpy(data)
                    if tensor.ndim > 1:
                        tensor = torch.mean(tensor, dim=-1)
                    if sr != sample_rate:
                        num_samples = int(tensor.size(0) * (sample_rate / sr))
                        tensor = f.interpolate(
                            tensor.view(1, 1, -1),
                            size=num_samples,
                            mode="linear",
                            align_corners=False,
                        ).view(-1)
                    noises.append(tensor)
                except (OSError, RuntimeError):
                    pass

        return cls(rir_tensors=rirs, noise_tensors=noises, sample_rate=sample_rate)

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply a randomly sampled acoustic condition to the audio waveform."""
        roll = float(torch.rand(1).item())

        if roll < self.p_clean:
            return waveform

        if roll < 0.50:
            if self.rirs:
                rir = self.rirs[int(torch.randint(0, len(self.rirs), (1,)).item())]
                return convolve_rir(waveform, rir)
            return waveform

        if roll < 0.80:
            if self.noises:
                noise = self.noises[int(torch.randint(0, len(self.noises), (1,)).item())]
                snr = float(torch.empty(1).uniform_(0.0, 30.0).item())
                return add_noise(waveform, noise, snr_db=snr)
            return waveform

        if roll < 0.90:
            return apply_telephony_filter(waveform, self.sample_rate)

        # Combined stress
        out = waveform
        if self.rirs:
            out = convolve_rir(out, self.rirs[int(torch.randint(0, len(self.rirs), (1,)).item())])
        if self.noises:
            noise_idx = int(torch.randint(0, len(self.noises), (1,)).item())
            out = add_noise(
                out, self.noises[noise_idx], snr_db=float(torch.empty(1).uniform_(5.0, 25.0).item())
            )
        return apply_telephony_filter(out, self.sample_rate)
