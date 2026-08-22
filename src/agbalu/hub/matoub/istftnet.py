"""The iSTFTNet decoder Matoub-82M inherits from Kokoro-82M.

Adapted from `hexgrad/Kokoro-82M`'s `istftnet.py` (Apache-2.0), itself adapted from
StyleTTS2's `Modules/istftnet.py` (MIT). Module attribute names are the published
checkpoint's `state_dict` keys; renaming one breaks `from_pretrained` for everybody who
downloaded the release.

Two deliberate departures from both upstreams. Weight normalisation is fused into the
weight by the exporter, so the convolutions here are plain — a released model has no
parametrisation to re-derive. And `InstanceNorm1d` is built without affine parameters, as
the training code had it: Kokoro turns them on to work around an ONNX export bug, which
adds tensors this checkpoint does not carry.

The harmonic source draws noise and an initial phase from the global RNG, so two calls on
the same text return different samples. That is inherited from HN-NSF, not a defect.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class AdaIN1d(nn.Module):
    """Instance norm whose scale and shift are read off the style vector."""

    def __init__(self, style_dim: int, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        h = self.fc(s).unsqueeze(-1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        normalised: Tensor = self.norm(x)
        return (1 + gamma) * normalised + beta


class AdaINResBlock1(nn.Module):
    """HiFi-GAN's residual block with AdaIN conditioning and a Snake nonlinearity."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
        style_dim: int = 64,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=_padding(kernel_size, d),
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1, padding=_padding(kernel_size)
                )
                for _ in dilation
            ]
        )
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilation])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilation])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation])

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        blocks = zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
            strict=True,
        )
        for conv1, conv2, norm1, norm2, alpha1, alpha2 in blocks:
            xt = norm1(x, s)
            xt = xt + (1 / alpha1) * (torch.sin(alpha1 * xt) ** 2)
            xt = conv1(xt)
            xt = norm2(xt, s)
            xt = xt + (1 / alpha2) * (torch.sin(alpha2 * xt) ** 2)
            x = conv2(xt) + x
        return x


class TorchSTFT(nn.Module):
    """The short-time transform the generator inverts to reach the waveform."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

    def _window(self, like: Tensor) -> Tensor:
        # Built per call rather than held: it is twenty derived floats, it is deliberately
        # absent from the checkpoint, and anything allocated in `__init__` is allocated on
        # the meta device that `from_pretrained` constructs the model under.
        return torch.hann_window(
            self.win_length, periodic=True, dtype=torch.float32, device=like.device
        )

    def transform(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        spectrum = torch.stft(
            waveform,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self._window(waveform),
            return_complex=True,
        )
        return torch.abs(spectrum), torch.angle(spectrum)

    def inverse(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        waveform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self._window(magnitude),
        )
        return waveform.unsqueeze(-2)


class SineGen(nn.Module):
    """Harmonic excitation for the F0 contour, one sine per overtone."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def _sine(self, f0_values: Tensor) -> Tensor:
        radians = (f0_values / self.sampling_rate) % 1
        initial_phase = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        initial_phase[:, 0] = 0
        radians[:, 0, :] = radians[:, 0, :] + initial_phase
        radians = F.interpolate(
            radians.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        phase = torch.cumsum(radians, dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: Tensor) -> tuple[Tensor, Tensor]:
        overtones = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)
        sine_waves = self._sine(f0 * overtones) * self.sine_amp
        voiced = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amplitude = voiced * self.noise_std + (1 - voiced) * self.sine_amp / 3
        noise = noise_amplitude * torch.randn_like(sine_waves)
        return sine_waves * voiced + noise, voiced


class SourceModuleHnNSF(nn.Module):
    """Harmonic-plus-noise source, merged to one excitation channel."""

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0: Tensor) -> Tensor:
        with torch.no_grad():
            sine_waves, _ = self.l_sin_gen(f0)
        merged: Tensor = self.l_tanh(self.l_linear(sine_waves))
        return merged


class Generator(nn.Module):
    """Upsampling stack ending in an inverse short-time transform."""

    def __init__(
        self,
        style_dim: int,
        resblock_kernel_sizes: tuple[int, ...],
        upsample_rates: tuple[int, ...],
        upsample_initial_channel: int,
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        upsample_kernel_sizes: tuple[int, ...],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        sampling_rate: int,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        scale = math.prod(upsample_rates) * gen_istft_hop_size
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=scale,
            harmonic_num=8,
            voiced_threshold=10,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=scale)
        self.ups = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True))
            ]
        )
        self.resblocks = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        channels = upsample_initial_channel
        for i in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            shapes = zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True)
            for kernel, dilation in shapes:
                self.resblocks.append(AdaINResBlock1(channels, kernel, dilation, style_dim))
            if i + 1 < len(upsample_rates):
                stride = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        channels,
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=(stride + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(channels, 7, (1, 3, 5), style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, channels, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(channels, 11, (1, 3, 5), style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = nn.Conv1d(channels, self.post_n_fft + 2, 7, 1, padding=3)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def forward(self, x: Tensor, s: Tensor, f0_curve: Tensor) -> Tensor:
        with torch.no_grad():
            f0 = self.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
            harmonic = self.m_source(f0).transpose(1, 2).squeeze(1)
            magnitude, phase = self.stft.transform(harmonic)
            source = torch.cat([magnitude, phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            excitation = self.noise_res[i](self.noise_convs[i](source), s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + excitation
            stacked = self.resblocks[i * self.num_kernels](x, s)
            for j in range(1, self.num_kernels):
                stacked = stacked + self.resblocks[i * self.num_kernels + j](x, s)
            x = stacked / self.num_kernels

        x = self.conv_post(F.leaky_relu(x))
        spectrum = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spectrum, phase)


class AdainResBlk1d(nn.Module):
    """The decoder's conditioning block, optionally doubling the frame rate."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        *,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample_type = "timepreserve" if upsample else "none"
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
        if upsample:
            self.pool: nn.Module = nn.ConvTranspose1d(
                dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1
            )
        else:
            self.pool = nn.Identity()

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.upsample_type != "none":
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if not self.learned_sc:
            return x
        projected: Tensor = self.conv1x1(x)
        return projected

    def _residual(self, x: Tensor, s: Tensor) -> Tensor:
        x = self.conv1(self.pool(self.actv(self.norm1(x, s))))
        out: Tensor = self.conv2(self.actv(self.norm2(x, s)))
        return out

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        combined: Tensor = self._residual(x, s) + self._shortcut(x)
        scaled: Tensor = combined * (2.0**-0.5)
        return scaled


class Decoder(nn.Module):
    """Aligned text features, pitch and energy to a 24 kHz waveform."""

    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        resblock_kernel_sizes: tuple[int, ...],
        upsample_rates: tuple[int, ...],
        upsample_initial_channel: int,
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        upsample_kernel_sizes: tuple[int, ...],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        sampling_rate: int,
    ) -> None:
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
            ]
        )
        self.F0_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.N_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.asr_res = nn.Sequential(nn.Conv1d(512, 64, kernel_size=1))
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
            sampling_rate,
        )

    def forward(self, asr: Tensor, pitch: Tensor, energy: Tensor, s: Tensor) -> Tensor:
        f0 = self.F0_conv(pitch.unsqueeze(1))
        n = self.N_conv(energy.unsqueeze(1))
        x = self.encode(torch.cat([asr, f0, n], dim=1), s)
        residual = self.asr_res(asr)
        carry = True
        for block in self.decode:
            if carry:
                x = torch.cat([x, residual, f0, n], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                carry = False
        waveform: Tensor = self.generator(x, s, pitch)
        return waveform


__all__ = ["AdaIN1d", "AdainResBlk1d", "Decoder"]
