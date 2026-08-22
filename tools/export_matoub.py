"""Turn a Matoub Stage 2 training checkpoint into a directory `transformers` can load.

The training checkpoint is 1.93 GB of optimizer moments, discriminators, an aligner, a
pitch extractor and an untrained diffusion sampler around 82M parameters of model. This
writes the model, with weight normalisation fused into the weights, and the speaker style
baked in as a 256-dim buffer so a downloader needs no reference clip and no StyleTTS2
checkout.

The style vector is read off a reference clip through the checkpoint's own
`style_encoder` and `predictor_encoder` — the two modules that exist only to produce it,
and are therefore not shipped. The clip must be a *restored* corpus clip from the voice
being exported, on the volume at `tts/voices/<voice>/restored/`, because the filterbank
below is the one `meldataset` trains on and a vector extracted on any other describes a
speaker the model has never heard.

    python3 -m tools.export_matoub \
        --checkpoint artifacts/matoub/epoch_2nd_00003.pth \
        --reference artifacts/matoub/reference_kab_male.wav \
        --out artifacts/release/Matoub-82M
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import torch
    from torch import Tensor

INFERENCE_MODULES: Final = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
"""What a release carries. `style_encoder`, `predictor_encoder`, `diffusion`,
`text_aligner`, `pitch_extractor`, `mpd`, `msd` and `wd` are training instruments."""

STYLE_MODULES: Final = ("style_encoder", "predictor_encoder")
"""Acoustic style first, prosodic second — the order the decoder and the predictor slice."""

MEL_PARAMS: Final = {"n_fft": 2048, "win_length": 1200, "hop_length": 300, "n_mels": 80}
"""`meldataset`'s filterbank. `torchaudio` defaults `sample_rate` to 16 kHz and therefore
`f_max` to 8 kHz, which is what the recipe trains on and what the style vector must match."""

MEL_MEAN: Final = -4.0
MEL_STD: Final = 4.0
TRIM_TOP_DB: Final = 30.0
SAMPLE_RATE: Final = 24_000


def _style_encoder(dim_in: int, style_dim: int, max_conv_dim: int) -> torch.nn.Module:
    """StyleTTS2's `StyleEncoder`, at the commit `modal_app.common` pins.

    Vendored rather than imported: the recipe is cloned inside the Modal image and this
    runs on a laptop. Only the shapes and the spectral-norm parametrisation matter — the
    weights come from the checkpoint.
    """
    import torch
    from torch import nn
    from torch.nn.utils.parametrizations import spectral_norm

    class LearnedDownSample(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.conv = spectral_norm(
                nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(2, 2), groups=dim, padding=1)
            )

        def forward(self, x: Tensor) -> Tensor:
            out: Tensor = self.conv(x)
            return out

    class ResBlk(nn.Module):
        def __init__(self, dim_in: int, dim_out: int) -> None:
            super().__init__()
            self.actv = nn.LeakyReLU(0.2)
            self.downsample_res = LearnedDownSample(dim_in)
            self.learned_sc = dim_in != dim_out
            self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
            self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
            if self.learned_sc:
                self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

        def _downsample(self, x: Tensor) -> Tensor:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return nn.functional.avg_pool2d(x, 2)

        def forward(self, x: Tensor) -> Tensor:
            shortcut: Tensor = self._downsample(self.conv1x1(x) if self.learned_sc else x)
            residual: Tensor = self.downsample_res(self.conv1(self.actv(x)))
            residual = self.conv2(self.actv(residual))
            return (shortcut + residual) / math.sqrt(2)

    class StyleEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[nn.Module] = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
            channels = dim_in
            for _ in range(4):
                out = min(channels * 2, max_conv_dim)
                blocks.append(ResBlk(channels, out))
                channels = out
            blocks.append(nn.LeakyReLU(0.2))
            blocks.append(spectral_norm(nn.Conv2d(channels, channels, 5, 1, 0)))
            blocks.append(nn.AdaptiveAvgPool2d(1))
            blocks.append(nn.LeakyReLU(0.2))
            self.shared = nn.Sequential(*blocks)
            self.unshared = nn.Linear(channels, style_dim)

        def forward(self, x: Tensor) -> Tensor:
            h: Tensor = self.shared(x)
            style: Tensor = self.unshared(h.view(h.size(0), -1))
            return style

    return StyleEncoder()


def strip_module_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def fuse_weight_norm(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Collapse `parametrizations.weight.original0/1` back into a plain `weight`.

    A release has no parametrisation to re-derive, and shipping one makes the class that
    loads it carry `weight_norm` calls whose only effect at inference is to recompute a
    constant.
    """
    marker = "parametrizations.weight.original"
    fused: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.endswith(f"{marker}0"):
            prefix = key[: -len(f"{marker}0")]
            direction = state[f"{prefix}{marker}1"]
            dims = [d for d in range(direction.dim()) if d != 0]
            norm = direction.norm(2, dim=dims, keepdim=True)
            fused[f"{prefix}weight"] = direction * (value / norm)
        elif not key.endswith(f"{marker}1"):
            fused[key] = value
    return fused


def style_vector(net: Mapping[str, Mapping[str, Tensor]], reference: Path) -> Tensor:
    """The 256-dim speaker style of one restored clip, acoustic half first."""
    import librosa
    import torch
    import torchaudio

    wave, _ = librosa.load(str(reference), sr=SAMPLE_RATE)
    trimmed, _ = librosa.effects.trim(wave, top_db=TRIM_TOP_DB)
    if trimmed.size == 0:
        message = f"{reference} is silence after trimming at {TRIM_TOP_DB} dB"
        raise ValueError(message)

    to_mel = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
    mel = to_mel(torch.from_numpy(trimmed).float().unsqueeze(0))
    mel = (torch.log(1e-5 + mel) - MEL_MEAN) / MEL_STD

    halves: list[Tensor] = []
    for name in STYLE_MODULES:
        encoder = _style_encoder(dim_in=64, style_dim=128, max_conv_dim=512)
        encoder.load_state_dict(strip_module_prefix(net[name]))
        encoder.eval()
        with torch.no_grad():
            halves.append(encoder(mel.unsqueeze(1)))
    return torch.cat(halves, dim=1)


def export(checkpoint: Path, reference: Path, out: Path) -> dict[str, object]:
    import torch

    sys.path.insert(0, str(Path("src").resolve()))
    from agbalu.hub.matoub.configuration_matoub import MatoubConfig
    from agbalu.hub.matoub.modeling_matoub import MatoubForTextToWaveform

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = payload["net"]
    missing = [name for name in (*INFERENCE_MODULES, *STYLE_MODULES) if name not in net]
    if missing:
        message = f"{checkpoint} carries no {missing}; it is not a Matoub Stage 2 checkpoint"
        raise ValueError(message)

    voice = style_vector(net, reference)

    weights: dict[str, Tensor] = {"voice": voice}
    for name in INFERENCE_MODULES:
        for key, value in fuse_weight_norm(strip_module_prefix(net[name])).items():
            weights[f"{name}.{key}"] = value

    config = MatoubConfig()
    model = MatoubForTextToWaveform(config)
    model.load_state_dict(weights, strict=True)
    model.eval()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    parameters = sum(p.numel() for p in model.parameters())
    written = sorted(p.name for p in out.iterdir() if p.is_file())
    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "iterations": int(payload.get("iters", -1)),
        "validation_loss": float(payload["val_loss"]) if "val_loss" in payload else None,
        "reference": str(reference),
        "parameters": parameters,
        "voice_norm": round(float(voice.norm()), 4),
        "voice_acoustic_norm": round(float(voice[:, :128].norm()), 4),
        "voice_prosodic_norm": round(float(voice[:, 128:].norm()), 4),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "weights_bytes": (out / "model.safetensors").stat().st_size,
        "files": written,
    }
    (out / "export.stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="a restored corpus clip from the voice being exported",
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    report = export(arguments.checkpoint, arguments.reference, arguments.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
