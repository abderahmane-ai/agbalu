"""Speech restoration, as the step that decides what band Matoub can be trained on.

Both candidate voices stop at about 7.9 kHz (`docs/tts_design.md` §1.3) and the base
synthesises at 24 kHz, so the top octave of the target has no evidence in the source and
a model trained on it learns to predict silence there. Sidon — w2v-BERT 2.0 feature
predictor into a DAC-style vocoder, MIT, `sarulab-speech/sidon-v0.1` — resynthesises
16 kHz input as 48 kHz and removes codec artifacts on the way, which is the defect these
clips actually carry: Common Voice mp3 is where the band went.

**The recovered band is synthesised, not recovered.** The vocoder writes what its
training distribution says a voice does above the source's stop band; no information
comes back from the speaker. It is the LibriTTS-R construction and it is disclosed on
the card rather than assumed away.

Two properties of the upstream cleansing script are deliberately not reproduced. It
zero-pads every clip to 21 seconds whatever its length, which doubles the work for a
corpus whose longest clip is 10.5 s; and it *truncates* anything longer with a warning,
which is a silent deletion of exactly the class this project refuses. `CHUNK_SECONDS` is
sized for this corpus and an over-long clip raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch
    from numpy.typing import NDArray

REPO: Final = "sarulab-speech/sidon-v0.1"
LICENCE: Final = "mit"
"""Read from the repository's own LICENSE and the model card's tag on 2026-08-15. It
contributes no weights to Matoub, so the Apache-2.0 decision on this project's releases
(CLAUDE.md §6.4) is untouched; attribution on the card is the whole obligation."""

CHECKPOINTS: Final[dict[str, tuple[str, str]]] = {
    "cuda": ("feature_extractor_cuda.pt", "decoder_cuda.pt"),
    "cpu": ("feature_extractor_cpu.pt", "decoder_cpu.pt"),
}
"""TorchScript, traced per device. Loading the cuda trace onto CPU fails inside the
deserializer naming neither file."""

SOURCE_RATE: Final = 16_000
OUTPUT_RATE: Final = 48_000

CHUNK_SECONDS: Final = 11.0
"""The longest clip of either candidate voice is 10.548 s. The upstream script's default
is 21.0 and it pads every clip to it, so this is half the feature-extraction work for a
corpus that never reaches the difference."""

MEL_BINS: Final = 80
FRAME_MS: Final = 25.0
SHIFT_MS: Final = 10.0
PREEMPHASIS: Final = 0.97
WINDOW: Final = "povey"
ENERGY_FLOOR: Final = 1.192092955078125e-07
STRIDE: Final = 2
NORM_EPSILON: Final = 1e-5
BATCH_DIMS: Final = 2
"""SeamlessM4T's filterbank front end, as the upstream cleansing code specifies it. These
are not defaults to tune: the feature predictor is a traced graph that was fitted to
exactly this parameterisation, and a mismatch degrades its output rather than raising."""


class RestoreError(Exception):
    """Audio the restorer will not process, or checkpoints it cannot load."""


def download(device: str, *, cache: Path | None = None) -> tuple[Path, Path]:
    """Fetch the traced feature predictor and vocoder for `device`."""
    from huggingface_hub import hf_hub_download

    if device not in CHECKPOINTS:
        message = f"no Sidon trace for device {device!r}; expected one of {sorted(CHECKPOINTS)}"
        raise RestoreError(message)
    where = None if cache is None else str(cache)
    predictor, vocoder = CHECKPOINTS[device]
    return (
        Path(hf_hub_download(REPO, predictor, cache_dir=where)),
        Path(hf_hub_download(REPO, vocoder, cache_dir=where)),
    )


def resample(audio: NDArray[np.float32], source: int, target: int) -> NDArray[np.float32]:
    """One channel from `source` to `target`, through the same soxr path the ASR pack uses."""
    import librosa
    import numpy as np

    if source == target:
        return np.ascontiguousarray(audio, dtype=np.float32)
    converted = librosa.resample(
        np.asarray(audio, dtype=np.float32), orig_sr=source, target_sr=target
    )
    return np.ascontiguousarray(converted, dtype=np.float32)


def filterbank(waveforms: torch.Tensor) -> torch.Tensor:
    """Normalised, stride-folded filterbank features for a batch of equal-length waveforms.

    Per-clip normalisation is over time within each mel bin, which is what the feature
    predictor was traced against; normalising over the batch would make one clip's
    features depend on its neighbours.
    """
    import torch
    from torchaudio.compliance import kaldi

    if waveforms.ndim != BATCH_DIMS:
        message = f"expected a (batch, samples) tensor, got shape {tuple(waveforms.shape)}"
        raise RestoreError(message)

    banks = [
        kaldi.fbank(
            waveform=waveform.unsqueeze(0),
            sample_frequency=SOURCE_RATE,
            num_mel_bins=MEL_BINS,
            frame_length=FRAME_MS,
            frame_shift=SHIFT_MS,
            dither=0.0,
            preemphasis_coefficient=PREEMPHASIS,
            remove_dc_offset=True,
            window_type=WINDOW,
            use_energy=False,
            energy_floor=ENERGY_FLOOR,
        )
        for waveform in waveforms
    ]
    stacked = torch.stack(banks)
    centred = stacked - stacked.mean(1, keepdim=True)
    normalised = centred / torch.sqrt(stacked.var(1, keepdim=True) + NORM_EPSILON)

    kept = (normalised.shape[1] // STRIDE) * STRIDE
    batch = normalised.shape[0]
    return normalised[:, :kept].reshape(batch, kept // STRIDE, MEL_BINS * STRIDE)


def _fit(restored: torch.Tensor, samples: int) -> torch.Tensor:
    """Trim or zero-extend the vocoder's output to the length the input implies."""
    import torch

    current = int(restored.shape[-1])
    if current == samples:
        return restored
    if current > samples:
        return restored[:samples]
    return torch.nn.functional.pad(restored, (0, samples - current))


class Restorer:
    """Sidon inference over batches of clips, at their stored rate in and 48 kHz out.

    Split in two because the two halves belong on different threads. `prepare` is per
    clip and entirely host-side — a resample and a filterbank — so a corpus build runs it
    on its loader pool; `restore_prepared` is the traced forward and is the only part that
    has to wait for the device.
    """

    def __init__(
        self,
        feature_extractor: Path,
        decoder: Path,
        device: torch.device,
        *,
        chunk_seconds: float = CHUNK_SECONDS,
    ) -> None:
        import torch

        if chunk_seconds <= 0:
            message = f"chunk_seconds must be positive, got {chunk_seconds}"
            raise RestoreError(message)
        self._device = device
        self._chunk_samples = round(chunk_seconds * SOURCE_RATE)
        self._features = torch.jit.load(str(feature_extractor), map_location=device).eval()
        self._decoder = torch.jit.load(str(decoder), map_location=device).eval()

    @property
    def chunk_seconds(self) -> float:
        return self._chunk_samples / SOURCE_RATE

    def prepare(self, audio: NDArray[np.float32], rate: int) -> tuple[torch.Tensor, int]:
        """One clip's feature rows, and the output samples its duration implies.

        Split out of `restore` so the resample and the filterbank — the two host-side
        costs of this path — run on a loader thread per clip rather than in front of the
        device. The rows are the ones a batch would produce: `filterbank` centres each
        clip over its own time axis, so a clip padded alone is not a clip measured alone.
        """
        import torch

        if rate < 1:
            message = f"sample rate must be positive, got {rate}"
            raise RestoreError(message)
        narrowed = resample(audio, rate, SOURCE_RATE)
        if narrowed.size > self._chunk_samples:
            message = (
                f"clip is {narrowed.size / SOURCE_RATE:.2f} s against a chunk of "
                f"{self.chunk_seconds:.2f} s; raise chunk_seconds rather than truncating"
            )
            raise RestoreError(message)
        padded = torch.zeros(1, self._chunk_samples, dtype=torch.float32)
        padded[0, : narrowed.size] = torch.from_numpy(narrowed)
        return filterbank(padded)[0], round(audio.size / rate * OUTPUT_RATE)

    def restore_prepared(
        self, features: Sequence[torch.Tensor], lengths: Sequence[int]
    ) -> list[NDArray[np.float32]]:
        """The device half: one forward over rows `prepare` already produced."""
        import numpy as np
        import torch

        if len(features) != len(lengths):
            message = f"{len(features)} feature rows against {len(lengths)} lengths"
            raise RestoreError(message)
        if not features:
            return []

        with torch.inference_mode():
            stacked = torch.stack(list(features)).to(self._device)
            hidden = self._features(stacked)["last_hidden_state"]
            decoded = self._decoder(hidden.transpose(1, 2)).cpu()

        return [
            np.ascontiguousarray(
                _fit(decoded[index].reshape(-1), samples).numpy(), dtype=np.float32
            )
            for index, samples in enumerate(lengths)
        ]
