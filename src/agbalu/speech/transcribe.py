"""Decode audio with a released Fadhma directory, on this machine.

A verification path, not a corpus pass: one clip per forward, no bucketing and no prefetch.
`modal_app.asr` is what scores a split, and it buckets by length because that run is 15,003
clips; pointing this at a corpus would work and would be slow.

The weights come from the published release directory rather than from a training
checkpoint, so what runs here is what a user of `agbalu/Fadhma-300M` gets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from agbalu.speech.vocabulary import decode

if TYPE_CHECKING:
    from numpy.typing import NDArray

SAMPLE_RATE: Final = 16_000
RELEASE_DIR: Final = Path("artifacts/release/Fadhma-300M")
TRAIN_SPLIT: Final = Path("data/processed/speech/train.jsonl")
LM_FILE: Final = "5gram.klm"
VOCAB_FILE: Final = "vocab.json"

AUDIO_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
)

DEFAULT_DEVICE: Final = "cpu"
"""CPU, not `select_device`. This exists to check that a decode is *correct*, and MPS has
already produced a wrong gather in this project without raising; a few clips on CPU cost
seconds. `--device` overrides it."""


class TranscriptionError(Exception):
    """A release directory, an audio file or a decoder that cannot be used."""


@dataclass(frozen=True)
class Transcription:
    path: Path
    text: str
    seconds: float


def audio_files(source: Path) -> list[Path]:
    """Every decodable clip under `source`, sorted. A file resolves to itself."""
    if source.is_file():
        return [source]
    if not source.is_dir():
        msg = f"{source} is neither a file nor a directory"
        raise TranscriptionError(msg)
    found = sorted(path for path in source.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    if not found:
        suffixes = ", ".join(sorted(AUDIO_SUFFIXES))
        msg = f"no audio under {source} — looked for {suffixes}"
        raise TranscriptionError(msg)
    return found


def load_audio(path: Path) -> NDArray[np.float32]:
    """16 kHz mono float32. Resampled only when the file is not already at the rate."""
    import librosa
    import soundfile

    try:
        audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as error:
        msg = f"{path} could not be read as audio: {error}"
        raise TranscriptionError(msg) from error

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def read_vocabulary(release: Path) -> dict[str, int]:
    """The CTC classes the release ships, which decide what any hypothesis can contain."""
    path = release / VOCAB_FILE
    if not path.is_file():
        msg = f"{path} is missing — run `make release REPO=fadhma` first"
        raise TranscriptionError(msg)
    loaded: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _unigrams(train_split: Path) -> list[str] | None:
    """The n-gram's unit inventory, when the training split is on this machine.

    pyctcdecode uses it to warn about vocabulary the language model cannot score. Absent
    locally it is `None`, which changes no hypothesis.
    """
    if not train_split.is_file():
        return None
    from agbalu.speech.lm import extract_unigrams

    targets = (json.loads(line)["text"] for line in train_split.read_text("utf-8").splitlines())
    return extract_unigrams(targets)


def transcribe(
    source: Path,
    release: Path = RELEASE_DIR,
    *,
    device: str = DEFAULT_DEVICE,
    fuse: bool = True,
    limit: int | None = None,
) -> list[Transcription]:
    """Decode every clip under `source`.

    `fuse` shallow-fuses the released 5-gram when it and `pyctcdecode` are both present,
    which is the published CER 8.01 / WER 25.65 path; greedy is 8.53 / 30.12. A missing
    n-gram warns and decodes greedily rather than raising.
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    if not (release / "config.json").is_file():
        msg = f"{release} is not a released model directory — run `make release REPO=fadhma`"
        raise TranscriptionError(msg)

    paths = audio_files(source)[:limit]
    vocabulary = read_vocabulary(release)

    decoder = None
    if fuse:
        from agbalu.speech.lm import build_ctc_decoder

        decoder = build_ctc_decoder(
            vocabulary, unigrams=_unigrams(TRAIN_SPLIT), kenlm_path=release / LM_FILE
        )

    # `PreTrainedModel.to` is decorated with a signature typing its argument as the model
    # rather than the device, so torch's own signature is the one to call through.
    placed: torch.nn.Module = Wav2Vec2ForCTC.from_pretrained(release)
    model = placed.to(torch.device(device))
    model.eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(release)

    results: list[Transcription] = []
    with torch.inference_mode():
        for path in paths:
            audio = load_audio(path)
            batch = extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
            )
            logits = model(
                batch.input_values.to(device), attention_mask=batch.attention_mask.to(device)
            ).logits[0]
            text = (
                str(decoder.decode(logits.cpu().numpy()))
                if decoder is not None
                else decode(logits.argmax(dim=-1).tolist(), vocabulary)
            )
            results.append(Transcription(path, text, len(audio) / SAMPLE_RATE))
    return results
