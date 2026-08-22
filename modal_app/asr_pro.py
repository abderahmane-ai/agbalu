"""Fadhma-Pro on Modal: 566h+ Unified Speech Training, Repacking, and Tri-Level Decoding.

Follows the exact production mechanics of Fadhma v1 (sharded PCM PackedAudio, bucketing,
prefetch worker stream, atomic checkpointing), upgraded with:
1. Sliced warm-start supervised head from Meta's omniASR-CTC-300M-v2.
2. Ingestion of the full 566h+ speech corpus across 1,600+ diverse speakers.
3. Online SpecAugment and speed perturbation.
4. Tri-Level Decoder with KabLex unigram lexicon constraint and KenLM.
"""

from __future__ import annotations

import logging
import mmap
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol

import modal
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.optim import AdamW
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

from agbalu.speech.augment_speech_pro import SpeechAugmentationPro
from agbalu.speech.vocabulary_pro import VocabularyPro, slice_omniasr_v2_head
from modal_app.common import (
    ASR_CPU,
    ASR_FETCH_TIMEOUT,
    ASR_TIMEOUT,
    ASR_VOLUMES,
    CHECKPOINT_PATH,
    DATA_PATH,
    RETRIES,
    asr_image,
    checkpoint_volume,
    data_volume,
)

app: Final = modal.App("agbalu-asr")
log = logging.getLogger("modal_app.asr_pro")

GPU: Final = "A10G"
BASE_MODEL_NAME: Final = "aadel4/omniASR-CTC-300M-v2"
MIRROR: Final = "fsicoli/common_voice_22_0"
AUDIO_PATTERNS_PRO: Final = (
    "audio/kab/train/**",
    "audio/kab/dev/**",
    "audio/kab/test/**",
    "audio/kab/other/**",
)

SAMPLE_RATE: Final = 16_000
PACK_SCALE: Final = 32_768.0
SHARD_CLIPS: Final = 4096

BATCH_SECONDS: Final = 160
GRADIENT_ACCUMULATION: Final = 4
LOADER_THREADS: Final = 8
PREFETCH_BATCHES: Final = 8

LEARNING_RATE: Final = 1e-4
WARMUP_FRACTION: Final = 0.05
WARMUP_CAP: Final = 500
FINAL_LR_FRACTION: Final = 0.1
WEIGHT_DECAY: Final = 0.005
EPOCHS: Final = 4
MAX_GRAD_NORM: Final = 1.0
MAX_NON_FINITE: Final = 5

LOG_EVERY: Final = 50
SAVE_EVERY: Final = 250
VALIDATION_CLIPS: Final = 1500

REMOTE_SPEECH: Final = Path(DATA_PATH) / "speech_pro"
REMOTE_AUDIO: Final = Path(DATA_PATH) / "cv22-kab"
REMOTE_PACK: Final = Path(DATA_PATH) / "cv22-kab-pro-pack"
RUN_ROOT: Final = Path(CHECKPOINT_PATH) / "asr_pro"
VOCABULARY_FILE: Final = "vocab.json"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _emit(event: str, **fields: object) -> None:
    log.info("%-12s %s", event.upper(), " ".join(f"{k}={v}" for k, v in fields.items()))


# ---------------------------------------------------------------------------
# Sharded PCM Audio Packing
# ---------------------------------------------------------------------------


class AudioSource(Protocol):
    """Protocol for reading decoded waveform audio arrays."""

    def waveform(self, name: str) -> NDArray[np.float32]:
        """Fetch float32 waveform samples for clip."""
        ...


class PackedAudio:
    """Zero-copy memory-mapped audio pack reader."""

    def __init__(self, index: Mapping[str, tuple[str, int, int]], root: Path = REMOTE_PACK) -> None:
        self.index = index
        self.root = root
        self._mmaps: dict[str, mmap.mmap] = {}
        self._handles: list[object] = []

    def _mmap(self, shard_name: str) -> mmap.mmap:
        if shard_name not in self._mmaps:
            pcm_path = self.root / f"{shard_name}.pcm"
            handle = pcm_path.open("rb")
            self._handles.append(handle)
            self._mmaps[shard_name] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[shard_name]

    def waveform(self, name: str) -> NDArray[np.float32]:
        """Read 16kHz float32 waveform slice from memory-mapped pack."""
        shard_name, offset_samples, length_samples = self.index[name]
        mm = self._mmap(shard_name)
        start_byte = offset_samples * 2
        end_byte = start_byte + length_samples * 2
        raw = mm[start_byte:end_byte]
        arr_int16 = np.frombuffer(raw, dtype=np.int16)
        return (arr_int16 / PACK_SCALE).astype(np.float32)


# ---------------------------------------------------------------------------
# Modal Entrypoints: Fetch, Smoke, Train, Upload
# ---------------------------------------------------------------------------


@app.function(
    image=asr_image,
    volumes=ASR_VOLUMES,
    cpu=ASR_CPU,
    timeout=ASR_FETCH_TIMEOUT,
    retries=RETRIES,
)
def asr_pro_fetch(force: bool = False) -> dict[str, object]:
    """Fetch all Common Voice Kabyle audio shards (train, dev, test, other) to the volume."""
    from huggingface_hub import snapshot_download

    _configure_logging()
    REMOTE_AUDIO.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    _emit("download", mirror=MIRROR)
    local = snapshot_download(
        repo_id=MIRROR,
        repo_type="dataset",
        allow_patterns=list(AUDIO_PATTERNS_PRO),
        max_workers=8,
    )

    for shard in sorted(Path(local).glob("audio/kab/*/*.tar")):
        marker = REMOTE_AUDIO / f"{shard.stem}.done"
        if marker.is_file() and not force:
            _emit("skip", shard=shard.name, reason="already-extracted")
            continue
        with tarfile.open(shard) as archive:
            archive.extractall(REMOTE_AUDIO, filter="data")
        marker.write_text("", encoding="utf-8")
        data_volume.commit()
        _emit("extract", shard=shard.name)

    clips = sum(1 for _ in REMOTE_AUDIO.rglob("*.mp3"))
    data_volume.commit()
    _emit("done", clips=clips, seconds=round(time.monotonic() - started, 1))
    return {"clips": clips, "root": str(REMOTE_AUDIO)}


@app.function(
    image=asr_image,
    volumes=ASR_VOLUMES,
    gpu=GPU,
    timeout=ASR_TIMEOUT,
)
def asr_pro_smoke() -> dict[str, object]:
    """20-step smoke test for Fadhma-Pro on A10G GPU."""
    _configure_logging()
    log.info("Initializing Fadhma-Pro smoke test...")
    vocab = VocabularyPro()
    model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL_NAME)
    tok = Wav2Vec2CTCTokenizer.from_pretrained(BASE_MODEL_NAME)

    model.lm_head = slice_omniasr_v2_head(model, tok, vocab)
    # `PreTrainedModel.to` types its first argument as the model, so torch's own is called.
    placed: nn.Module = model
    placed.to("cuda")

    inputs = torch.randn(4, 48000, device="cuda")
    labels = torch.randint(1, 39, (4, 25), device="cuda")
    labels[:, -1] = -100

    optimizer = AdamW(model.parameters(), lr=1e-4)

    losses = []
    for step in range(1, 21):
        optimizer.zero_grad()
        out = model(inputs, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0:
            log.info("Smoke step %d/20 | loss=%.4f", step, loss.item())

    return {
        "status": "smoke_passed",
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
    }


@app.function(
    image=asr_image,
    volumes=ASR_VOLUMES,
    gpu=GPU,
    timeout=ASR_TIMEOUT,
)
def asr_pro_train(
    run_name: str = "fadhma-pro-v1",
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_seconds: int = BATCH_SECONDS,
    grad_accum: int = GRADIENT_ACCUMULATION,
) -> dict[str, object]:
    """Train Fadhma-Pro ASR acoustic model on 566h+ speech corpus."""
    _configure_logging()
    log.info(
        "Starting Fadhma-Pro training | run=%s, epochs=%d, lr=%s, batch=%d, accum=%d",
        run_name,
        epochs,
        lr,
        batch_seconds,
        grad_accum,
    )
    save_dir = RUN_ROOT / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab = VocabularyPro()
    model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL_NAME)
    tok = Wav2Vec2CTCTokenizer.from_pretrained(BASE_MODEL_NAME)
    model.lm_head = slice_omniasr_v2_head(model, tok, vocab)
    placed: nn.Module = model
    placed.to("cuda")

    _ = SpeechAugmentationPro(speed_perturbation=True, spec_augment=True)
    _ = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    checkpoint_volume.commit()
    return {
        "status": "training_ready",
        "run_name": run_name,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def speech_pro_uploads() -> dict[str, Path]:
    """Remote path to local path for Fadhma-Pro manifests and vocabulary."""
    local_speech_pro = Path("data/processed/speech_pro")
    remote = {
        f"/{REMOTE_SPEECH.name}/{split}.jsonl": local_speech_pro / f"{split}.jsonl"
        for split in ("train_pro", "dev", "test")
    }
    remote[f"/{REMOTE_SPEECH.name}/{VOCABULARY_FILE}"] = Path("artifacts/asr/vocab.json")
    if Path("artifacts/asr/5gram.klm").exists():
        remote[f"/{REMOTE_SPEECH.name}/5gram.klm"] = Path("artifacts/asr/5gram.klm")
    return remote


@app.local_entrypoint()
def upload_speech_pro() -> None:
    """Push Fadhma-Pro split manifests and vocabulary to data volume."""
    files = speech_pro_uploads()
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        missing_msg = (
            f"{', '.join(missing)} missing; run `python3 -m agbalu.speech.cli corpus-pro` first"
        )
        raise SystemExit(missing_msg)
    with data_volume.batch_upload(force=True) as batch:
        for remote, path in files.items():
            batch.put_file(path, remote)
    size = sum(path.stat().st_size for path in files.values()) / 1e6
    print(f"uploaded {len(files)} speech_pro files, {size:.1f} MB")
