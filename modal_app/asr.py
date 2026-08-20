"""Fadhma: a CTC fine-tune on Common Voice Kabyle, and the baselines it must beat.

The base is Meta's Omnilingual ASR SSL encoder rather than `w2v-bert-2.0`, for one
reason that is measured rather than argued: `kab_Latn` is one of its 1,668 languages,
so Kabyle is already in the pretraining. Fadhma trains a 300M CTC model on AƔBALU-normalised
transcripts using a 40-class character vocabulary derived from AƔBALU-Text v1.

A CTC head, not a sequence-to-sequence decoder, because Fadhma's output enters the
corpus. Whisper's decoder is an internal language model and hallucinates fluent text
nobody said; worse for Kabyle specifically, six of the ten Kabyle-specific letters have
no token in its vocabulary and decode through UTF-8 byte fallback, so it can emit byte
pairs that are not characters at all.

Four entrypoints, each runnable alone and each idempotent: `fetch` puts the mp3 shards on
the volume, `repack` decodes them once into 16 kHz int16 shards, `finetune` trains, and
`evaluate` scores a checkpoint on a held-out split.

`repack` exists because a step's cost turned out to be a *file count*, not a compute
number. One optimiser step needs ~186 separate clips, and on a volume no worker had read,
every one of those is a network round-trip: the first real run held **51 s/step against a
warm volume's 7.45**, cancelled at step 500 having spent about $14 without reaching its
first validation. Decoding each split once into large sequential shards removes the
per-clip open, the cold fetch and the resample together, and it runs without a GPU.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from agbalu.speech.release import BASE_MODEL, CTC_OVERRIDES
from modal_app.common import (
    ASR_CPU,
    ASR_FETCH_TIMEOUT,
    ASR_REPACK_CPU,
    ASR_REPACK_TIMEOUT,
    ASR_TIMEOUT,
    ASR_VOLUMES,
    CHECKPOINT_PATH,
    DATA_PATH,
    KENLM_CPU,
    KENLM_TIMEOUT,
    RETRIES,
    app,
    asr_image,
    call_owner,
    checkpoint_volume,
    data_volume,
    kenlm_image,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from concurrent.futures import Executor, Future

    import numpy as np
    import torch
    from numpy.typing import NDArray
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from transformers import Wav2Vec2FeatureExtractor

    from agbalu.model.checkpoint import TrainingState
    from agbalu.speech.corpus import Clip

GPU: Final = "A10G"

MIRROR: Final = "fsicoli/common_voice_22_0"
AUDIO_PATTERNS: Final = ("audio/kab/train/**", "audio/kab/dev/**", "audio/kab/test/**")
"""Not `other` or `invalidated`: both are unvalidated audio, and 2.9 GiB of it."""

SAMPLE_RATE: Final = 16_000

PACK_SCALE: Final = 32_768.0
"""Full-scale float to int16. Halving the bytes against float32 is not the reason — the
reason is that `preprocessor_config.json` sets `do_normalize: true`, so the extractor
rescales every clip to zero mean and unit variance and any constant factor here is
divided straight back out. 16-bit quantisation therefore sits below a distinction the
model can make, and well below the noise floor of the mp3 it was decoded from."""

SHARD_CLIPS: Final = 4096
"""Clips per pack shard: ~450 MB at the corpus's 3.41 s mean.

The unit of resumption, and that is the whole reason it exists. The train split is 38
shards; a container that dies at 30 of them re-reads one shard's worth of mp3 rather than
the split. `pack_gaps` derives what is left from the manifest against the shard indices,
so there is no marker file that can disagree with the bytes."""

LOCAL_SPEECH: Final = Path("data/processed/speech")
LOCAL_VOCABULARY: Final = Path("artifacts/asr/vocab.json")
LOCAL_LM_BINARY: Final = Path("artifacts/asr/5gram.klm")
"""Paths written on the operator's machine and never inside a container.

`LOCAL_LM_BINARY` is produced by ``make speech TASK=lm`` (Step 7) and uploaded
to the data volume by ``make modal-upload-speech``.  When absent, `asr_score`
still runs but decodes without KenLM shallow fusion."""

REMOTE_LM_FILE: Final = "5gram.klm"
"""Filename under `REMOTE_SPEECH` on the data volume."""

SPLITS: Final = ("train", "dev", "test")
VOCABULARY_FILE: Final = "vocab.json"
DEFAULT_RUN: Final = "fadhma-v1"
"""`modal_app.launch` spawns against this name, so it is defined once and imported."""

REMOTE_SPEECH: Final = Path(DATA_PATH) / "speech"
REMOTE_AUDIO: Final = Path(DATA_PATH) / "cv22-kab"
REMOTE_PACK: Final = Path(DATA_PATH) / "cv22-kab-pack"
RESULTS: Final = Path(DATA_PATH) / "asr"
RUN_ROOT: Final = Path(CHECKPOINT_PATH) / "asr"
"""`Path`, not the `PurePosixPath` the mount points are declared as: everything here
opens, globs and writes them, and only the container ever evaluates this module."""

BATCH_SECONDS: Final = 160
"""Batches are bucketed by duration rather than fixed in rows: padding a 1.2 s clip to
sit beside a 12 s one wastes the compute the padding occupies, and Common Voice's mean
is 3.46 s against a 20 s ceiling. 160 seconds of audio is ~46 clips at the mean.

**Audio seconds is not the unit the memory is in.** The first convolution is kernel 10,
stride 5, 512 channels, so it expands one 16 kHz channel into 512 at 3.2 kHz — 102 input
samples become 102.4 elements each. `feat_extract_norm` is `layer`, and `layer_norm` sits
on torch's autocast fp32 list, so that output is upcast before the norm: 320 s produced
512 × 1,024,000 × 4 B = **1.95 GiB in one tensor** and OOM'd an A10G at step 2, once
AdamW's two moment buffers existed. Halving this halves the conv transient and the stored
transformer activations together."""

GRADIENT_ACCUMULATION: Final = 4
"""Four micro-batches of `BATCH_SECONDS` is the same 640 audio seconds per optimiser step
the halved batch was taken from, so the schedule and `total` are unchanged by the fix."""

LOADER_THREADS: Final = 8
PREFETCH_BATCHES: Final = 8
"""Batches decoded ahead of the one being trained on, and the pool that decodes them.

Loading was the whole clock: 304 ms per clip in training against 315 ms in validation,
which does no backward pass at all, and 1.08 TFLOP/s from a card whose bf16 peak is 125.
`soundfile` and `soxr` do their work in C with the GIL released, so threads are what
parallelises this; the lookahead is what puts it behind the GPU instead of in front of it.

**Eight workers measured 7.4×** — 55.5 s per step to 7.45 — which also rules out CPU
starvation as the original cause: at 0.125 cores, eight would have bought ~64×.

**Eight is the wall, measured, not assumed.** Twenty-four threads on twelve cores came
back at **8.3 s per step, 11% worse**, so the near-linear scaling to eight does not continue
and the remaining bound is shared — the GIL over librosa's Python layer, or the volume
itself. Raising this is a change that has been tried and cost a run. Depth matches the pool
so no worker idles, at ~240 MB of decoded audio in host memory."""

LEARNING_RATE: Final = 1e-4
WARMUP_FRACTION: Final = 0.05
WARMUP_CAP: Final = 500
"""Warmup is a *share* of the schedule, capped, the way `agbalu.llm.recipe` does it — not a
fixed count. A fixed 500 steps means a 20-step smoke never leaves warmup and trains at 4%
of the peak, so it verifies the plumbing and nothing about whether the model learns."""

FINAL_LR_FRACTION: Final = 0.1
"""Decay to a tenth, the same floor `agbalu.llm.recipe` uses. A CTC head starts random
against a frozen feature encoder; a rate that never comes down leaves the last epochs
bouncing instead of settling."""
WEIGHT_DECAY: Final = 0.005
EPOCHS: Final = 10
MAX_GRAD_NORM: Final = 1.0

MAX_NON_FINITE: Final = 5
LOG_EVERY: Final = 100
SAVE_EVERY: Final = 250
"""~35 minutes of exposure at the measured 8.3 s/step, against a ~30 s write.

1000 was chosen when a step took 55 s, which put 15 hours of work behind one preemption."""

EXPECTED_SECONDS_PER_STEP: Final = 7.45
FLOOR_FACTOR: Final = 3.0
FLOOR_CHECK_AFTER: Final = 20
"""The budget guard. Not a prediction of the rate — a bound on what a run may spend before
a human sees a number.

`EXPECTED_SECONDS_PER_STEP` is the last rate actually measured on a warm volume, so the
packed run should beat it; the check only fires above **22.35 s/step**, which no healthy
configuration of this loop has ever produced. The cold run held 50.7–77.9 and would have
stopped here at step 20 — about 17 minutes and $0.42, against the 9.5 hours and ~$14 it
actually spent before anyone could see a step time.

Twenty steps rather than five because the first step of a run carries cudnn autotune and
the allocator's first pass at every bucket shape, and that step is excluded from the
average outright."""

VALIDATION_CLIPS: Final = 1500
"""Clips of dev scored *during* training, not all 15,002.

The full split is 15.21 hours of audio. Decoding all of it at every validation costs more
than the training it interrupts, for a number whose only job is to rank checkpoints.
The subset is a fixed stride through the duration-sorted split, so it spans the length
range and is identical at every validation — a moving subset would make two checkpoints
incomparable. `asr_score` uses the whole split, because that number is published."""

PREVIEW_CLIPS: Final = 3
"""Every validation prints this many hypothesis/reference pairs. A loss curve does not
show whether the model has learned to emit `ɣ`; the text does."""

log: Final = logging.getLogger("agbalu.asr")


def _configure_logging() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("modal").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _emit(event: str, **fields: float | int | str | None) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    log.info("%-10s %s", event, parts)


class Waveforms(Protocol):
    """A source of 16 kHz mono audio, addressed by clip name.

    All `evaluate` ever asks of the pack, and the whole reason it is stated as a
    protocol: the TTS baseline scores synthesis that exists only in memory, and a
    second copy of the scoring loop is how two conditions stop being comparable.
    """

    def waveform(self, name: str) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One scoring pass, and the two examples that make it readable."""

    loss: float
    word_error: float
    character_error: float
    utterances: int
    previews: tuple[tuple[str, str], ...]
    pairs: tuple[tuple[str, str], ...] = ()
    """Every reference and hypothesis, so a caller can score a subset of the same
    decode rather than paying for a second one. `as_dict` writes only `previews`."""

    def as_dict(self) -> dict[str, object]:
        return {
            "loss": round(self.loss, 4),
            "wer_percent": round(100 * self.word_error, 4),
            "cer_percent": round(100 * self.character_error, 4),
            "utterances": self.utterances,
            "previews": [{"reference": r, "hypothesis": h} for r, h in self.previews],
        }

    def as_fields(self) -> dict[str, float | int | str | None]:
        return {
            "loss": round(self.loss, 4),
            "wer": round(100 * self.word_error, 3),
            "cer": round(100 * self.character_error, 3),
            "utterances": self.utterances,
        }


@app.function(image=asr_image, volumes=ASR_VOLUMES, timeout=ASR_FETCH_TIMEOUT, retries=RETRIES)
def asr_fetch(*, force: bool = False) -> dict[str, object]:
    """Pull the Kabyle audio shards onto the volume and unpack them.

    Extraction is per shard and skipped when its marker exists, so a retried container
    resumes rather than repeating 7.89 GiB of work.
    """
    from huggingface_hub import snapshot_download

    _configure_logging()
    REMOTE_AUDIO.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    local = snapshot_download(
        repo_id=MIRROR,
        repo_type="dataset",
        allow_patterns=[*AUDIO_PATTERNS, "transcript/kab/*.tsv"],
    )
    _emit("download", seconds=round(time.monotonic() - started, 1), path=local)

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


def index_audio(root: Path) -> dict[str, Path]:
    """Clip name to its path on the volume, since the tars nest by split and shard."""
    return {path.name: path for path in root.rglob("*.mp3")}


def shard_paths(split: str, root: Path = REMOTE_PACK) -> list[Path]:
    """Every written shard index for a split, in order. Empty when nothing is packed.

    `root` is a parameter rather than the constant these functions would otherwise close
    over, so the pack can be built and read under a tmp directory in the suite. The default
    is the mount point, so no caller inside the container passes it.
    """
    return sorted((root / split).glob("*.json"))


def _shard_samples(index: Path) -> Path:
    return index.with_suffix(".pcm")


def next_shard(split: str, root: Path = REMOTE_PACK) -> int:
    """One past the highest shard number present, valid or not.

    A count is not a number: with shards 000 and 002 on disk — 001 rejected by `read_shard`
    and its index removed — `len(shard_paths(...))` returns 2 and the next write lands on
    **002**, destroying a good shard. Numbering from the maximum never reuses a number, so a
    rejected shard costs its bytes and nothing else.
    """
    numbers = [int(path.stem) for path in shard_paths(split, root) if path.stem.isdigit()]
    return max(numbers, default=-1) + 1


def read_shard(index: Path) -> list[tuple[str, int, int]]:
    """A shard's `(clip, offset, length)` rows, or `[]` if it is unusable.

    Unusable rather than fatal: a container killed mid-write leaves a shard whose declared
    sample count does not match its file, and the right response is to pack that shard
    again, not to fail the run. The index is written *after* its samples and renamed into
    place, so this can only reject a partial write, never a complete one.
    """
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    samples = _shard_samples(index)
    if not samples.is_file():
        return []
    expected = int(payload.get("samples", -1))
    if samples.stat().st_size != expected * 2 or int(payload.get("sample_rate", 0)) != SAMPLE_RATE:
        return []
    return [(str(name), int(offset), int(length)) for name, offset, length in payload["clips"]]


@dataclass(frozen=True, slots=True)
class PackedAudio:
    """Decoded audio for one split, read by offset out of a handful of open shards.

    A `waveform` call is one `pread` of an exact byte range and a cast: no file open, no
    decode, no resample. Because the pack is written in `pack_order`, a micro-batch's clips
    are adjacent, so those reads walk forward through one shard and the kernel's readahead
    works with the loop instead of against it.

    `os.pread` rather than `mmap`: it takes its offset per call instead of sharing the
    descriptor's position, which is what makes it safe to call from all eight loader threads
    at once, and it asks the filesystem for exactly the bytes wanted. These shards live on a
    FUSE-mounted volume, where a page-fault-driven access pattern is the one thing that
    cannot be checked without paying for a container.

    The descriptors stay open for the run — reopening per clip is precisely the per-clip
    file open this class exists to remove, and a split is at most 38 of them.
    """

    descriptors: tuple[int, ...]
    spans: Mapping[str, tuple[int, int, int]]

    def waveform(self, name: str) -> NDArray[np.float32]:
        import numpy as np

        shard, offset, length = self.spans[name]
        raw = os.pread(self.descriptors[shard], length * 2, offset * 2)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / PACK_SCALE


def load_pack(split: str, clips: Sequence[Clip], root: Path = REMOTE_PACK) -> PackedAudio:
    """Map a split's shards and refuse to proceed if any clip is unpacked.

    Every clip, not most: a missing waveform surfaces otherwise as a `KeyError` from
    inside a loader thread thousands of steps in, on a container that has already been paid
    for. `pack_gaps` is the same predicate read the other way round.
    """
    opened: list[int] = []
    spans: dict[str, tuple[int, int, int]] = {}
    for index in shard_paths(split, root):
        rows = read_shard(index)
        if not rows:
            continue
        opened.append(os.open(_shard_samples(index), os.O_RDONLY))
        for name, offset, length in rows:
            spans[name] = (len(opened) - 1, offset, length)

    absent = [clip.clip for clip in clips if clip.clip not in spans]
    if absent:
        for descriptor in opened:
            os.close(descriptor)
        message = (
            f"{len(absent)} of {len(clips)} {split} clips are not in the pack "
            f"(first: {absent[0]}); run `make modal-asr-repack`"
        )
        raise RuntimeError(message)
    return PackedAudio(descriptors=tuple(opened), spans=spans)


def pack_gaps(split: str, clips: Sequence[Clip], root: Path = REMOTE_PACK) -> list[Clip]:
    """The clips of a split that no valid shard holds yet, in manifest order."""
    packed: set[str] = set()
    for index in shard_paths(split, root):
        packed.update(name for name, _, _ in read_shard(index))
    return [clip for clip in clips if clip.clip not in packed]


def _quantise(wave: NDArray[np.float32]) -> NDArray[np.int16]:
    import numpy as np

    bounded = np.clip(np.rint(wave * PACK_SCALE), -PACK_SCALE, PACK_SCALE - 1)
    quantised: NDArray[np.int16] = bounded.astype(np.int16)
    return quantised


def _load_audio(path: Path) -> NDArray[np.float32]:
    import librosa
    import numpy as np
    import soundfile

    audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def stream[T, R](
    items: Iterable[T], load: Callable[[T], R], pool: Executor, depth: int
) -> Iterator[R]:
    """Apply `load` to each item on `pool`, `depth` in flight, yielding in source order.

    Bounded because the point is to stay a few batches ahead of the GPU, not to decode the
    split into memory. `pool` belongs to the caller: a generator abandoned mid-epoch then
    leaves its futures to finish and be collected, rather than shutting down an executor
    while the loop that owns it is still submitting.
    """
    pending: deque[Future[R]] = deque()
    source = iter(items)
    for item in islice(source, depth):
        pending.append(pool.submit(load, item))
    while pending:
        result = pending.popleft().result()
        for item in islice(source, 1):
            pending.append(pool.submit(load, item))
        yield result


def _waveforms(batch: Sequence[Clip], audio: Waveforms) -> list[NDArray[np.float32]]:
    """Read one batch out of the pack, off the training thread.

    Still threaded and still prefetched: the decode is gone, but the first touch of a page
    is a volume read, and the lookahead is what keeps that behind the GPU rather than in
    front of it.
    """
    return [audio.waveform(clip.clip) for clip in batch]


def pack_order(clips: Sequence[Clip]) -> list[Clip]:
    """Duration then name — the order `buckets` groups in, and the order the pack is written.

    Load-bearing, and the reason it is a named function shared with `buckets` rather than a
    sort in each: a batch is a *contiguous run* of this order, so writing the pack in it
    turns one micro-batch into one sequential span of about 5 MB. Written in manifest order
    instead, the same batch is ~46 reads scattered across 16 GB, which is the access pattern
    the pack exists to remove — it would have cost the decode and kept the seeks.
    """
    return sorted(clips, key=lambda c: (c.duration_ms, c.clip))


def buckets(clips: Sequence[Clip], seconds: int) -> list[list[Clip]]:
    """Group duration-sorted clips into batches of roughly equal total audio.

    Sorting first is what makes the padding cheap. The caller shuffles the batch order,
    so the model does not see every short clip before every long one.

    The clip name breaks ties, so the batches are a function of the *set* of clips and
    not of the order they were read in: a corpus rebuild that reorders rows must not
    silently rebatch, or two runs meant to be compared are not comparable.
    """
    batches: list[list[Clip]] = []
    current: list[Clip] = []
    budget = seconds * 1000
    longest = 0
    for clip in pack_order(clips):
        if current and max(longest, clip.duration_ms) * (len(current) + 1) > budget:
            batches.append(current)
            current, longest = [], 0
        current.append(clip)
        longest = max(longest, clip.duration_ms)
    if current:
        batches.append(current)
    return batches


def _collate(
    batch: Sequence[Clip],
    waves: Sequence[NDArray[np.float32]],
    encode: Mapping[str, int],
    extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad, normalise and encode one already-decoded batch.

    The feature extractor does the padding, not this function. The checkpoint's
    `preprocessor_config.json` sets `do_normalize: true`, so the encoder was pretrained on
    zero-mean unit-variance waveforms; feeding raw floats trains it on a distribution it
    has never seen. The normalisation is also per-sample over the *unpadded* region, which
    is why the attention mask has to be produced by the same call that applies it.
    """
    import numpy as np
    import torch

    features = extractor(
        list(waves),
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    targets = [[encode[ch] for ch in clip.target] for clip in batch]
    longest = max(len(target) for target in targets)
    # -100 is what `Wav2Vec2ForCTC` reads as "not a label", which is how it derives each
    # row's target length. A pad id here would be scored as a blank instead.
    labels = np.full((len(targets), longest), -100, dtype=np.int64)
    for row, target in enumerate(targets):
        labels[row, : len(target)] = target

    return (
        features["input_values"].to(device),
        features["attention_mask"].to(device),
        torch.from_numpy(labels).to(device),
    )


def build_model(vocabulary: Mapping[str, int], device: torch.device) -> nn.Module:
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    from agbalu.speech.vocabulary import PAD_TOKEN

    config = Wav2Vec2Config.from_pretrained(BASE_MODEL)
    config.update(
        {**CTC_OVERRIDES, "vocab_size": len(vocabulary), "pad_token_id": vocabulary[PAD_TOKEN]}
    )
    model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL, config=config, ignore_mismatched_sizes=True)
    # The convolutional front end is trained on 4.3M hours and a CTC fine-tune
    # destabilises it. 4.2M parameters of 315M.
    model.freeze_feature_encoder()
    # Narrowed to `nn.Module` before the move: `PreTrainedModel.to` is wrapped in a
    # decorator whose annotation types its first argument as the model rather than the
    # device, so torch's own signature is the one to call through.
    placed: nn.Module = model
    return placed.to(device)


def _decode_batch(decoder: object, logits: list[NDArray[np.float32]], pool: object) -> list[str]:
    """One batch through the fused decoder, across `pool`'s processes when there is one.

    `decode_batch` applies the same `decode` to the same matrices, so the transcripts are
    the sequential path's; what changes is where they are computed. A beam search is Python
    holding the GIL over a KenLM query, which makes it the one stage in this loop that
    threads cannot spread and processes can — and it is the stage the device waits behind.
    pyctcdecode falls back to a loop itself when the pool was built with a spawn context.
    """
    batched: object = getattr(decoder, "decode_batch", None)
    if pool is not None and callable(batched):
        return [str(text) for text in batched(pool, logits)]
    single: object = getattr(decoder, "decode", None)
    if not callable(single):
        message = f"{type(decoder).__name__} has no `decode`, so it cannot be a CTC decoder"
        raise TypeError(message)
    return [str(single(matrix)) for matrix in logits]


def evaluate(
    model: nn.Module,
    batches: Sequence[Sequence[Clip]],
    audio: Waveforms,
    vocabulary: Mapping[str, int],
    extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    pool: Executor,
    step_seed: int = 0,
    decoder: object | None = None,
    decode_pool: object | None = None,
) -> Evaluation:
    import torch

    from agbalu.speech.metrics import cer, wer
    from agbalu.speech.vocabulary import decode, encoder

    encode = encoder(vocabulary)
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    total = 0.0
    enabled = device.type == "cuda"
    loaded = stream(batches, lambda batch: _waveforms(batch, audio), pool, PREFETCH_BATCHES)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        for batch, waves in zip(batches, loaded, strict=True):
            values, mask, labels = _collate(batch, waves, encode, extractor, device)
            output = model(input_values=values, attention_mask=mask, labels=labels)
            total += float(output.loss)
            if decoder is not None:
                logits_np = output.logits.detach().cpu().to(torch.float32).numpy()
                references.extend(clip.target for clip in batch)
                hypotheses.extend(_decode_batch(decoder, list(logits_np), decode_pool))
            else:
                for clip, ids in zip(batch, output.logits.argmax(dim=-1).tolist(), strict=True):
                    references.append(clip.target)
                    hypotheses.append(decode(ids, vocabulary))
    model.train()

    pairs = list(zip(references, hypotheses, strict=True))
    num_pairs = len(pairs)
    cycle_offset = (step_seed * 7) % max(num_pairs, 1)
    step = max(num_pairs // max(PREVIEW_CLIPS, 1), 1)
    indices = [(cycle_offset + i * step) % num_pairs for i in range(PREVIEW_CLIPS)]
    previews = tuple(pairs[idx] for idx in indices if idx < num_pairs)
    return Evaluation(
        loss=total / max(len(batches), 1),
        word_error=wer(references, hypotheses).rate,
        character_error=cer(references, hypotheses).rate,
        utterances=len(references),
        previews=previews,
        pairs=tuple(pairs),
    )


def _optimizer(model: nn.Module) -> Optimizer:
    import torch

    decayed = [p for name, p in model.named_parameters() if p.requires_grad and "bias" not in name]
    plain = [p for name, p in model.named_parameters() if p.requires_grad and "bias" in name]
    return torch.optim.AdamW(
        [{"params": decayed, "weight_decay": WEIGHT_DECAY}, {"params": plain, "weight_decay": 0.0}],
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


@dataclass(slots=True)
class Session:
    """Everything one training run needs, assembled before the loop starts."""

    device: torch.device
    vocabulary: dict[str, int]
    encode: dict[str, int]
    extractor: Wav2Vec2FeatureExtractor
    model: nn.Module
    optimizer: Optimizer
    train_batches: list[list[Clip]]
    dev_batches: list[list[Clip]]
    audio: PackedAudio
    dev_audio: PackedAudio
    """Separate from `audio` because the pack is per split, where the mp3 index this
    replaced was one glob over every split at once."""
    state: TrainingState
    clips: int
    total_steps: int = 0
    """Set once the schedule is known; `_step_optimizer` reads it for the decay."""


def subsample(clips: Sequence[Clip], cap: int) -> list[Clip]:
    """A fixed stride through a duration-sorted split, so the subset spans its range.

    Deterministic on purpose: two checkpoints validated on different subsets are not
    comparable, and `best.pt` is chosen by exactly that comparison.
    """
    ordered = sorted(clips, key=lambda c: (c.duration_ms, c.clip))
    if cap <= 0 or len(ordered) <= cap:
        return ordered
    return ordered[:: max(len(ordered) // cap, 1)][:cap]


def _prepare(directory: Path, batch_seconds: int) -> Session:
    """Load the corpus, the vocabulary and the model, and resume if there is a checkpoint."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor

    from agbalu.model import checkpoint
    from agbalu.model.checkpoint import TrainingState
    from agbalu.speech.corpus import read
    from agbalu.speech.vocabulary import encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary: dict[str, int] = json.loads(
        (REMOTE_SPEECH / VOCABULARY_FILE).read_text(encoding="utf-8")
    )
    train_clips = read(REMOTE_SPEECH / "train.jsonl")
    dev_clips = read(REMOTE_SPEECH / "dev.jsonl")
    dev_clips = subsample(dev_clips, VALIDATION_CLIPS)
    audio = load_pack("train", train_clips)
    dev_audio = load_pack("dev", dev_clips)

    model = build_model(vocabulary, device)
    optimizer = _optimizer(model)
    state = TrainingState()
    resume = checkpoint.resume_point(directory)
    if resume is not None:
        state = checkpoint.load(directory, model, optimizer, name=resume.name)

    return Session(
        device=device,
        vocabulary=vocabulary,
        encode=encoder(vocabulary),
        extractor=Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL),
        model=model,
        optimizer=optimizer,
        train_batches=buckets(train_clips, batch_seconds),
        dev_batches=buckets(dev_clips, batch_seconds),
        audio=audio,
        dev_audio=dev_audio,
        state=state,
        clips=len(train_clips),
    )


def _epoch_schedule(
    session: Session, batches: Sequence[Sequence[Clip]], pool: Executor
) -> tuple[list[tuple[int, Sequence[Clip]]], Iterator[list[NDArray[np.float32]]]]:
    """The order this epoch trains in, and the loader feeding it.

    Seeded on the epoch, so a resumed run replays the order it was in, and sliced at
    `batch_in_epoch` rather than skipped inside the loop, so a resumed epoch does not
    decode the batches it is about to discard.
    """
    import random

    order = list(range(len(batches)))
    random.Random(session.state.epoch).shuffle(order)
    remaining: list[tuple[int, Sequence[Clip]]] = [
        (position, batches[index])
        for position, index in enumerate(order)
        if position >= session.state.batch_in_epoch
    ]
    loaded = stream(
        (batch for _, batch in remaining),
        lambda batch: _waveforms(batch, session.audio),
        pool,
        PREFETCH_BATCHES,
    )
    return remaining, loaded


def _micro_batch(
    session: Session, batch: Sequence[Clip], waves: Sequence[NDArray[np.float32]]
) -> Tensor:
    """Forward and backward one micro-batch, returning the loss **still on the device**.

    Returning `float(loss)` would synchronise on every micro-batch. The finiteness
    check is deferred to the optimiser step, where the gradient norm has to come back to
    the host anyway, so the whole step costs one sync instead of `GRADIENT_ACCUMULATION`.
    """
    import torch

    values, mask, labels = _collate(batch, waves, session.encode, session.extractor, session.device)
    enabled = session.device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        loss = session.model(input_values=values, attention_mask=mask, labels=labels).loss
    (loss / GRADIENT_ACCUMULATION).backward()
    detached: Tensor = loss.detach()
    return detached


def steps_per_epoch(batches: int) -> int:
    """How often to validate: once an epoch, derived rather than fixed.

    A fixed 2,000 gave **four** points across a 10-epoch run — at epochs 2.4, 4.9, 7.3 and
    9.7 — which cannot show where the curve flattens, and `best.pt` was being chosen from
    four candidates. Validation costs 85 s against ~1.7 h of training per epoch, so ten
    points cost 1.4%. Derived from the schedule because a constant drifts the moment
    `BATCH_SECONDS` or `GRADIENT_ACCUMULATION` moves.
    """
    return max(1, batches // GRADIENT_ACCUMULATION)


def warmup_steps(total: int) -> int:
    """How many steps the rate spends climbing, for a schedule of `total` steps."""
    if total <= 0:
        return WARMUP_CAP
    return max(1, min(WARMUP_CAP, int(total * WARMUP_FRACTION)))


def learning_rate(step: int, total: int) -> float:
    """Linear warmup, then linear decay to `FINAL_LR_FRACTION` of the peak.

    Decay is not decoration on a CTC fine-tune: the head starts random against a frozen
    feature encoder, and a rate that never comes down leaves the last epochs bouncing
    rather than settling — which is what the pilot's own cells look like at their floor.
    """
    warmup = warmup_steps(total)
    if step < warmup:
        return LEARNING_RATE * (step + 1) / warmup
    if total <= warmup:
        return LEARNING_RATE
    through = min(1.0, (step - warmup + 1) / (total - warmup))
    return LEARNING_RATE * (1.0 - through * (1.0 - FINAL_LR_FRACTION))


def _step_optimizer(session: Session, batch: Sequence[Clip]) -> float | None:
    """Clip, then step only if the gradient is finite. `None` means the step was skipped.

    The check has to sit **before** `optimizer.step()`: a non-finite loss produces
    non-finite gradients, and stepping on them writes NaN into every trainable weight,
    from which no later batch recovers. Checking after the step would report the problem
    and have already caused it.

    One synchronisation per optimiser step, not per micro-batch: the clipped norm is
    non-finite exactly when the accumulated gradient is, so it answers both questions.
    """
    import math

    import torch

    state, optimizer = session.state, session.optimizer
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(state.step, session.total_steps)
    norm = float(torch.nn.utils.clip_grad_norm_(session.model.parameters(), MAX_GRAD_NORM))

    if not math.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        state.consecutive_non_finite += 1
        _emit("non-finite", step=state.step, run_length=state.consecutive_non_finite)
        if state.consecutive_non_finite >= MAX_NON_FINITE:
            message = f"{MAX_NON_FINITE} consecutive non-finite gradients; stopping"
            raise RuntimeError(message)
        return None

    state.consecutive_non_finite = 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state.step += 1
    state.tokens_seen += sum(len(clip.target) for clip in batch)
    return norm


def throughput_breach(
    *, seconds: float, steps: int, expected: float, factor: float
) -> float | None:
    """The measured seconds per step when it is far enough off `expected` to stop, else `None`.

    A guard on spend, not on quality, so it is deliberately loose: `factor` of 3 leaves
    every rate this loop has ever produced healthy and still catches the one failure that
    has actually happened, a volume whose pages no worker has read.
    """
    if steps < 1 or expected <= 0 or factor <= 0:
        return None
    measured = seconds / steps
    return measured if measured > expected * factor else None


@dataclass(slots=True)
class ThroughputGuard:
    """Bounds what one call may spend before a human sees a rate.

    `observe` is called once per completed optimiser step and answers one question once:
    the first call only starts the clock — deliberately, so this run's first step, which
    carries cudnn autotune and the allocator's first pass at every bucket shape, is
    excluded rather than averaged in — and the check itself happens on the `after`-th step
    measured. After that it is inert, because a run past its first minutes has already been
    committed to and a mid-run slowdown is a preemption, not a misconfiguration.
    """

    factor: float
    after: int = FLOOR_CHECK_AFTER
    expected: float = EXPECTED_SECONDS_PER_STEP
    started: float | None = None
    steps: int = 0
    breached: float | None = None

    def observe(self) -> float | None:
        if self.started is None:
            self.started = time.monotonic()
            return None
        self.steps += 1
        if self.steps != self.after:
            return None
        self.breached = throughput_breach(
            seconds=time.monotonic() - self.started,
            steps=self.steps,
            expected=self.expected,
            factor=self.factor,
        )
        return self.breached


def _report(
    session: Session, *, loss: float, grad_norm: float, audio_seconds: float, started: float
) -> None:
    """One line per logged step. `realtime` is audio seconds transcribed per wall second,
    which is the rate that says whether 144 hours fits the call — not tokens per second."""
    elapsed = time.monotonic() - started
    _emit(
        "step",
        step=session.state.step,
        loss=round(loss, 4),
        grad_norm=round(grad_norm, 3),
        lr=round(session.optimizer.param_groups[0]["lr"], 7),
        audio_hours=round(audio_seconds / 3600, 3),
        realtime=round(audio_seconds / elapsed, 1) if elapsed else None,
    )


def _summary(
    run: str,
    state: TrainingState,
    *,
    audio_seconds: float,
    trained: int,
    aborted: str | None = None,
    seconds_per_step: float | None = None,
) -> dict[str, object]:
    """The result, with what *this* call did beside what the checkpoint carries.

    `steps`, `epochs` and `best_validation_loss` are cumulative and a resume inherits all
    three, so a call that trained nothing reports the previous one's numbers as its own
    unless `steps_this_run` sits next to them.

    `aborted` is present only when the run stopped itself, so a caller reading the result
    can tell a finished schedule from a guard that fired — the two are otherwise identical
    in shape, and a spawned call's result is all anyone sees.
    """
    summary: dict[str, object] = {
        "run": run,
        "steps": state.step,
        "steps_this_run": trained,
        "epochs": state.epoch,
        "best_validation_loss": state.best_validation_loss,
        "audio_hours": round(audio_seconds / 3600, 3),
    }
    if aborted is not None:
        summary["aborted"] = aborted
    if seconds_per_step is not None:
        summary["seconds_per_step"] = round(seconds_per_step, 2)
    return summary


def _checkpoint_best(directory: Path, session: Session, pool: Executor) -> None:
    """Validate, print sample transcriptions, and keep the checkpoint only if it is a new best."""
    from agbalu.model import checkpoint

    state = session.state
    scored = evaluate(
        session.model,
        session.dev_batches,
        session.dev_audio,
        session.vocabulary,
        session.extractor,
        session.device,
        pool,
        step_seed=state.step,
    )
    _emit(
        "eval",
        step=state.step,
        epoch=state.epoch,
        loss=round(scored.loss, 4),
        wer_pct=round(scored.word_error * 100, 2),
        cer_pct=round(scored.character_error * 100, 2),
        char_acc_pct=round((1.0 - scored.character_error) * 100, 2),
    )
    log.info("--- Validation Sentence Previews (Step %d, Epoch %d) ---", state.step, state.epoch)
    for idx, (reference, hypothesis) in enumerate(scored.previews, start=1):
        log.info("  [%d] REF : %s", idx, reference)
        log.info("      HYP : %s", hypothesis)
    if scored.loss < state.best_validation_loss:
        state.best_validation_loss = scored.loss
        checkpoint.save(directory, session.model, session.optimizer, state, name=checkpoint.BEST)
        checkpoint_volume.commit()
        _emit(
            "best",
            step=state.step,
            epoch=state.epoch,
            loss=round(scored.loss, 4),
            wer_pct=round(scored.word_error * 100, 2),
            cer_pct=round(scored.character_error * 100, 2),
        )


def _write_shard(
    directory: Path,
    number: int,
    clips: Sequence[Clip],
    waves: Sequence[NDArray[np.int16]],
) -> int:
    """Write one shard's samples and then its index, and return the samples written.

    Order matters and is the integrity mechanism: the index names the byte count, and it is
    renamed into place only once the samples are on disk, so `read_shard` can distinguish a
    complete shard from a container killed mid-write without a checksum of its own.
    """
    import numpy as np

    samples = directory / f"{number:03d}.pcm"
    index = directory / f"{number:03d}.json"
    rows: list[tuple[str, int, int]] = []
    offset = 0
    partial = samples.with_suffix(".pcm.partial")
    with partial.open("wb") as handle:
        for clip, wave in zip(clips, waves, strict=True):
            handle.write(np.ascontiguousarray(wave).tobytes())
            rows.append((clip.clip, offset, int(wave.size)))
            offset += int(wave.size)
    partial.replace(samples)

    payload = {"sample_rate": SAMPLE_RATE, "dtype": "int16", "samples": offset, "clips": rows}
    staged = index.with_suffix(".json.partial")
    staged.write_text(json.dumps(payload), encoding="utf-8")
    staged.replace(index)
    return offset


@app.function(
    image=asr_image,
    cpu=ASR_REPACK_CPU,
    volumes=ASR_VOLUMES,
    timeout=ASR_REPACK_TIMEOUT,
    retries=RETRIES,
)
def repack_shard_task(args: tuple[str, int, list[Clip]]) -> dict[str, object]:
    """Decode and write one 4096-clip shard in parallel across Modal containers."""
    split, shard_number, batch = args
    _configure_logging()
    audio = index_audio(REMOTE_AUDIO)
    directory = REMOTE_PACK / split
    directory.mkdir(parents=True, exist_ok=True)

    existing_paths = list(directory.glob(f"shard_{shard_number:04d}_*.pcm"))
    if existing_paths:
        return {
            "split": split,
            "shard": shard_number,
            "clips": len(batch),
            "samples": 0,
            "status": "already_exists",
        }

    at = time.monotonic()
    workers = int(ASR_REPACK_CPU * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        waves = list(
            stream(
                batch,
                lambda clip: _quantise(_load_audio(audio[clip.clip])),
                pool,
                workers,
            )
        )
    samples = _write_shard(directory, shard_number, batch, waves)
    data_volume.commit()
    elapsed = time.monotonic() - at
    _emit(
        "shard",
        split=split,
        shard=shard_number,
        clips=len(batch),
        seconds=round(elapsed, 1),
        clips_per_second=round(len(batch) / elapsed, 1) if elapsed else None,
    )
    return {
        "split": split,
        "shard": shard_number,
        "clips": len(batch),
        "samples": samples,
        "status": "written",
    }


@app.function(
    image=asr_image,
    cpu=ASR_CPU,
    volumes=ASR_VOLUMES,
    timeout=ASR_REPACK_TIMEOUT,
    retries=RETRIES,
)
def asr_repack(*, threads: int = 0, splits: str = "", force: bool = False) -> dict[str, object]:
    """Decode the mp3 shards into 16 kHz int16 in parallel across Modal containers.

    No GPU: this is entirely volume reads and C-library decodes.
    Uses `repack_shard_task.map` to parallelise shard creation across Modal CPU workers,
    reducing total repack time from ~3.5 hours to ~15-20 minutes.
    """
    from agbalu.model import lock
    from agbalu.speech.corpus import read

    _ = threads
    _configure_logging()
    wanted = tuple(name for name in splits.split(",") if name) or SPLITS
    audio = index_audio(REMOTE_AUDIO)
    if not audio:
        message = f"no mp3 clips under {REMOTE_AUDIO}; run `make modal-asr TASK=fetch` first"
        raise RuntimeError(message)

    owner = call_owner()
    lock.acquire(REMOTE_PACK, owner, force=force)
    started = time.monotonic()
    written: dict[str, dict[str, float]] = {}
    try:
        tasks: list[tuple[str, int, list[Clip]]] = []
        for split in wanted:
            directory = REMOTE_PACK / split
            directory.mkdir(parents=True, exist_ok=True)
            clips = read(REMOTE_SPEECH / f"{split}.jsonl")
            gaps = pack_order(pack_gaps(split, clips))
            missing = [clip.clip for clip in gaps if clip.clip not in audio]
            if missing:
                message = (
                    f"{len(missing)} {split} clips have no mp3 on the volume "
                    f"(first: {missing[0]}); re-run `TASK=fetch`"
                )
                raise RuntimeError(message)
            _emit("repack", split=split, clips=len(clips), remaining=len(gaps))

            number = next_shard(split)
            for start in range(0, len(gaps), SHARD_CLIPS):
                batch = gaps[start : start + SHARD_CLIPS]
                tasks.append((split, number, batch))
                number += 1

        if tasks:
            log.info("Spawning %d parallel tasks across Modal containers...", len(tasks))
            results = list(repack_shard_task.map(tasks))
            for result in results:
                split_name = str(result["split"])
                tally = written.setdefault(split_name, {"shards": 0, "samples": 0})
                tally["shards"] += 1
                tally["samples"] += int(result["samples"])

            for tally in written.values():
                tally["hours"] = round(tally["samples"] / SAMPLE_RATE / 3600, 3)
    finally:
        lock.release(REMOTE_PACK, owner)

    _emit("done", seconds=round(time.monotonic() - started, 1))
    return {"root": str(REMOTE_PACK), "splits": written}


def _conclude(
    run: str,
    directory: Path,
    session: Session,
    *,
    audio_seconds: float,
    trained: int,
    breached: float | None,
) -> dict[str, object]:
    """Save, commit and report — the same three steps whether the schedule ran out or the
    throughput guard stopped it, because both leave a checkpoint worth resuming from."""
    from agbalu.model import checkpoint

    state = session.state
    checkpoint.save(directory, session.model, session.optimizer, state)
    checkpoint_volume.commit()
    if breached is None:
        _emit("done", steps=state.step, epochs=state.epoch)
    else:
        log.error(
            "stopping after %d steps: %.1f s/step against an expected %.2f, so this run "
            "would cost %.0f× its projection. The cause that has actually happened is a "
            "volume whose pages no worker has read — check the pack covers every split, "
            "then resume. `--floor 0` overrides this.",
            trained,
            breached,
            EXPECTED_SECONDS_PER_STEP,
            breached / EXPECTED_SECONDS_PER_STEP,
        )
        _emit("aborted", reason="throughput", seconds_per_step=round(breached, 2))
    return _summary(
        run,
        state,
        audio_seconds=audio_seconds,
        trained=trained,
        aborted=None if breached is None else "throughput",
        seconds_per_step=breached,
    )


@app.function(image=asr_image, timeout=ASR_FETCH_TIMEOUT + ASR_TIMEOUT)
def asr_pipeline(
    *,
    run: str = DEFAULT_RUN,
    epochs: int = EPOCHS,
    max_steps: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Fetch the audio, repack it, then train, as one spawned call.

    On a volume that already holds both this costs the seconds the first two need to see
    their own output. Three calls rather than one function because the download and the
    repack want CPU and no GPU, and doing either inside the training container bills an A10
    to watch it.

    No volumes and no retries of its own: it holds a container for the run's whole length,
    and all three parts are separately retried, resumable and idempotent.
    """
    _configure_logging()
    fetched = asr_fetch.remote()
    _emit("fetched", clips=int(str(fetched.get("clips", 0))))
    asr_repack.remote()
    trained: dict[str, object] = asr_finetune.remote(
        run=run, epochs=epochs, max_steps=max_steps, force=force
    )
    return trained


@app.function(
    image=asr_image,
    gpu=GPU,
    cpu=ASR_CPU,
    volumes=ASR_VOLUMES,
    timeout=ASR_TIMEOUT,
    retries=RETRIES,
)
def asr_finetune(
    *,
    run: str = DEFAULT_RUN,
    epochs: int = EPOCHS,
    batch_seconds: int = BATCH_SECONDS,
    max_steps: int = 0,
    force: bool = False,
    floor: float = FLOOR_FACTOR,
) -> dict[str, object]:
    """Train the CTC head and the encoder above the frozen convolutional front end.

    Prefixed `asr_` because every module registers against one `modal.App` and
    `modal_app.mt` already owns `finetune`: a second function of that name does not
    raise, it silently replaces the first on the deployed app.

    Resumable by construction: `agbalu.model.checkpoint` is the same atomic, checksummed
    writer the encoder run uses, and `agbalu.model.lock` is what stops a second container
    spending GPU hours on the same directory.

    `floor` multiplies `EXPECTED_SECONDS_PER_STEP` to bound what the call may spend before
    a human sees a rate; `0` disables it. A breach **returns** rather than raising, and that
    is the point: `retries=RETRIES` is five, it cannot tell a bug from a preemption, and a
    slow volume is deterministic — raising here would buy the same diagnosis five times.
    """
    from agbalu.model import checkpoint, lock

    _configure_logging()
    directory = RUN_ROOT / run
    directory.mkdir(parents=True, exist_ok=True)

    owner = call_owner()
    lock.acquire(directory, owner, force=force)
    try:
        session = _prepare(directory, batch_seconds)
        state, train_batches = session.state, session.train_batches
        spe = steps_per_epoch(len(train_batches))
        validate_every = 250
        total = max_steps or epochs * spe
        session.total_steps = total
        _emit(
            "start",
            run=run,
            device=str(session.device),
            classes=len(session.vocabulary),
            clips=session.clips,
            batches=len(train_batches),
            steps=total,
            validate_every=validate_every,
            resumed_at=state.step or None,
        )
        if state.step >= total:
            log.warning(
                "nothing to train: resumed at step %d of %d. Raise --max-steps, or use a "
                "fresh --run — every rate this run could report would describe an empty "
                "loop, and the losses below are the checkpoint's, not this run's",
                state.step,
                total,
            )
            return _summary(run, state, audio_seconds=0.0, trained=0)

        entered = state.step
        started = time.monotonic()
        audio_seconds = 0.0
        guard = ThroughputGuard(factor=floor)
        with ThreadPoolExecutor(max_workers=LOADER_THREADS) as pool:
            while state.step < total:
                remaining, loaded = _epoch_schedule(session, train_batches, pool)
                for (position, batch), waves in zip(remaining, loaded, strict=True):
                    state.batch_in_epoch = position + 1
                    loss = _micro_batch(session, batch, waves)
                    audio_seconds += sum(clip.duration_ms for clip in batch) / 1000
                    if (position + 1) % GRADIENT_ACCUMULATION:
                        continue

                    grad_norm = _step_optimizer(session, batch)
                    if grad_norm is None:
                        continue
                    # `entered + 1`, not 1: a resumed run's first step is not step 1, and
                    # keying on the absolute number leaves it silent until the next
                    # multiple of `LOG_EVERY` — ten minutes of nothing on a resume.
                    if state.step in (entered + 1, total) or state.step % LOG_EVERY == 0:
                        _report(
                            session,
                            loss=float(loss),
                            grad_norm=grad_norm,
                            audio_seconds=audio_seconds,
                            started=started,
                        )
                    if guard.observe() is not None:
                        break
                    if state.step % SAVE_EVERY == 0:
                        checkpoint.save(directory, session.model, session.optimizer, state)
                        checkpoint_volume.commit()
                        lock.refresh(directory, owner)
                        _emit("checkpoint", step=state.step, epoch=state.epoch, run=run)
                    if state.step % validate_every == 0 or state.step == total:
                        _checkpoint_best(directory, session, pool)
                    if state.step >= total:
                        break
                if guard.breached is not None:
                    break
                state.epoch += 1
                state.batch_in_epoch = 0

        return _conclude(
            run,
            directory,
            session,
            audio_seconds=audio_seconds,
            trained=state.step - entered,
            breached=guard.breached,
        )
    finally:
        lock.release(directory, owner)


@app.function(
    image=asr_image,
    gpu=GPU,
    cpu=ASR_CPU,
    volumes=ASR_VOLUMES,
    timeout=ASR_TIMEOUT,
    retries=RETRIES,
)
def asr_score(
    *,
    run: str = DEFAULT_RUN,
    split: str = "test",
    batch_seconds: int = BATCH_SECONDS,
    limit: int = 0,
) -> dict[str, object]:
    """Score the fine-tune on a held-out split, under `agbalu.speech.metrics`' policy."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor

    from agbalu.model import checkpoint
    from agbalu.speech.corpus import read

    _configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary: dict[str, int] = json.loads(
        (REMOTE_SPEECH / VOCABULARY_FILE).read_text(encoding="utf-8")
    )
    clips = read(REMOTE_SPEECH / f"{split}.jsonl")
    if limit:
        clips = clips[:limit]

    from agbalu.speech.lm import build_ctc_decoder, extract_unigrams

    train_clips = read(REMOTE_SPEECH / "train.jsonl")
    unigrams = extract_unigrams(c.target for c in train_clips)
    kenlm_path = REMOTE_SPEECH / REMOTE_LM_FILE
    model = build_model(vocabulary, device)
    checkpoint.load(RUN_ROOT / run, model, name=checkpoint.BEST, restore_rng=False)
    decoder = build_ctc_decoder(vocabulary, unigrams=unigrams, kenlm_path=kenlm_path)
    with ThreadPoolExecutor(max_workers=LOADER_THREADS) as pool:
        scored = evaluate(
            model,
            buckets(clips, batch_seconds),
            load_pack(split, clips),
            vocabulary,
            Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL),
            device,
            pool,
            decoder=decoder,
        )

    payload: dict[str, object] = {
        "system": run,
        "base": BASE_MODEL,
        "split": split,
        "clips": len(clips),
        "lm_decoder": "pyctcdecode",
        **scored.as_dict(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{run}-{split}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    _emit("scored", system=run, split=split, **scored.as_fields())
    return payload


def speech_uploads() -> dict[str, Path]:
    """Remote path to local path for everything the run reads off the volume but audio.

    Separate from the entrypoint so the set can be asserted against what the container
    opens without a test having to talk to Modal.
    """
    remote = {
        f"/{REMOTE_SPEECH.name}/{split}.jsonl": LOCAL_SPEECH / f"{split}.jsonl" for split in SPLITS
    }
    remote[f"/{REMOTE_SPEECH.name}/{VOCABULARY_FILE}"] = LOCAL_VOCABULARY
    remote[f"/{REMOTE_SPEECH.name}/{REMOTE_LM_FILE}"] = LOCAL_LM_BINARY
    return remote


@app.local_entrypoint()
def upload_speech() -> None:
    """Push the split manifests and the CTC vocabulary to the data volume.

    `fetch` brings the audio and nothing else, so without this the fine-tune resolves
    `vocab.json` on the volume, does not find it, and fails minutes into a GPU container.
    The four files are ~60 MB against the audio's 20-odd GB.
    """
    files = speech_uploads()
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        message = (
            f"{', '.join(missing)} missing; run `make speech TASK=corpus` "
            f"and `make speech TASK=vocabulary` first"
        )
        raise SystemExit(message)

    with data_volume.batch_upload(force=True) as batch:
        for remote, path in files.items():
            batch.put_file(path, remote)
    size = sum(path.stat().st_size for path in files.values()) / 1e6
    print(f"uploaded {len(files)} speech files, {size:.1f} MB")


@app.function(
    image=kenlm_image,
    cpu=KENLM_CPU,
    volumes={str(DATA_PATH): data_volume},
    timeout=KENLM_TIMEOUT,
)
def asr_lm() -> dict[str, object]:
    """Build the shallow-fusion n-gram from AƔBALU-Text v1, on the volume.

    Here rather than on the operator's machine because `lmplz` and `build_binary` are C++
    programs that need cmake and Boost, and the published 186 MB binary was first built in
    a container assembled by hand — so the command that produced it existed nowhere in the
    repository. `agbalu.speech.lm` owns the arguments; this owns the machine.
    """
    from agbalu.speech.lm import LM_ORDER, LM_PRUNE, build_arpa, compile_binary

    _configure_logging()
    corpus = Path(DATA_PATH) / "text" / "agbalu-text-v1.jsonl"
    if not corpus.is_file():
        message = f"no corpus at {corpus}; run `make modal-upload TASK=corpus` first"
        raise FileNotFoundError(message)

    plain = Path(DATA_PATH) / "speech" / "lm_sentences.txt"
    plain.parent.mkdir(parents=True, exist_ok=True)
    sentences = 0
    with plain.open("w", encoding="utf-8") as sink, corpus.open(encoding="utf-8") as source:
        for line in source:
            text = json.loads(line).get("text", "").strip()
            if text:
                sink.write(text + "\n")
                sentences += 1
    _emit("corpus", sentences=sentences, path=str(plain))

    arpa = plain.with_name("5gram.arpa")
    binary = REMOTE_SPEECH / REMOTE_LM_FILE
    build_arpa(plain, arpa, order=LM_ORDER)
    compile_binary(arpa, binary)
    arpa.unlink()
    plain.unlink()
    data_volume.commit()
    return {
        "sentences": sentences,
        "order": LM_ORDER,
        "prune": LM_PRUNE,
        "path": str(binary),
        "megabytes": round(binary.stat().st_size / 1e6, 1),
    }


@app.local_entrypoint()
def run_asr(
    task: str = "finetune",
    run: str = DEFAULT_RUN,
    split: str = "test",
    epochs: int = EPOCHS,
    max_steps: int = 0,
    limit: int = 0,
    force: bool = False,
) -> None:
    """`fetch` once, then `repack` once, then `finetune`, then `evaluate`."""
    if task == "fetch":
        result = asr_fetch.remote(force=force)
    elif task == "repack":
        result = asr_repack.remote()
    elif task == "finetune":
        result = asr_finetune.remote(run=run, epochs=epochs, max_steps=max_steps, force=force)
    elif task == "evaluate":
        result = asr_score.remote(run=run, split=split, limit=limit)
    elif task == "lm":
        result = asr_lm.remote()
    else:
        message = f"unknown task {task!r}; expected fetch, repack, lm, finetune or evaluate"
        raise SystemExit(message)
    print(json.dumps(result, ensure_ascii=False, indent=2))
