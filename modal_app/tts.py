"""Task 12.1: what Matoub has to beat, measured before anything is trained.

`facebook/mms-tts-kab` is a Kabyle VITS voice that already exists — `cc-by-nc-4.0`, so
it is a baseline and never a base — and nobody has scored it on Kabyle. It is the exact
analogue of `facebook/omniASR-CTC-300M` on the ASR side: an honest comparison that costs
no training. A baseline measured afterwards is a baseline chosen to be beaten, which is
why this runs before any voice is fine-tuned.

The instrument is Cycle-CER: synthesise held-out text, transcribe it with `Fadhma-300M`,
measure the character error against the text that was synthesised. A bare Cycle-CER is
uninterpretable, so this reports two conditions through one model, one decoder and one
normalisation policy:

1. Fadhma on the prompt set's **real human audio** — the floor for this text.
2. Fadhma on **`mms-tts-kab`'s synthesis** of the same text — the baseline.

The result is the delta. Fadhma's own floor is 8.01% CER over the full test split, so a
Cycle-CER of 9% is not "9% error"; it is a point either side of the measurement's floor.

Two measured properties of the baseline shape this code. Its vocabulary has **no `o`,
`p` or `v`**, and `VitsTokenizer` deletes what it cannot represent without saying so —
`povo` reaches the model as nothing at all — so what was actually voiced is read back out
of the token ids and the scores are reported again over the prompts it voiced in full.
And its duration predictor is **stochastic**: two calls on one sentence returned
waveforms of different lengths, so every synthesis here is seeded from its position.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import batched
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from agbalu.speech.release import BASE_MODEL
from agbalu.tts import cycle
from agbalu.tts.acoustics import summarise
from agbalu.tts.restore import LICENCE as RESTORE_LICENCE
from agbalu.tts.restore import REPO as RESTORE_REPO
from agbalu.tts.voices import TOP
from modal_app.asr import (
    DEFAULT_RUN,
    LOADER_THREADS,
    MIRROR,
    REMOTE_AUDIO,
    REMOTE_LM_FILE,
    REMOTE_SPEECH,
    RUN_ROOT,
    SAMPLE_RATE,
    VOCABULARY_FILE,
    Evaluation,
    Waveforms,
    buckets,
    build_model,
    evaluate,
    index_audio,
    load_pack,
    stream,
)
from modal_app.common import (
    ASR_CPU,
    ASR_REPACK_CPU,
    ASR_VOLUMES,
    DATA_PATH,
    RETRIES,
    TTS_CORPUS_TIMEOUT,
    TTS_DRIVER_CPU,
    TTS_TIMEOUT,
    TTS_VOICE_CPU,
    app,
    asr_image,
    data_volume,
    tts_image,
)

if TYPE_CHECKING:
    from collections.abc import Container, Iterator, Mapping, Sequence
    from concurrent.futures import Executor, Future

    import numpy as np
    import torch
    from numpy.typing import NDArray
    from torch import nn
    from transformers import Wav2Vec2FeatureExtractor

    from agbalu.speech.corpus import Clip
    from agbalu.tts.acoustics import ClipProfile
    from agbalu.tts.corpus import Utterance
    from agbalu.tts.prompts import Prompt
    from agbalu.tts.restore import Restorer

type Measured = tuple[str, ClipProfile]
"""One rendered clip's arm and its profile — what a writer thread hands back."""

GPU: Final = "A10G"

BASELINE: Final = "facebook/mms-tts-kab"
BASELINE_LICENCE: Final = "cc-by-nc-4.0"
"""Read from the Hub API on 2026-08-14. Non-commercial, so it can be measured against
and never fine-tuned from: CLAUDE.md §6.4 puts Apache-2.0 on this project's weights."""

SEED: Final = 0
"""Base seed for the duration predictor, offset by the prompt's position.

`use_stochastic_duration_prediction` is true in this checkpoint's config. Two calls on
one sentence returned waveforms of *different lengths*; seeding before each returned
bit-identical audio. Without this the published number does not reproduce."""

BATCH_SECONDS: Final = 160
"""Audio seconds per scoring batch, the same bucket budget the ASR fine-tune uses, so
the two conditions are batched exactly as the published 8.01% was."""

LOG_EVERY: Final = 200

AUDIO_THREADS: Final = 2 * int(TTS_VOICE_CPU)
"""Threads reading and rendering audio, against `TTS_VOICE_CPU` cores.

Twice the cores, deliberately, because the smoke measured what these threads wait on: a
clip prepared on one thread cost ~1.3 s where its whole CPU cost is ~5 ms, so this pool is
almost entirely blocked on a FUSE round trip and a thread per core would leave the volume
idle between them. `ASR_CPU`'s eight-thread wall does not transfer — it was measured with
the main thread driving an A10 and holding the GIL, where here the main thread issues one
traced forward per `RESTORE_BATCH` clips."""

MIN_DECODE_WORKERS: Final = 2
"""Below this the pool is not built at all: one worker pays for a fork and the pickling of
every logits matrix to buy no concurrency, and pyctcdecode's own sequential path is faster."""

RESTORE_BATCH: Final = 16
"""Clips per restoration pass, as the upstream cleansing script batches them. Every clip
is zero-padded to `restore.CHUNK_SECONDS`, so a batch is a fixed 11 s × 16 kHz × 16."""

REMOTE_TTS: Final = Path(DATA_PATH) / "tts"
PROMPTS_FILE: Final = "prompts.jsonl"
VOICES_FILE: Final = "voices.json"
CORPUS_FILE: Final = "corpus.stats.json"
CORPUS_ROOT: Final = REMOTE_TTS / "voices"
SMOKE_FILE: Final = "corpus.smoke.json"
SMOKE_ROOT: Final = REMOTE_TTS / "voices-smoke"
"""Where a capped build writes. A smoke levels its dozen clips to their own median, so
writing them under the corpus's own paths would leave differently-gained wavs among the
real ones and a 20-clip payload where the result belongs."""
RESULTS: Final = Path(DATA_PATH) / "bench"
RESULT_FILE: Final = "tts-baseline.json"

OOD_FILE: Final = "ood_texts.txt"
"""The adversarial branch's Kabyle text. Uploaded beside the prompt set because 12.6 reads
it from the volume and the recipe's own German file would steer the SLM loss."""

LOCAL_PROMPTS: Final = Path("data/processed/tts") / PROMPTS_FILE
LOCAL_OOD: Final = Path("data/processed/tts") / OOD_FILE
LOCAL_VOICES: Final = Path("data/processed/tts") / VOICES_FILE
LOCAL_CORPUS: Final = Path("data/processed/tts") / CORPUS_FILE
LOCAL_RESULT: Final = Path("data/processed/bench") / RESULT_FILE
"""Paths on the operator's machine, never inside a container. The prompt set is built by
`make tts TASK=prompts` and pushed by `make modal-upload TASK=tts`; the result comes back
through `run_tts`."""

log: Final = logging.getLogger("agbalu.tts")


def _configure_logging() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("modal").setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _emit(event: str, **fields: float | int | str | None) -> None:
    parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    log.info("%-10s %s", event, parts)


@dataclass(frozen=True, slots=True)
class Synthesis:
    """Waveforms held in memory, and what the front end deleted to produce them.

    Satisfies `modal_app.asr.Waveforms`, which is what lets the synthetic condition go
    through the same scoring loop as the real one instead of a second copy of it.
    """

    waves: Mapping[str, NDArray[np.float32]]
    voiced: Mapping[str, str]
    seconds: float
    elapsed: float

    def waveform(self, name: str) -> NDArray[np.float32]:
        return self.waves[name]

    def deleted(self, prompt: Prompt) -> Counter[str]:
        """Speakable characters the front end dropped from this prompt, as a multiset.

        Against the CTC target rather than the raw text: the tokenizer also drops `?`,
        `.` and every other mark outside its 38 symbols, and a punctuation mark is not
        something a voice can fail to say. `ctc_target` is the same reduction both error
        rates are computed under, so this measures what the scores measure.
        """
        return Counter(prompt.target) - Counter(self.voiced[prompt.clip])


def synthesise(prompts: Sequence[Prompt], device: torch.device) -> Synthesis:
    """Voice every prompt with the baseline, one forward pass each, seeded per prompt.

    One prompt per pass rather than a padded batch: the model is 36.3M parameters and
    measured 311 ms per prompt on a laptop CPU, so an A10 spends its time elsewhere, and
    a batched call would have to trim its output by `sequence_lengths` to stay correct.
    """
    import torch
    from transformers import AutoTokenizer, VitsModel

    tokenizer = AutoTokenizer.from_pretrained(BASELINE)
    loaded = VitsModel.from_pretrained(BASELINE)
    # `PreTrainedModel.to` is wrapped in a decorator whose annotation types its first
    # argument as the model rather than the device, so torch's own signature is the one
    # to call through — `build_model` narrows the same way.
    placed: nn.Module = loaded
    model = placed.to(device).eval()
    if loaded.config.sampling_rate != SAMPLE_RATE:
        message = (
            f"{BASELINE} synthesises at {loaded.config.sampling_rate} Hz where the decoder "
            f"takes {SAMPLE_RATE} Hz; the scoring path has no resampler"
        )
        raise RuntimeError(message)

    waves: dict[str, NDArray[np.float32]] = {}
    voiced: dict[str, str] = {}
    samples = 0
    started = time.monotonic()
    for position, prompt in enumerate(prompts):
        inputs = tokenizer(prompt.text, return_tensors="pt")
        # What the model was given, read back out of its own ids rather than predicted:
        # the tokenizer lower-cases and then silently drops every character outside its
        # 38-symbol vocabulary, which is 2.17% of this corpus's transcripts.
        # One sequence decodes to one string; the union is the batch signature.
        decoded = tokenizer.decode(inputs["input_ids"][0])
        spoken = decoded if isinstance(decoded, str) else "".join(decoded)
        if not spoken.strip():
            message = (
                f"{BASELINE} voices nothing of {prompt.clip} ({prompt.text!r}): every "
                f"character is outside its vocabulary, so there is no audio to score"
            )
            raise RuntimeError(message)

        torch.manual_seed(SEED + position)
        with torch.no_grad():
            output = model(**{name: value.to(device) for name, value in inputs.items()})
        wave = output.waveform[0].detach().float().cpu().numpy()

        waves[prompt.clip] = wave
        voiced[prompt.clip] = spoken
        samples += int(wave.size)
        if position == 0 or (position + 1) % LOG_EVERY == 0 or position + 1 == len(prompts):
            elapsed = time.monotonic() - started
            _emit(
                "synth",
                prompt=position + 1,
                of=len(prompts),
                audio_seconds=round(samples / SAMPLE_RATE, 1),
                realtime=round(samples / SAMPLE_RATE / elapsed, 1) if elapsed else None,
            )

    return Synthesis(
        waves=waves,
        voiced=voiced,
        seconds=samples / SAMPLE_RATE,
        elapsed=time.monotonic() - started,
    )


def _condition(name: str, scored: Evaluation, *, audio_seconds: float) -> cycle.Condition:
    """An `Evaluation` as the harness's condition, without rescoring it.

    The rates were computed inside `evaluate` by the same `agbalu.speech.metrics` the
    harness scores through, so these are those numbers rather than a second opinion.
    """
    return cycle.Condition(
        name=name,
        utterances=scored.utterances,
        cer_percent=round(100 * scored.character_error, 4),
        wer_percent=round(100 * scored.word_error, 4),
        loss=round(scored.loss, 4),
        audio_seconds=round(audio_seconds, 1),
        previews=scored.previews,
    )


def deletions(
    prompts: Sequence[Prompt], synthesis: Synthesis
) -> tuple[dict[str, object], set[str]]:
    """What the baseline's front end deleted, and the targets it voiced in full."""
    dropped: Counter[str] = Counter()
    affected = 0
    intact: set[str] = set()
    for prompt in prompts:
        missing = synthesis.deleted(prompt)
        if missing:
            affected += 1
            dropped.update(missing)
        else:
            intact.add(prompt.target)
    summary: dict[str, object] = {
        "prompts_with_deleted_characters": affected,
        "deleted_characters": dict(sorted(dropped.items())),
    }
    return summary, intact


@dataclass(frozen=True, slots=True)
class Scoring:
    """Fadhma as its published 8.01% was measured: one checkpoint, one vocabulary, one
    shallow fusion. Both Phase 12 measurements that read audio go through this, because a
    Cycle-CER against a differently-built decoder is comparing two decoders."""

    model: nn.Module
    vocabulary: Mapping[str, int]
    decoder: object
    extractor: Wav2Vec2FeatureExtractor

    def evaluate(
        self,
        batches: Sequence[Sequence[Clip]],
        audio: Waveforms,
        device: torch.device,
        pool: Executor,
        decode_pool: object | None = None,
    ) -> Evaluation:
        return evaluate(
            self.model,
            batches,
            audio,
            self.vocabulary,
            self.extractor,
            device,
            pool,
            decoder=self.decoder,
            decode_pool=decode_pool,
        )


@contextmanager
def decoding_pool(workers: int) -> Iterator[object | None]:
    """Processes for the fused decode, forked from a parent that has touched no device.

    A beam search is the one stage of the transcript pass that threads cannot spread: it
    is Python around a KenLM query and it holds the GIL throughout, so the device waits
    behind it one clip at a time. pyctcdecode's own answer is a `fork` pool — it keeps the
    language model in a class variable precisely so a forked child inherits it rather than
    unpickling 186 MB per worker — and it falls back to a loop under a spawn context.

    **The caller forks before it builds a model.** A fork inherits neither a CUDA context
    nor the other threads' locks, so a parent that has already done either hands its
    children a broken copy of both.
    """
    import multiprocessing

    if workers < MIN_DECODE_WORKERS:
        yield None
        return
    pool = multiprocessing.get_context("fork").Pool(processes=workers)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()


def vocabulary_table() -> dict[str, int]:
    table: dict[str, int] = json.loads(
        (REMOTE_SPEECH / VOCABULARY_FILE).read_text(encoding="utf-8")
    )
    return table


def ctc_decoder(vocabulary: Mapping[str, int], train_clips: Sequence[Clip]) -> object:
    """The fused decoder alone, built before anything touches the device.

    Separate from `scoring_stack` because of what has to happen between the two: the
    decode is spread across forked processes, and a fork is only safe from a parent that
    has neither initialised CUDA nor started a thread. Building the decoder first is what
    lets the pool inherit the 186 MB n-gram instead of loading it sixteen times.
    """
    from agbalu.speech.lm import build_ctc_decoder, extract_unigrams

    return build_ctc_decoder(
        vocabulary,
        unigrams=extract_unigrams(clip.target for clip in train_clips),
        kenlm_path=REMOTE_SPEECH / REMOTE_LM_FILE,
    )


def scoring_stack(run: str, device: torch.device, decoder: object | None = None) -> Scoring:
    """Load the decoder every condition in this phase is read through."""
    from transformers import Wav2Vec2FeatureExtractor

    from agbalu.model import checkpoint
    from agbalu.speech.corpus import read as read_clips

    vocabulary = vocabulary_table()
    model = build_model(vocabulary, device)
    checkpoint.load(RUN_ROOT / run, model, name=checkpoint.BEST, restore_rng=False)
    if decoder is None:
        decoder = ctc_decoder(vocabulary, read_clips(REMOTE_SPEECH / "train.jsonl"))
    return Scoring(
        model=model,
        vocabulary=vocabulary,
        decoder=decoder,
        extractor=Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL),
    )


@app.function(
    image=asr_image,
    gpu=GPU,
    cpu=ASR_CPU,
    volumes=ASR_VOLUMES,
    timeout=TTS_TIMEOUT,
)
def tts_baseline(
    *, run: str = DEFAULT_RUN, batch_seconds: int = BATCH_SECONDS, limit: int = 0
) -> dict[str, object]:
    """Score `mms-tts-kab` and the prompt set's real audio through the same decoder.

    `limit` truncates the prompt set, and it is this phase's smoke: the real audio comes
    off the same volume whose cold state cost ~$14 and a night in Phase 5, and a short
    call measures that before the full one pays for it.

    No retries. Every failure this can produce — a prompt set that was never uploaded, a
    clip absent from the pack, a checkpoint that is not there — is deterministic, and
    retrying would buy the same diagnosis five times.
    """
    import torch

    from agbalu.tts.prompts import read as read_prompts

    _configure_logging()
    prompts = read_prompts(REMOTE_TTS / PROMPTS_FILE)
    if limit:
        prompts = prompts[:limit]
    real_clips: list[Clip] = [prompt.as_clip() for prompt in prompts]
    real_audio = load_pack("test", real_clips)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring = scoring_stack(run, device)
    _emit(
        "start",
        device=str(device),
        prompts=len(prompts),
        speakers=len({prompt.speaker for prompt in prompts}),
        decoder=run,
        baseline=BASELINE,
    )

    synthesis = synthesise(prompts, device)
    synthetic_clips = [
        prompt.as_clip(duration_ms=int(1000 * synthesis.waveform(prompt.clip).size / SAMPLE_RATE))
        for prompt in prompts
    ]
    front_end, intact = deletions(prompts, synthesis)

    with ThreadPoolExecutor(max_workers=LOADER_THREADS) as pool:
        floor = scoring.evaluate(buckets(real_clips, batch_seconds), real_audio, device, pool)
        baseline = scoring.evaluate(
            buckets(synthetic_clips, batch_seconds), synthesis, device, pool
        )

    report = cycle.Report(
        (
            _condition(
                cycle.FLOOR,
                floor,
                audio_seconds=sum(clip.duration_ms for clip in real_clips) / 1000,
            ),
            _condition(cycle.BASELINE, baseline, audio_seconds=synthesis.seconds),
        )
    )
    delta = report.delta(cycle.BASELINE)
    payload: dict[str, object] = {
        "task": "12.1",
        "decoder": {
            "system": run,
            "base": BASE_MODEL,
            "lm_decoder": "pyctcdecode",
            "kenlm": (REMOTE_SPEECH / REMOTE_LM_FILE).is_file(),
        },
        "prompt_set": {
            "path": str(REMOTE_TTS / PROMPTS_FILE),
            "prompts": len(prompts),
            "speakers": len({prompt.speaker for prompt in prompts}),
            "words": sum(prompt.words for prompt in prompts),
        },
        "front_end": {
            "model": BASELINE,
            "licence": BASELINE_LICENCE,
            "sampling_rate": SAMPLE_RATE,
            "seed": SEED,
            "synthesis_seconds": round(synthesis.elapsed, 1),
            **front_end,
        },
        **report.as_dict(),
        "voiced_in_full": {
            "prompts": len(intact),
            cycle.FLOOR: cycle.restricted(cycle.FLOOR, floor.pairs, intact).as_dict(),
            cycle.BASELINE: cycle.restricted(cycle.BASELINE, baseline.pairs, intact).as_dict(),
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / RESULT_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    _emit("floor", **floor.as_fields())
    _emit("baseline", **baseline.as_fields())
    _emit("done", cycle_cer_delta=delta, deleted=len(prompts) - len(intact))
    return payload


DECODE_LOCK: Final = threading.Lock()
"""libsndfile 1.2.2's MPEG path is not safe to enter from two threads at once, and the clips
are mp3. Measured: three workers decoding the same three files 800 times crash the
interpreter with SIGBUS inside `sf_open` in 6 of 6 runs, and 0 of 8 under this lock. A
warm-up decode on the main thread does not help, so it is the concurrent open rather than a
one-time global init. Only the decode is serialised; the resample, the filterbank, the cut
and the write stay on the pool."""


def _native(path: Path) -> tuple[NDArray[np.float32], int]:
    """One clip at the rate it was stored at.

    `asr._load_audio` resamples to 16 kHz, and the band is what a resample destroys: the
    pack's Nyquist is 8 kHz, so nothing read through it can say whether a recording was
    band-limited before it arrived.
    """
    import numpy as np
    import soundfile

    with DECODE_LOCK:
        audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio), int(rate)


@app.function(
    image=asr_image,
    cpu=ASR_REPACK_CPU,
    volumes=ASR_VOLUMES,
    timeout=TTS_TIMEOUT,
)
def tts_voices(*, top: int = TOP, limit: int = 0) -> dict[str, object]:
    """Task 12.3: identify the candidate voices and profile every clip of each.

    No GPU — mp3 decodes and FFTs. Every clip of a candidate is measured rather than a
    sample of them, because `data/raw` lands in source order and a prefix of a
    source-ordered file is not a sample of it.

    `limit` caps the clips profiled per voice and is the smoke: it proves the audio, the
    transcripts and the corpus are all where this expects them before the full call pays
    for a cold volume. **It takes a prefix, so its numbers are not a sample of the
    voice** — the first 20 clips put voice 1's median peak at 0.7367 where all 19,426 give
    0.2023. `run_voices` writes nothing locally when it is set, and that is why.
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import snapshot_download

    from agbalu.speech.corpus import read as read_clips
    from agbalu.tts.acoustics import profile, summarise
    from agbalu.tts.voices import identify

    _configure_logging()
    started = time.monotonic()

    train = read_clips(REMOTE_SPEECH / "train.jsonl")
    held_out = read_clips(REMOTE_SPEECH / "dev.jsonl") + read_clips(REMOTE_SPEECH / "test.jsonl")
    transcripts = (
        Path(
            snapshot_download(
                repo_id=MIRROR, repo_type="dataset", allow_patterns=["transcript/kab/*.tsv"]
            )
        )
        / "transcript"
    )
    voices = identify(train, held_out, transcripts / "kab", top=top)
    for voice in voices:
        _emit(
            "voice",
            rank=voice.rank,
            hours=round(voice.hours, 2),
            clips=voice.clips,
            gender=voice.demographics.gender or "unlabelled",
        )

    audio = index_audio(REMOTE_AUDIO)
    if not audio:
        message = f"no mp3 clips under {REMOTE_AUDIO}; run `make modal-asr TASK=fetch` first"
        raise RuntimeError(message)

    by_speaker = {
        voice.speaker: [clip for clip in train if clip.speaker == voice.speaker] for voice in voices
    }
    workers = int(ASR_REPACK_CPU * 2)
    measured: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for voice in voices:
            clips = by_speaker[voice.speaker][: limit or None]
            absent = [clip.clip for clip in clips if clip.clip not in audio]
            if absent:
                message = (
                    f"{len(absent)} of {len(clips)} clips of rank {voice.rank} are not on the "
                    f"volume (first: {absent[0]}); run `make modal-asr TASK=fetch`"
                )
                raise RuntimeError(message)
            at = time.monotonic()

            def _profile_clip(clip: Clip) -> ClipProfile:
                return profile(*_native(audio[clip.clip]))

            profiles = list(stream(clips, _profile_clip, pool, workers))
            elapsed = time.monotonic() - at
            _emit("profiled", rank=voice.rank, clips=len(profiles), seconds=round(elapsed, 1))
            measured.append({**voice.as_dict(), "acoustics": summarise(profiles)})

    payload: dict[str, object] = {
        "task": "12.3",
        "corpus": str(REMOTE_SPEECH / "train.jsonl"),
        "clips_per_voice": limit or None,
        "voices": measured,
        "seconds": round(time.monotonic() - started, 1),
    }
    (REMOTE_TTS / VOICES_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    _emit("done", voices=len(measured), seconds=round(time.monotonic() - started, 1))
    return payload


@dataclass(frozen=True, slots=True)
class Candidate:
    """A voice 12.3 measured, as 12.4 needs it: who is speaking and what to call them."""

    speaker: str
    gender: str
    rank: int


def candidates(path: Path) -> tuple[Candidate, ...]:
    """The measured voices, read rather than re-derived.

    Depending on 12.3's artifact is what makes the ordering gate explicit: nothing
    establishes that either voice is usable until it has been profiled, and the gender
    label that names the corpus is resolved there over every table the account appears in.
    """
    if not path.is_file():
        message = f"{path} missing; run `make modal-tts TASK=voices` first"
        raise RuntimeError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    voices = payload.get("voices") if isinstance(payload, dict) else None
    if not isinstance(voices, list) or not voices:
        message = f"{path} carries no voices"
        raise RuntimeError(message)

    found: list[Candidate] = []
    for entry in voices:
        speaker = entry.get("speaker") if isinstance(entry, dict) else None
        gender = entry.get("gender") if isinstance(entry, dict) else None
        rank = entry.get("rank") if isinstance(entry, dict) else None
        if not isinstance(speaker, str) or not isinstance(gender, str) or not isinstance(rank, int):
            message = f"{path} carries a voice without a speaker, gender and rank"
            raise TypeError(message)
        found.append(Candidate(speaker=speaker, gender=gender, rank=rank))
    return tuple(found)


def clip_error_rates(batches: Sequence[Sequence[Clip]], scored: Evaluation) -> dict[str, float]:
    """Per-clip CER, joined to the clips by the order `evaluate` appends them in.

    The length check is the join's proof. `Evaluation` carries no clip ids, so a silent
    misalignment here would attribute one clip's transcript error to another and drop the
    wrong recordings — the defect class a fixed-layout reader already caused once.
    """
    from agbalu.speech.metrics import cer

    ordered = [clip for batch in batches for clip in batch]
    if len(ordered) != len(scored.pairs):
        message = (
            f"{len(ordered)} clips against {len(scored.pairs)} scored pairs; the per-clip "
            f"join would misattribute transcripts"
        )
        raise RuntimeError(message)
    return {
        clip.clip: cer([reference], [hypothesis]).rate
        for clip, (reference, hypothesis) in zip(ordered, scored.pairs, strict=True)
    }


def survey(
    clips: Sequence[Clip],
    rates: Mapping[str, float],
    audio: Mapping[str, Path],
    pool: Executor,
    workers: int,
) -> tuple[list[Utterance], Counter[str]]:
    """Profile every clip and take its verdict. One pass, on the raw audio, for both arms."""
    from agbalu.tts import corpus as voice_corpus
    from agbalu.tts.acoustics import profile

    def _profile_clip(clip: Clip) -> ClipProfile:
        return profile(*_native(audio[clip.clip]))

    kept: list[Utterance] = []
    dropped: Counter[str] = Counter()
    for clip, measured in zip(clips, stream(clips, _profile_clip, pool, workers), strict=True):
        verdict = voice_corpus.consider(clip, measured, rates.get(clip.clip))
        if isinstance(verdict, str):
            dropped[verdict] += 1
        else:
            kept.append(verdict)
    return kept, dropped


COMMIT_EVERY: Final = 20
"""Render batches between volume commits, so a death loses at most that much work.

A full voice is tens of thousands of clips and the function runs for hours; Modal
preempts and timeouts are normal at that length. Committing every `COMMIT_EVERY`
batches (≈320 clips) bounds the lost work to one interval, and a restart picks up
from the last committed batch through clip-level resume below."""

"""**The unit prepared ahead of the device is one clip, not one batch.** Measured, on the
20-clip smoke of 2026-08-15: a batch prepared on a single thread put `host` at 16.5 s
against the device's 2.3, about 1.3 s per clip — which is a volume fetch and not
arithmetic, since the same decode is 5 ms of CPU. The survey pass reads the same clips
`workers`-wide and does twenty in a second, so the fetch parallelises and batching it onto
one thread is what stopped it: lookahead over batches gave four fetches in flight where
the container has room for `workers`."""

WRITE_DEPTH: Final = 2 * RESTORE_BATCH
"""Cut clips allowed to be outstanding with the writers before the loop waits on them.

One batch's worth of both arms, so the writes of the batch just decoded overlap the
forward of the next one and no more: deeper only trades host memory for a longer tail at
the end, since the drain has to finish before the summary can be taken."""

RENDER_LOG_EVERY: Final = 100
"""Batches between render lines. At `RESTORE_BATCH` that is ~1,600 clips, and the first
and last batch are logged whatever this is — a voice is ~2,200 batches, so a cadence that
only divides the full run leaves a smoke silent."""


def _wav(root: Path, utterance: Utterance) -> Path:
    return root / f"{utterance.stem()}.wav"


def _written(root: Path) -> set[str]:
    """The stems an arm already holds, listed once rather than stat'd per clip.

    35,428 clips over two arms is 70,856 `stat` calls against one directory read, and
    these are FUSE paths where a round trip is a network call, not a page.
    """
    return {path.stem for path in root.glob("*.wav")}


def _arms_todo(
    already: Mapping[str, Container[str]], utterance: Utterance, *, resume: bool
) -> tuple[list[str], list[str]]:
    """The arms this clip still owes, and the arms that already hold it.

    Under `resume`, an arm whose wav exists is skipped: the render that wrote it
    committed, so re-running it would spend GPU to reproduce bytes. A change to the
    render path invalidates that assumption and the operator must wipe the partial
    arm directory first.
    """
    if not resume:
        return list(already), []
    stem = utterance.stem()
    todo = [arm for arm, written in already.items() if stem not in written]
    return todo, [arm for arm in already if arm not in todo]


def _write_wav(path: Path, wave: NDArray[np.float32], rate: int) -> None:
    """Write through a staged name, so a killed container leaves no half-clip behind.

    `resume` trusts the presence of a wav as proof it was rendered, which makes a partial
    write indistinguishable from a finished one — and the corpus would carry a truncated
    clip that nothing downstream can tell from a short utterance. `_write_shard` stages
    the audio pack for the same reason.
    """
    import soundfile

    staged = path.with_name(path.name + ".partial")
    # Named, not inferred: `soundfile` reads the container off the extension, and the
    # staged name has none it recognises.
    soundfile.write(staged, wave, rate, subtype="PCM_16", format="WAV")
    staged.replace(path)


def warm_audio_stack() -> None:
    """Import and compile the audio stack once, on one thread, before the pool exists.

    `librosa` and `torchaudio` are imported inside the functions that use them, so without
    this the first render batch pays their import and numba's compilation — and pays it
    with every thread of the pool blocked on one import lock. Measured on the 20-clip
    smoke of 2026-08-15: ~25 s of `host` for twelve clips against ~16 s for seventeen,
    which is a constant wearing a rate's clothes. At 19,426 clips it would vanish into the
    noise; at twelve it *is* the number, and a smoke whose throughput is its own warmup
    cannot say whether the full run is healthy.
    """
    import numpy as np
    import torch

    from agbalu.tts.restore import OUTPUT_RATE, SOURCE_RATE, filterbank, resample

    silence = np.zeros(SOURCE_RATE, dtype=np.float32)
    resample(silence, SOURCE_RATE, OUTPUT_RATE)
    filterbank(torch.zeros(1, SOURCE_RATE, dtype=torch.float32))


def _render_clip(
    arm: str,
    root: Path,
    utterance: Utterance,
    wave: NDArray[np.float32],
    source: int,
    rate: int,
) -> Measured:
    """Cut, level, write and measure one clip of one arm — the whole writer thread."""
    from agbalu.tts import corpus as voice_corpus
    from agbalu.tts.acoustics import profile
    from agbalu.tts.restore import resample

    cut = voice_corpus.apply(resample(wave, source, rate), rate, utterance)
    _write_wav(_wav(root, utterance), cut, rate)
    return arm, profile(cut, rate)


def _profile_written(arm: str, root: Path, utterance: Utterance) -> Measured:
    """Measure a clip a resume found already rendered, so the summary is the whole arm's."""
    from agbalu.tts.acoustics import profile

    return arm, profile(*_native(_wav(root, utterance)))


@dataclass(frozen=True, slots=True)
class Prepared:
    """One clip's host-side work, done off the device: one read, derived for both arms.

    `raw` is already at the output rate and `features` are already filterbank rows, so
    the only thing left for the main thread is the traced forward. `present` names the
    arms a resume found on the volume, which are profiled rather than rendered.
    """

    utterance: Utterance
    raw: NDArray[np.float32] | None
    features: torch.Tensor | None
    samples: int
    present: tuple[str, ...]

    @property
    def rendered(self) -> int:
        return (self.raw is not None) + (self.features is not None)


def render_arms(
    utterances: Sequence[Utterance],
    audio: Mapping[str, Path],
    restorer: Restorer,
    voice_root: Path,
    pool: Executor,
    *,
    workers: int,
    resume: bool = True,
) -> dict[str, object]:
    """Write both arms, with the device the only stage that runs on the main thread.

    Each source clip is read **once** and feeds both the `raw` resample and the restored
    pass. Everything either side of the vocoder — the fetch, the decode, the two
    resamples, the filterbank, the cut, the gain, the write and the profile — is C or
    torch with the GIL released, so all of it runs on `pool` while the device works on
    the batch in front: the loop blocks only where the traced forward is.

    Clips are prepared one at a time and grouped into batches as they arrive, so `workers`
    fetches are in flight rather than one per batch in flight. The measurement that made
    that the shape is under `RESTORE_BATCH`.

    A clip already present in an arm is not re-rendered and the volume is committed every
    `COMMIT_EVERY` batches, so a preempted run resumes from the last durable batch rather
    than from zero. Profiles for skipped clips are read back from their wavs so the
    acoustic summary is the whole arm's across a resume and not this call's share of it.

    The three stage clocks in the result are the instrument for the next decision: they
    say whether the wall is the device, the volume or the writers, which is not
    answerable from a total.
    """
    from agbalu.tts import corpus as voice_corpus
    from agbalu.tts.restore import OUTPUT_RATE as RESTORED_RATE
    from agbalu.tts.restore import resample

    rate = voice_corpus.OUTPUT_RATE
    roots = {arm: voice_root / arm for arm in voice_corpus.ARMS}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    already = {arm: _written(root) if resume else set[str]() for arm, root in roots.items()}

    def _prepare(utterance: Utterance) -> Prepared:
        todo, present = _arms_todo(already, utterance, resume=resume)
        if not todo:
            return Prepared(utterance, None, None, 0, tuple(present))
        # One read per clip, not one per arm: the restored pass and the raw resample are
        # two derivations of the same decode, and reading twice is what the volume charges
        # for twice.
        loaded = _native(audio[utterance.clip])
        rows, samples = restorer.prepare(*loaded) if "restored" in todo else (None, 0)
        return Prepared(
            utterance=utterance,
            raw=resample(*loaded, rate) if "raw" in todo else None,
            features=rows,
            samples=samples,
            present=tuple(present),
        )

    batches = ceil(len(utterances) / RESTORE_BATCH)
    profiles: dict[str, list[ClipProfile]] = {arm: [] for arm in roots}
    pending: deque[Future[Measured]] = deque()
    clock = {"host": 0.0, "device": 0.0, "write": 0.0}
    done = 0
    started = time.monotonic()

    def _drain(limit: int) -> None:
        at = time.monotonic()
        while len(pending) > limit:
            arm, measured = pending.popleft().result()
            profiles[arm].append(measured)
        clock["write"] += time.monotonic() - at

    prepared = batched(stream(utterances, _prepare, pool, workers), RESTORE_BATCH)
    at = time.monotonic()
    for index, batch in enumerate(prepared):
        clock["host"] += time.monotonic() - at
        wanted = [ready for ready in batch if ready.features is not None]
        at = time.monotonic()
        restored = restorer.restore_prepared(
            [rows for ready in wanted if (rows := ready.features) is not None],
            [ready.samples for ready in wanted],
        )
        clock["device"] += time.monotonic() - at

        for ready in batch:
            if ready.raw is not None:
                pending.append(
                    pool.submit(
                        _render_clip, "raw", roots["raw"], ready.utterance, ready.raw, rate, rate
                    )
                )
            for arm in ready.present:
                pending.append(pool.submit(_profile_written, arm, roots[arm], ready.utterance))
            done += ready.rendered
        for ready, wave in zip(wanted, restored, strict=True):
            pending.append(
                pool.submit(
                    _render_clip,
                    "restored",
                    roots["restored"],
                    ready.utterance,
                    wave,
                    RESTORED_RATE,
                    rate,
                )
            )
        _drain(WRITE_DEPTH)

        if index % COMMIT_EVERY == 0:
            data_volume.commit()
        if index == 0 or (index + 1) % RENDER_LOG_EVERY == 0 or index + 1 == batches:
            elapsed = time.monotonic() - started
            _emit(
                "render",
                voice=voice_root.name,
                batch=index + 1,
                of=batches,
                clips=done,
                clips_per_second=round(done / elapsed, 1) if elapsed else None,
                device_share=round(clock["device"] / elapsed, 2) if elapsed else None,
            )
        at = time.monotonic()

    _drain(0)
    data_volume.commit()
    elapsed = time.monotonic() - started
    return {
        "seconds": round(elapsed, 1),
        "clips_rendered": done,
        # Where the wall is, rather than what the run cost. `host` is the loop waiting on a
        # prepared batch — the volume and the mp3 — `device` the traced forward, and `write`
        # the loop waiting on the writers. They sum to the wall; whichever dominates is the
        # only one worth touching next.
        "stage_seconds": {name: round(value, 1) for name, value in clock.items()},
        "arms": {arm: _arm_summary(roots[arm], profiles[arm]) for arm in roots},
    }


def _arm_summary(root: Path, profiles: Sequence[ClipProfile]) -> dict[str, object]:
    return {"root": str(root), "clips": len(profiles), "acoustics": summarise(profiles)}


def report_path(root: Path, name: str) -> Path:
    """A voice's own record, and the marker that says it needs no container.

    Read before the decode and the survey rather than after them: the two passes in front
    of the render cost a GPU hour between them on a full voice, and paying that to hand
    back a file that was already on the volume is what a resume exists not to do.
    """
    return root / name / f"{name}.report.json"


def build_voice(
    candidate: Candidate,
    clips: Sequence[Clip],
    scoring: Scoring,
    restorer: Restorer,
    device: torch.device,
    pool: Executor,
    *,
    audio: Mapping[str, Path],
    batch_seconds: int,
    root: Path,
    dev: int,
    cycle: int,
    workers: int,
    decode_pool: object | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """One voice, both arms, from one set of decisions taken on the raw audio."""
    from agbalu.tts import corpus as voice_corpus

    name = voice_corpus.name_for(candidate.gender)
    absent = [clip.clip for clip in clips if clip.clip not in audio]
    if absent:
        message = (
            f"{len(absent)} of {len(clips)} clips of {name} are not on the volume "
            f"(first: {absent[0]}); run `make modal-asr TASK=fetch`"
        )
        raise RuntimeError(message)

    batches = buckets(clips, batch_seconds)
    scored = scoring.evaluate(
        batches, load_pack("train", list(clips)), device, pool, decode_pool=decode_pool
    )
    rates = clip_error_rates(batches, scored)
    _emit(
        "transcripts",
        voice=name,
        clips=len(rates),
        cer_percent=round(100 * scored.character_error, 4),
    )

    kept, dropped = survey(clips, rates, audio, pool, workers)
    levelled, target = voice_corpus.levelled(kept)
    utterances = voice_corpus.assign_splits(levelled, dev=dev, cycle=cycle)
    _emit("kept", voice=name, clips=len(utterances), dropped=sum(dropped.values()))

    voice_root = root / name
    rendered = render_arms(
        utterances, audio, restorer, voice_root, pool, workers=workers, resume=resume
    )
    for arm in voice_corpus.ARMS:
        arm_root = voice_root / arm
        voice_corpus.write_lists(arm_root, name, utterances, audio_dir=str(arm_root))
    voice_corpus.write_manifest(voice_root, name, utterances)

    payload = {
        **voice_corpus.report(name, candidate.speaker, utterances, dropped, target),
        "rank": candidate.rank,
        "root": str(voice_root),
        "arms": rendered.pop("arms"),
        "render": rendered,
    }
    report_path(root, name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    _emit("voice-done", voice=name, clips=len(utterances), **_stage_fields(rendered))
    return payload


def _stage_fields(rendered: Mapping[str, object]) -> dict[str, float | int | str | None]:
    """The render's stage clocks, flattened onto the closing log line."""
    stages = rendered.get("stage_seconds")
    if not isinstance(stages, dict):
        return {}
    return {f"{name}_s": value for name, value in stages.items() if isinstance(value, float | int)}


@app.function(
    image=tts_image,
    gpu=GPU,
    cpu=TTS_VOICE_CPU,
    volumes=ASR_VOLUMES,
    timeout=TTS_CORPUS_TIMEOUT,
    retries=RETRIES,
)
def tts_voice(args: tuple[str, str, int, str, int, int, bool]) -> dict[str, object]:
    """One candidate voice, both arms, on its own container.

    A voice is the unit of work because nothing crosses between them: the level target,
    the splits and the arms are all per voice, so two containers finish in the time one
    takes and the bill is the same — an A10 for half as long, twice.

    Retried, because at this length a preemption is ordinary and resume is cheap: the
    report is checked before anything is loaded, and inside the render a committed wav is
    never written twice.

    The order of the first three steps is load-bearing and not stylistic. The fused decoder
    is built, then the decode pool is forked, and only then is the device touched — a fork
    from a parent that has initialised CUDA or started a thread is the deadlock this
    codebase has no appetite for, and forking first is also what lets sixteen processes
    share one copy of the 186 MB n-gram.
    """
    speaker, gender, rank, run, batch_seconds, limit, force = args

    import torch

    from agbalu.speech.corpus import read as read_clips
    from agbalu.tts import corpus as voice_corpus
    from agbalu.tts.restore import Restorer, download

    _configure_logging()
    started = time.monotonic()
    data_volume.reload()
    candidate = Candidate(speaker=speaker, gender=gender, rank=rank)
    name = voice_corpus.name_for(gender)
    root = SMOKE_ROOT if limit else CORPUS_ROOT
    dev, cycle = voice_corpus.split_sizes(limit)

    # A capped call is the smoke, and a smoke that resumes proves nothing: it would report
    # the previous call's numbers for a path this one never ran. `force` says the same of a
    # rebuild — a corpus rebuilt because its audio was wrong must not return that audio's
    # report, and every wav is rewritten rather than trusted.
    resume = not (limit or force)
    done = report_path(root, name)
    if resume and done.is_file():
        _emit("resume", voice=name, status="complete")
        return cast("dict[str, object]", json.loads(done.read_text(encoding="utf-8")))

    train = read_clips(REMOTE_SPEECH / "train.jsonl")
    clips = [clip for clip in train if clip.speaker == speaker][: limit or None]
    if not clips:
        message = f"{REMOTE_SPEECH / 'train.jsonl'} holds no clips for speaker {speaker}"
        raise RuntimeError(message)
    decoder = ctc_decoder(vocabulary_table(), train)

    with decoding_pool(int(TTS_VOICE_CPU)) as decode_pool:
        # One intra-op thread per operation: the tensor work worth parallelising is on the
        # device, and the host-side torch here is one filterbank per clip on a thread of
        # its own, where torch's own pool would be `workers` pools fighting for `workers`
        # cores.
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scoring = scoring_stack(run, device, decoder=decoder)
        feature_extractor, vocoder = download(device.type)
        restorer = Restorer(feature_extractor, vocoder, device)
        audio = index_audio(REMOTE_AUDIO)
        warm_audio_stack()
        _emit(
            "start",
            voice=name,
            device=str(device),
            clips=len(clips),
            decoder=run,
            workers=AUDIO_THREADS,
            limit=limit or None,
        )

        with ThreadPoolExecutor(max_workers=AUDIO_THREADS) as pool:
            payload = build_voice(
                candidate,
                clips,
                scoring,
                restorer,
                device,
                pool,
                audio=audio,
                batch_seconds=batch_seconds,
                root=root,
                dev=dev,
                cycle=cycle,
                workers=AUDIO_THREADS,
                decode_pool=decode_pool,
                resume=resume,
            )

    _emit("done", voice=name, seconds=round(time.monotonic() - started, 1))
    return payload


def _corpus_payload(
    run: str, limit: int, built: Sequence[dict[str, object]], started: float
) -> dict[str, object]:
    from agbalu.tts.restore import CHUNK_SECONDS

    return {
        "task": "12.4",
        "corpus": str(REMOTE_SPEECH / "train.jsonl"),
        "clips_per_voice": limit or None,
        "decoder": {"system": run, "base": BASE_MODEL},
        "restoration": {
            "model": RESTORE_REPO,
            "licence": RESTORE_LICENCE,
            "chunk_seconds": CHUNK_SECONDS,
            "batch": RESTORE_BATCH,
        },
        "voices": list(built),
        "seconds": round(time.monotonic() - started, 1),
    }


@app.function(
    image=asr_image,
    cpu=TTS_DRIVER_CPU,
    volumes=ASR_VOLUMES,
    timeout=TTS_CORPUS_TIMEOUT,
)
def tts_corpus(
    *,
    run: str = DEFAULT_RUN,
    batch_seconds: int = BATCH_SECONDS,
    limit: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Task 12.4: cut both candidate voices into a restored and a raw training corpus.

    The voices run as one container each and this one only fans them out and assembles
    their reports, so it asks for no GPU: waiting on `.map` from inside an A10 container
    bills the card for the wait.

    `limit` caps the clips per voice and is this task's smoke. It is the call that answers
    the three questions the restoration arm rests on before the full one pays for them:
    whether the band edge moves above the 7.9 kHz both voices stop at, whether Fadhma's
    error rate survives resynthesis, and whether the traced checkpoints load at all. A
    capped call takes the smoke's held-out sizes and writes under its own root, so it
    neither refuses to split nor overwrites a corpus.

    `force` rebuilds a voice whose report is already on the volume. Without it a voice that
    has been built returns that report, so a call carrying a fix would report the defect it
    was launched to remove.
    """
    from agbalu.tts import corpus as voice_corpus

    _configure_logging()
    started = time.monotonic()
    data_volume.reload()
    voices = candidates(REMOTE_TTS / VOICES_FILE)
    # Resolved here rather than in the worker: an unnameable voice is a deterministic
    # failure, and inside a retried GPU container it is that failure five times over.
    for voice in voices:
        voice_corpus.name_for(voice.gender)
    payload_path = REMOTE_TTS / (SMOKE_FILE if limit else CORPUS_FILE)
    tasks = [
        (voice.speaker, voice.gender, voice.rank, run, batch_seconds, limit, force)
        for voice in voices
    ]
    _emit("start", voices=len(voices), decoder=run, limit=limit or None, force=force or None)

    built: list[dict[str, object]] = []
    payload = _corpus_payload(run, limit, built, started)
    for report in tts_voice.map(tasks):
        built.append(report)
        # Persisted per voice so a driver that dies keeps the voices already built, and the
        # next spawn resumes the rest from their committed wavs.
        payload = _corpus_payload(run, limit, built, started)
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        data_volume.commit()

    _emit("done", voices=len(built), seconds=round(time.monotonic() - started, 1))
    return payload


@app.local_entrypoint()
def upload_tts() -> None:
    """Push the prompt set and candidate voices to the data volume."""
    missing = [path for path in (LOCAL_PROMPTS, LOCAL_OOD) if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        message = f"{names} missing; run `make tts TASK=prompts` and `make tts TASK=ood` first"
        raise SystemExit(message)
    paths_to_upload = [LOCAL_PROMPTS, LOCAL_OOD]
    if LOCAL_VOICES.is_file():
        paths_to_upload.append(LOCAL_VOICES)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(LOCAL_PROMPTS, f"/{REMOTE_TTS.name}/{PROMPTS_FILE}")
        batch.put_file(LOCAL_OOD, f"/{REMOTE_TTS.name}/{OOD_FILE}")
        if LOCAL_VOICES.is_file():
            batch.put_file(LOCAL_VOICES, f"/{REMOTE_TTS.name}/{VOICES_FILE}")
    for path in paths_to_upload:
        print(f"uploaded {path} ({path.stat().st_size / 1e3:.1f} kB)")


@app.local_entrypoint()
def run_tts(run: str = DEFAULT_RUN, limit: int = 0) -> None:
    """Score the baseline, and write the result into the repository.

    `--limit 20` first. It is this phase's smoke and it costs cents: it reads the same
    volume whose cold state cost ~$14 in Phase 5, and it proves the prompt set, the
    checkpoint and the pack are all where this expects them.
    """
    payload = tts_baseline.remote(run=run, limit=limit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if limit:
        print("\nsmoke run — nothing written locally; drop --limit for the result")
        return
    LOCAL_RESULT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_RESULT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n-> {LOCAL_RESULT}")


def band_edge(arm: object) -> float | None:
    """The median band edge an arm's summary reports, or `None` where none was measured.

    It is the number 12.4 exists to move: both voices stop at about 7.9 kHz, so a restored
    arm whose median edge has not risen is a restoration that did nothing.
    """
    acoustics = arm.get("acoustics") if isinstance(arm, dict) else None
    fields = acoustics.get("fields") if isinstance(acoustics, dict) else None
    edge = fields.get("band_edge_hz") if isinstance(fields, dict) else None
    median = edge.get("median") if isinstance(edge, dict) else None
    return float(median) if isinstance(median, float | int) else None


def flat_topped(arm: object) -> int | None:
    """How many of an arm's written clips carry a flat-topped waveform.

    Beside the band edge because the two are read together: restoration that widens the
    band and clips the wave has not produced a corpus, and the arm this is zero on by
    construction is `raw`, whose peak the gain was resolved against.
    """
    acoustics = arm.get("acoustics") if isinstance(arm, dict) else None
    count = acoustics.get("clips_with_clipping") if isinstance(acoustics, dict) else None
    return int(count) if isinstance(count, int) else None


@app.function(image=asr_image, volumes=ASR_VOLUMES, timeout=TTS_TIMEOUT)
def tts_results(smoke: bool = False) -> dict[str, object]:
    """Whatever `tts_corpus` has written, read off the volume."""
    data_volume.reload()
    path = REMOTE_TTS / (SMOKE_FILE if smoke else CORPUS_FILE)
    if not path.is_file():
        return {}
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return payload


@app.local_entrypoint()
def pull_corpus(limit: int = 0) -> None:
    """Copy the built corpus's stats into the repository and print what it holds.

    Separate from building it, because the build is spawned and nothing waits for it: the
    audio stays on the volume, which is where 12.6 reads it, and only the numbers come back.
    `--limit` reads back the smoke `tts_corpus` was given the same flag for, and a smoke's
    dozen clips are printed and never written: they are a sample of a dozen.
    """
    payload = tts_results.remote(smoke=bool(limit))
    if not payload:
        print("nothing on the volume yet — `make modal-status FUNCTION=tts_corpus`")
        return

    if not limit:
        LOCAL_CORPUS.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_CORPUS.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    from agbalu.tts.corpus import ARMS

    voices = payload.get("voices")
    rows = voices if isinstance(voices, list) else []
    print(
        f"{'voice':<12}{'clips':>8}{'hours':>8}{'dropped':>9}{'flat-topped':>14}"
        "  band edge, restored vs raw"
    )
    for row in rows:
        arms = row.get("arms", {})
        edges = (band_edge(arms.get(arm)) for arm in ARMS)
        moved = " vs ".join("—" if edge is None else f"{edge:,.0f} Hz" for edge in edges)
        clipped = " / ".join(
            "—" if count is None else f"{count:,}"
            for count in (flat_topped(arms.get(arm)) for arm in ARMS)
        )
        print(
            f"{row['voice']:<12}{row['clips']:>8,}{row['hours']:>8.2f}"
            f"{sum(row['dropped'].values()):>9,}{clipped:>14}  {moved}"
        )
    if limit:
        print("\nsmoke run — nothing written locally; drop --limit for the result")
        return
    print(f"\n-> {LOCAL_CORPUS}")


@app.local_entrypoint()
def run_voices(top: int = TOP, limit: int = 0) -> None:
    """Profile the candidate voices, and write the result into the repository.

    `--limit 20` first, for the same reason `run_tts` does: it reads the volume whose cold
    state cost ~$14 in Phase 5, and a capped call proves the audio and the transcripts are
    where this expects them before the full one pays for it.
    """
    payload = tts_voices.remote(top=top, limit=limit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if limit:
        print("\nsmoke run — nothing written locally; drop --limit for the result")
        return
    LOCAL_VOICES.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_VOICES.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n-> {LOCAL_VOICES}")
