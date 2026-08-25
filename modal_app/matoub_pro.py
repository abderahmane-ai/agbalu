"""Matoub-Pro: the corpus and the fine-tune, on one app of their own.

`docs/matoub_pro_design.md` is the evidence. What this changes against `modal_app.matoub`,
and why each change is here rather than argued for downstream:

1. **The phoneme target keeps its punctuation.** `agbalu.tts.g2p_pro` reads `Clip.text`,
   already normalised, instead of the ASR target. Kokoro's table carries `, . ; : ! ? … “ ” —`
   as trained rows and v1 fed it none of them, so it was fitted to sequences in which no
   utterance ends and it produces no pauses at all.
2. **Clips are joined into sessions.** Consecutive Common Voice ids are one sitting, the seam
   between two of them is kept exactly as recorded, and a run is packed to ten seconds. That
   supervises the pause after a full stop and lifts the corpus past the five-second floor
   `SLMAdversarialLoss` gates on.
3. **Stage 1 is the corpus, not two speakers.** Every train-split speaker that survives an
   acoustic filter goes in, so the style encoder and the diffusion sampler have a style space
   to model rather than two points.
4. **The schedule is upstream's ratios**, diffusion included: `lambda_diff` and `lambda_sty`
   at 1.0, `TMA_epoch` a quarter in, `joint_epoch` half way. v1 ran with the sampler off and
   reached joint training in two of four epochs.
5. **The SLM has heard Kabyle.** `ylacombe/omniASR_W2V_300M_SSL` is Apache-2.0 and is Fadhma's
   own base; `microsoft/wavlm-base-plus` is English. It is truncated to twelve layers, so the
   discriminator keeps the shape WavLM gave it and the run keeps its card.

**It runs on an A10G.** `loss_dur` and `loss_ce` are computed over the whole utterance's text,
so `max_len` never touches the pause supervision and the change worth most costs no memory.
Stage 1 keeps the (200, 4) v1 measured to fit; Stage 2 takes (200, 2), the same
`max_len x batch` product as the point that ran, with `resources/styletts2_pro.patch` removing
the three memory terms the longer utterances would otherwise add.

Every deterministic failure is forced onto a CPU container by `matoub_pro_prepare` and is
*returned* rather than raised, because `retries` is five and cannot tell a bug from a
preemption.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import modal

from agbalu.tts import corpus_pro
from agbalu.tts.acoustics import summarise
from agbalu.tts.corpus import CER_LIMIT, OUTPUT_RATE, PEAK_CEILING_DBFS
from agbalu.tts.restore import OUTPUT_RATE as RESTORED_RATE
from agbalu.tts.restore import Restorer, download, resample
from modal_app.asr import REMOTE_AUDIO, REMOTE_SPEECH, index_audio
from modal_app.common import (
    DATA_PATH,
    RESOURCES,
    RETRIES,
    STYLETTS2_PATH,
    checkpoint_volume,
    data_volume,
    matoub_image,
    tts_image,
)
from modal_app.matoub import (
    BASE_CONFIG,
    BASE_REPO,
    BASE_REVISION,
    BASE_WEIGHTS,
    STAGE1,
    STAGE2,
    CapacityError,
    DeterministicError,
    PreconditionError,
    assert_base_matches_vendored,
    assert_recipe_imports,
    convert_base,
    latest_checkpoint,
    run_stage,
    write_symbol_table,
)
from modal_app.tts import native, write_wav

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from agbalu.speech.corpus import Clip
    from agbalu.tts.acoustics import ClipProfile
    from agbalu.tts.corpus_pro import Joined, Source
    from agbalu.tts.vocabulary import Vocabulary

APP_NAME: Final = "agbalu-matoub"
"""Not the shared app. A subset deploy replaces an app's function set, so a module spawned
against while something else is training has to own the app it is addressed through."""

app: Final = modal.App(APP_NAME)
log: Final = logging.getLogger("agbalu.matoub_pro")

VOICE: Final = "kab_male"
"""The one voice Pro trains. `kab_female` stays Phase 12's and is untouched."""

POOL: Final = "stage1"

CORPUS_ROOT: Final = Path(DATA_PATH) / "tts_pro"
WORK_ROOT: Final = CORPUS_ROOT / "matoub"
V1_TTS: Final = Path(DATA_PATH) / "tts"
V1_REPORT: Final = V1_TTS / "corpus.stats.json"
V1_MANIFEST: Final = V1_TTS / "voices" / VOICE / f"{VOICE}.manifest.jsonl"
OOD_SOURCE: Final = V1_TTS / "ood_texts.txt"

PRO_PATCH: Final = RESOURCES / "styletts2_pro.patch"
STAGE2_PATCH: Final = RESOURCES / "styletts2_stage2_resume.patch"
PATCHED_FILES: Final = ("train_first.py", "train_second.py", "meldataset.py", "losses.py")
"""Every file the two patches touch, checked out before either is applied: `git apply`
refuses a diff that is already applied, and a preempted container is retried."""

GPU: Final = "A10G"
CORPUS_CPU: Final = 16.0
TRAIN_CPU: Final = 8.0
"""Eight cores for the trainer, measured on the ASR path: Modal's default request is an
eighth of a core and it shows up as a slow step rather than as a CPU error."""

TIMEOUT: Final = 24 * 60 * 60
CORPUS_TIMEOUT: Final = 12 * 60 * 60

MAX_LEN: Final[Mapping[str, int]] = {STAGE1: 200, STAGE2: 200}
BATCH: Final[Mapping[str, int]] = {STAGE1: 4, STAGE2: 2}
"""The A10G operating point. v1 ran (200, 4) through six Stage 1 epochs and died at (200, 4)
in Stage 2's first joint step, asking 21.41 GiB of 22.06. The joint step is dominated by
waveform length times batch, so Stage 2 takes the same 400 units at twice the window."""

BATCH_PERCENTAGE: Final[Mapping[str, float]] = {STAGE1: 0.5, STAGE2: 1.0}
"""`slmadv` breaks out at `len(wav) >= batch_percentage * len(waves)` and returns `None` if
it collected one sample or fewer. At batch 2 anything under 1.0 silently removes the
adversary from every joint step."""

SLM_MODEL: Final = "ylacombe/omniASR_W2V_300M_SSL"
SLM_LICENCE: Final = "apache-2.0"
SLM_LAYERS: Final = 12
SLM_HIDDEN: Final = 1024
SLM_MIN_SHARE: Final = 0.5
"""How much of the encoder is kept, what that leaves the discriminator, and the fraction of
joint steps the adversarial branch must fire on. `nlayers` is `SLM_LAYERS + 1` because the
embedding output is a hidden state too, which is why WavLM's twelve layers configure as 13."""

MIN_OOD_PHONEMES: Final = 120
MIN_OOD_LINES: Final = 500
"""`slmadv` needs the OOD text to predict past `SLMADV_MIN_LEN` or every sample is skipped.
Measured on `/data/tts/ood_texts.txt`: 1,775 of its 5,000 lines reach this floor."""

SLMADV_MIN_LEN: Final = 300
SLMADV_MAX_LEN: Final = 500
"""In mel frames. The gate is `mel_input_length > min_len`, so 300 asks for utterances past
3.75 s — most of the joined corpus, and none of v1's."""

SEED: Final = 0

POOL_HOURS: Final = 80.0
"""How much of the train split Stage 1 takes, best speakers first.

A budget, not a threshold: 144.31 hours are available and write 24.9 GB of 24 kHz PCM_16 to
the volume where 80 write 13.8. The acoustic filter decides *which* speakers; this decides
how many of them one run pays for."""

MIN_SPEAKER_HOURS: Final = 0.25
MIN_SPEECH_TO_FLOOR_DB: Final = 25.0
MIN_SPEECH_FRAME_SHARE: Final = 0.40
MAX_ZEROED_SHARE: Final = 0.25
"""The acoustic filter, keyed on what 12.3 measured rather than on taste. Rank 2 — the voice
this project shipped — reads 36.05 dB speech-to-floor, a 0.687 speech share and 1.8% of its
clips gated to digital zero; rank 1, which 12.3 recommended against, reads 59.7% zeroed.
Every threshold admits the first and is what separates it from the second. `band_edge_hz` is
deliberately absent: every Common Voice clip stops near 7.9 kHz, so it separates nothing."""

RENDER_LOG_EVERY: Final = 500
RESTORE_BATCH: Final = 16
COMMIT_EVERY: Final = 2_000

DIFF_RATIO: Final = 5
TMA_RATIO: Final = 4
JOINT_RATIO: Final = 2
"""Upstream's schedule as fractions of the run: `diff_epoch` 20 of 200, `TMA_epoch` 50 of
200, `joint_epoch` 50 of 100 in Stage 2. v1 wrote 999, 0, and a third."""


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _emit(event: str, **fields: object) -> None:
    log.info("%s %s", event, json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str))


def _refuse(reason: str, **fields: object) -> dict[str, object]:
    """Report a precondition failure as this call's result, so Modal does not retry it."""
    payload: dict[str, object] = {"trained_this_run": False, "refused": reason, **fields}
    _emit("refused", **payload)
    return payload


def corpus_root(*, limit: int) -> Path:
    """Where one build lives, keyed on its cap so a smoke never shares a real run's audio."""
    return CORPUS_ROOT / (f"corpus-smoke-{limit}" if limit else "corpus")


def work_root(*, limit: int) -> Path:
    return WORK_ROOT / (f"run-smoke-{limit}" if limit else "run")


# --------------------------------------------------------------------------- the corpus


@dataclass(frozen=True, slots=True)
class SpeakerAudio:
    """One speaker's surviving clips and the aggregate the filter reads."""

    speaker: str
    sources: tuple[Source, ...]
    acoustics: dict[str, object]

    @property
    def hours(self) -> float:
        return sum(source.duration_s for source in self.sources) / 3600.0


def _statistic(payload: Mapping[str, object], field: str, key: str) -> float | None:
    """One number out of `agbalu.tts.acoustics.summarise`, or `None` where it measured none."""
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return None
    entry = fields.get(field)
    if not isinstance(entry, dict):
        return None
    value = entry.get(key)
    return float(value) if isinstance(value, int | float) else None


def admissible(
    acoustics: Mapping[str, object], hours: float, *, min_hours: float = MIN_SPEAKER_HOURS
) -> str | None:
    """The reason a speaker is refused from Stage 1, or `None` if it is taken."""
    if hours < min_hours:
        return "under-the-hours-floor"
    floor = _statistic(acoustics, "speech_to_floor_db", "median")
    if floor is None or floor < MIN_SPEECH_TO_FLOOR_DB:
        return "noisy"
    share = _statistic(acoustics, "speech_frame_share", "median")
    if share is None or share < MIN_SPEECH_FRAME_SHARE:
        return "mostly-silence"
    clips = acoustics.get("clips")
    zeroed = acoustics.get("clips_with_zeroed_silence")
    if (
        isinstance(clips, int)
        and isinstance(zeroed, int)
        and clips
        and zeroed / clips > MAX_ZEROED_SHARE
    ):
        return "silence-gated-to-digital-zero"
    return None


def inherited(manifest: Path) -> tuple[frozenset[str], dict[str, float]]:
    """12.4's held-out clips and its per-clip transcript error, taken whole.

    Inheriting the split makes MP.A1 and MP.E5 one measurement: Matoub-82M's three conditions
    were drawn from these clips, and an utterance Pro trained on would turn its own Cycle-CER
    into a memorisation score. Inheriting the error rates keeps 12.4's transcript gate on the
    voice without a second pass over Fadhma.
    """
    reserved = corpus_pro.held_out(manifest)
    rates: dict[str, float] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("cer")
        if isinstance(value, int | float):
            rates[str(row["clip"])] = float(value)
    return reserved, rates


def voice_speaker(report: Path) -> str:
    """Which Common Voice account the published voice is, read from 12.4's own report."""
    if not report.is_file():
        message = f"cannot name {VOICE}'s speaker: {report} is not on the volume"
        raise RuntimeError(message)
    payload = json.loads(report.read_text(encoding="utf-8"))
    voices = payload.get("voices")
    if isinstance(voices, list):
        for entry in voices:
            if isinstance(entry, dict) and entry.get("voice") == VOICE:
                return str(entry["speaker"])
    message = f"{report} names no voice called {VOICE}"
    raise RuntimeError(message)


def gather(
    clips: Sequence[Clip],
    audio: Mapping[str, Path],
    pool: ThreadPoolExecutor,
    *,
    reserved: frozenset[str],
    rates: Mapping[str, float],
) -> tuple[list[SpeakerAudio], dict[str, int]]:
    """Every speaker's surviving clips, grouped, with the aggregate the filter reads.

    `native` never resamples: the band is what a resample destroys, and the acoustic filter
    is a claim about the recording rather than about the pack it was read through.
    """
    from agbalu.tts.acoustics import profile

    def _one(clip: Clip) -> tuple[str, ClipProfile]:
        return clip.clip, profile(*native(audio[clip.clip]))

    profiles = dict(pool.map(_one, clips))
    dropped: dict[str, int] = {}
    sources: dict[str, list[Source]] = {}
    kept: dict[str, list[ClipProfile]] = {}
    for clip in clips:
        result = corpus_pro.verdict(
            clip, profiles[clip.clip], rates.get(clip.clip), reserved=reserved, cer_limit=CER_LIMIT
        )
        if isinstance(result, str):
            dropped[result] = dropped.get(result, 0) + 1
            continue
        sources.setdefault(clip.speaker, []).append(result)
        kept.setdefault(clip.speaker, []).append(profiles[clip.clip])
    speakers = [
        SpeakerAudio(speaker=name, sources=tuple(rows), acoustics=summarise(kept[name]))
        for name, rows in sources.items()
    ]
    speakers.sort(key=lambda item: item.hours, reverse=True)
    return speakers, dropped


def select(
    speakers: Sequence[SpeakerAudio],
    *,
    hours: float,
    keep: str,
    min_hours: float = MIN_SPEAKER_HOURS,
) -> tuple[list[SpeakerAudio], dict[str, int]]:
    """The Stage 1 pool: admissible speakers, longest first, up to the budget.

    `keep` is taken whether or not the budget has room, because it is the voice Stage 2
    fine-tunes and a Stage 1 that never heard it is the wrong backbone.
    """
    taken: list[SpeakerAudio] = []
    refused: dict[str, int] = {}
    budget = 0.0
    for speaker in speakers:
        if speaker.speaker == keep:
            taken.append(speaker)
            budget += speaker.hours
            continue
        reason = admissible(speaker.acoustics, speaker.hours, min_hours=min_hours)
        if reason is None and budget + speaker.hours > hours and taken:
            reason = "over-the-hours-budget"
        if reason is not None:
            refused[reason] = refused.get(reason, 0) + 1
            continue
        taken.append(speaker)
        budget += speaker.hours
    return taken, refused


def _restore_batch(
    sources: Sequence[Source], audio: Mapping[str, Path], restorer: Restorer
) -> dict[str, NDArray[np.float32]]:
    """One batch of clips through Sidon, keyed by clip name and left at Sidon's own rate."""
    prepared = []
    lengths = []
    names = []
    for source in sources:
        wave, rate = native(audio[source.clip])
        rows, samples = restorer.prepare(wave, rate)
        prepared.append(rows)
        lengths.append(samples)
        names.append(source.clip)
    restored = restorer.restore_prepared(prepared, lengths)
    return dict(zip(names, restored, strict=True))


def render(
    utterances: Sequence[Joined],
    sources: Mapping[str, Source],
    audio: Mapping[str, Path],
    restorer: Restorer,
    root: Path,
) -> dict[str, object]:
    """Write every utterance as one wav: restore each member, cut it, gain it, join them.

    The seam is the point. Interior members keep their own trailing and leading silence
    untouched, so the pause between two sentences — and the breath taken in it — is the
    speaker's rather than a constant chosen here. Only the outer edges are trimmed.

    Resumable at utterance granularity: a committed wav is never rewritten, which is what
    makes a preemption cost minutes instead of the run.
    """
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    written = {path.stem for path in root.glob("*.wav")}
    todo = [item for item in utterances if item.name not in written]
    _emit("render-start", utterances=len(utterances), todo=len(todo), root=str(root))

    ceiling = float(10.0 ** (PEAK_CEILING_DBFS / 20.0))
    started = time.monotonic()
    done = 0
    for index in range(0, len(todo), RESTORE_BATCH):
        chunk = todo[index : index + RESTORE_BATCH]
        needed = sorted({member.clip for item in chunk for member in item.members})
        waves = _restore_batch([sources[clip] for clip in needed], audio, restorer)
        for item in chunk:
            pieces = []
            for member in item.members:
                wave = np.asarray(waves[member.clip], dtype=np.float32)
                start = max(0, round(member.start_s * RESTORED_RATE))
                stop = min(int(wave.size), round(member.stop_s * RESTORED_RATE))
                if stop <= start:
                    message = f"{item.name} cuts {member.clip} to nothing"
                    raise RuntimeError(message)
                pieces.append(wave[start:stop] * np.float32(member.gain))
            joined = resample(np.concatenate(pieces), RESTORED_RATE, OUTPUT_RATE)
            peak = float(np.abs(joined).max())
            if peak > ceiling:
                joined = joined * np.float32(ceiling / peak)
            write_wav(root / f"{item.name}.wav", joined, OUTPUT_RATE)
            done += 1
            if done % RENDER_LOG_EVERY == 0:
                _emit("render", done=done, of=len(todo), seconds=round(time.monotonic() - started))
            if done % COMMIT_EVERY == 0:
                data_volume.commit()
    data_volume.commit()
    return {
        "written": done,
        "skipped": len(utterances) - len(todo),
        "seconds": round(time.monotonic() - started, 1),
    }


def write_lists(
    root: Path,
    utterances: Sequence[Joined],
    *,
    audio_dir: Path,
    speaker_ids: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """One list per split, in the format the recipe reads."""
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "dev"):
        rows = [item for item in utterances if item.split == split]
        lines = list(
            corpus_pro.training_list(rows, audio_dir=str(audio_dir), speaker_ids=speaker_ids)
        )
        (root / f"{split}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        counts[split] = len(rows)
    return counts


def write_manifest(root: Path, name: str, utterances: Sequence[Joined]) -> None:
    """Every decision taken per utterance, including the ones a filelist cannot carry."""
    with (root / f"{name}.manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in utterances:
            handle.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")


@app.function(
    image=tts_image,
    gpu=GPU,
    cpu=CORPUS_CPU,
    volumes={str(DATA_PATH): data_volume},
    timeout=CORPUS_TIMEOUT,
    retries=RETRIES,
)
def matoub_pro_corpus(
    *, hours: float = POOL_HOURS, limit: int = 0, force: bool = False
) -> dict[str, object]:
    """Both Pro corpora in one container: the voice Stage 2 fine-tunes and the Stage 1 pool.

    One container because every clip is profiled once and restored once, and both corpora are
    cut from the same decisions. `limit` caps the clips and is this task's smoke; it writes
    under its own root, so a capped call can never overwrite a real corpus.
    """
    import torch

    from agbalu.speech.corpus import read as read_clips

    _configure_logging()
    data_volume.reload()
    root = corpus_root(limit=limit)
    marker = root / "corpus_pro.stats.json"
    if marker.is_file() and not force:
        _emit("corpus-already", root=str(root))
        cached: dict[str, object] = json.loads(marker.read_text(encoding="utf-8"))
        return cached

    started = time.monotonic()
    speaker = voice_speaker(V1_REPORT)
    reserved, rates = inherited(V1_MANIFEST)
    clips = list(read_clips(REMOTE_SPEECH / "train.jsonl"))
    if limit:
        wanted = [clip for clip in clips if clip.speaker == speaker][:limit]
        clips = wanted + [clip for clip in clips if clip.speaker != speaker][:limit]
    audio = index_audio(REMOTE_AUDIO)
    absent = [clip.clip for clip in clips if clip.clip not in audio]
    if absent:
        message = (
            f"{len(absent)} of {len(clips)} clips are not on the volume (first: {absent[0]}); "
            f"run `make modal-asr TASK=fetch`"
        )
        raise RuntimeError(message)
    _emit("corpus-start", clips=len(clips), reserved=len(reserved), voice_speaker=speaker[:12])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor, vocoder = download(device.type)
    restorer = Restorer(predictor, vocoder, device)

    with ThreadPoolExecutor(max_workers=int(CORPUS_CPU)) as pool:
        speakers, dropped = gather(clips, audio, pool, reserved=reserved, rates=rates)
    _emit("profiled", speakers=len(speakers), dropped=sum(dropped.values()))
    min_hours = 0.0 if limit else MIN_SPEAKER_HOURS
    chosen, refused = select(speakers, hours=hours, keep=speaker, min_hours=min_hours)
    _emit(
        "pool",
        speakers=len(chosen),
        hours=round(sum(item.hours for item in chosen), 2),
        refused=refused,
    )

    names = {
        item.speaker: (VOICE if item.speaker == speaker else f"spk{index:03d}")
        for index, item in enumerate(sorted(chosen, key=lambda entry: entry.speaker))
    }
    reports: dict[str, object] = {}
    every: list[corpus_pro.Joined] = []
    lookup: dict[str, Source] = {}
    for item in chosen:
        name = names[item.speaker]
        built, dropped_utterances, target = corpus_pro.build(list(item.sources), name)
        if not built:
            refused["no-utterance-survived-packing"] = (
                refused.get("no-utterance-survived-packing", 0) + 1
            )
            continue
        every.extend(built)
        lookup.update({source.clip: source for source in item.sources})
        reports[name] = {
            **corpus_pro.report(
                name, item.speaker, built, dropped_utterances, target, considered=len(item.sources)
            ),
            "acoustics": item.acoustics,
        }

    audio_dir = root / "audio"
    rendered = render(every, lookup, audio, restorer, audio_dir)

    dev = corpus_pro.SMOKE_DEV if limit else corpus_pro.DEV_UTTERANCES
    voice_rows = corpus_pro.assign_dev([u for u in every if u.voice == VOICE], dev=dev)
    non_voice = [u for u in every if u.voice != VOICE]
    pool_rows = (
        corpus_pro.assign_dev(non_voice, dev=dev)
        if len(non_voice) > dev
        else [replace(u, split="train") for u in non_voice]
    )
    pool_all = [*pool_rows, *voice_rows]
    pool_speakers = corpus_pro.assign_speaker_ids([u.voice for u in pool_all])
    counts = {
        VOICE: write_lists(root / VOICE, voice_rows, audio_dir=audio_dir, speaker_ids={VOICE: 0}),
        POOL: write_lists(root / POOL, pool_all, audio_dir=audio_dir, speaker_ids=pool_speakers),
    }
    write_manifest(root, VOICE, voice_rows)
    write_manifest(root, POOL, pool_rows)

    payload: dict[str, object] = {
        "task": "MP.B3",
        "root": str(root),
        "audio": str(audio_dir),
        "voice": VOICE,
        "speaker": speaker,
        "hours_budget": hours,
        "clips_considered": len(clips),
        "held_out_by_12_4": len(reserved),
        "dropped": dict(sorted(dropped.items())),
        "speakers_refused": dict(sorted(refused.items())),
        "speakers_taken": len(reports),
        "speakers": reports,
        "lists": counts,
        "render": rendered,
        "seconds": round(time.monotonic() - started, 1),
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    _emit("corpus-done", lists=counts, render=rendered, seconds=payload["seconds"])
    return payload


# --------------------------------------------------------------------------- the recipe


def _git(*arguments: str) -> None:
    """A `git` in the vendored recipe. Failure is a `PreconditionError`, never a raw one."""
    executable = shutil.which("git")
    if executable is None:
        message = "git is not on PATH in this image, so the recipe patches cannot be applied"
        raise PreconditionError(message)
    finished = subprocess.run(  # noqa: S603
        [executable, "-C", str(STYLETTS2_PATH), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        message = (
            f"git {' '.join(arguments)} exited {finished.returncode} in {STYLETTS2_PATH}: "
            f"{finished.stderr.strip() or finished.stdout.strip()}"
        )
        raise PreconditionError(message)


def patches() -> tuple[Path, ...]:
    """Both diffs, in the order they apply. The pro patch is generated on top of the resume
    patch, so the order is the patch's own and not a preference."""
    for patch in (STAGE2_PATCH, PRO_PATCH):
        if not patch.is_file():
            message = f"no recipe patch at {patch.resolve()}; `resources/` is not mounted"
            raise PreconditionError(message)
    return (STAGE2_PATCH, PRO_PATCH)


def apply_patches(*, revert: bool = False) -> list[str]:
    """Patch a clean tree, optionally putting it back. Every failure is deterministic."""
    _git("checkout", "--", *PATCHED_FILES)
    applied: list[str] = []
    for patch in patches():
        _git("apply", str(patch.resolve()))
        applied.append(patch.name)
    if revert:
        _git("checkout", "--", *PATCHED_FILES)
    return applied


def select_ood(source: Path, destination: Path) -> int:
    """The OOD lines long enough for `slmadv` to predict past `SLMADV_MIN_LEN`.

    A file of shorter lines makes the branch skip every sample, which prints `d_loss_slm 0`
    and reads exactly like a healthy run.
    """
    if not source.is_file():
        message = f"no OOD text at {source}; run `make modal-upload TASK=tts`"
        raise PreconditionError(message)
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if len(line.strip()) >= MIN_OOD_PHONEMES
    ]
    if len(lines) < MIN_OOD_LINES:
        message = (
            f"{len(lines)} OOD lines reach {MIN_OOD_PHONEMES} phonemes, under the "
            f"{MIN_OOD_LINES} this run needs; the SLM branch would draw the same few forever"
        )
        raise PreconditionError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One stage's schedule, resolved once so the config and the resume path agree."""

    stage: str
    root: Path
    logs: Path
    train_list: Path
    val_list: Path
    pretrained: Path
    resuming: bool
    epochs: int
    max_len: int
    batch: int
    final: Path

    @property
    def load_only_params(self) -> bool:
        return not self.resuming

    @property
    def diff_epoch(self) -> int:
        return max(1, -(-self.epochs // DIFF_RATIO))

    @property
    def tma_epoch(self) -> int:
        return max(1, -(-self.epochs // TMA_RATIO))

    @property
    def joint_epoch(self) -> int:
        return max(1, -(-self.epochs // JOINT_RATIO))

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "epochs": self.epochs,
            "max_len": self.max_len,
            "batch": self.batch,
            "diff_epoch": self.diff_epoch,
            "tma_epoch": self.tma_epoch,
            "joint_epoch": self.joint_epoch,
            "pretrained": str(self.pretrained),
            "resuming": self.resuming,
        }


def render_config(plan: StagePlan, *, workers: int) -> Path:
    """The recipe's config, written from this repository rather than edited by hand.

    Deliberately not `modal_app.matoub.render_config`. That function records what produced
    `Matoub-82M` and is history; this is the corrected regime, and the differences — diffusion
    on, upstream's epoch ratios, a Kabyle SLM, `slmadv` lengths the corpus can satisfy — are
    the whole of Pro.
    """
    import yaml

    config = {
        "log_dir": str(plan.logs),
        "seed": SEED,
        "batch_size": plan.batch,
        "epochs": plan.epochs,
        "epochs_1st": plan.epochs,
        "epochs_2nd": plan.epochs,
        "save_freq": 1,
        "log_interval": 10,
        "max_len": plan.max_len,
        "pretrained_model": str(plan.pretrained),
        "first_stage_path": "first_stage.pth",
        "load_only_params": plan.load_only_params,
        "second_stage_load_pretrained": plan.resuming,
        "resuming_second_stage": plan.resuming,
        "F0_path": "Utils/JDC/bst.t7",
        "ASR_config": "Utils/ASR/config.yml",
        "ASR_path": "Utils/ASR/epoch_00080.pth",
        "PLBERT_dir": "Utils/PLBERT/",
        "data_params": {
            "train_data": str(plan.train_list),
            "val_data": str(plan.val_list),
            "root_path": str(plan.root),
            "OOD_data": str(plan.root / "OOD_texts.txt"),
            "min_length": MIN_OOD_PHONEMES,
            "num_workers": workers,
        },
        "preprocess_params": {
            "sr": OUTPUT_RATE,
            "spect_params": {
                "n_fft": 2048,
                "win_length": 1200,
                "hop_length": 300,
                "n_mels": 80,
                "fmin": 0,
                "fmax": 8000,
            },
        },
        "model_params": {
            "dim_in": 64,
            "n_token": 178,
            "hidden_dim": 512,
            "style_dim": 128,
            "max_dur": 50,
            "multispeaker": True,
            "n_mels": 80,
            "dropout": 0.2,
            "n_layer": 3,
            "text_encoder_kernel_size": 5,
            "decoder": {
                "type": "istftnet",
                "upsample_rates": [10, 6],
                "upsample_kernel_sizes": [20, 12],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "gen_istft_n_fft": 20,
                "gen_istft_hop_size": 5,
            },
            "diffusion": {
                "embedding_mask_proba": 0.1,
                "transformer": {
                    "num_layers": 3,
                    "num_heads": 8,
                    "head_features": 64,
                    "multiplier": 2,
                },
                "dist": {
                    "sigma_data": 0.2,
                    "estimate_sigma_data": True,
                    "mean": -3.0,
                    "std": 1.0,
                },
            },
            "plbert": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_hidden_layers": 12,
                "dropout": 0.1,
            },
            "slm": {
                "model": SLM_MODEL,
                "sr": 16000,
                "hidden": SLM_HIDDEN,
                "nlayers": SLM_LAYERS + 1,
                "initial_channel": 64,
            },
        },
        "loss_params": {
            "lambda_gen": 1.0,
            "lambda_mel": 5.0,
            "lambda_dur": 1.0,
            "lambda_ce": 20.0,
            "lambda_F0": 1.0,
            "lambda_norm": 1.0,
            "lambda_s2s": 1.0,
            "lambda_mono": 1.0,
            "lambda_slm": 1.0,
            "lambda_diff": 1.0,
            "lambda_sty": 1.0,
            "TMA_epoch": plan.tma_epoch,
            "diff_epoch": plan.diff_epoch,
            "joint_epoch": plan.joint_epoch,
        },
        "optimizer_params": {"lr": 0.0001, "bert_lr": 0.00001, "ft_lr": 0.0001},
        "slmadv_params": {
            "min_len": SLMADV_MIN_LEN,
            "max_len": SLMADV_MAX_LEN,
            "batch_percentage": BATCH_PERCENTAGE[plan.stage],
            "iter": 10,
            "thresh": 5,
            "scale": 0.01,
            "sig": 1.5,
        },
    }
    path = plan.root / f"config_{plan.stage}.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def stage_logs(root: Path, stage: str) -> Path:
    return root / "logs" / (VOICE if stage == STAGE2 else STAGE1)


def stage_two_setup(root: Path, *, epoch: int) -> Path:
    """Copy the named Stage 1 epoch in as `first_stage.pth`, the only path the recipe reads.

    `train_second.py` resolves `first_stage_path` against `log_dir` and ignores
    `pretrained_model` outside the resume branch, so naming a Stage 1 epoch does nothing on
    its own.
    """
    source = stage_logs(root, STAGE1)
    chosen = (
        latest_checkpoint(source, "epoch_1st")
        if epoch < 0
        else source / f"epoch_1st_{epoch:05d}.pth"
    )
    if chosen is None or not chosen.is_file():
        message = f"no Stage 1 checkpoint in {source} to continue from"
        raise PreconditionError(message)
    destination = stage_logs(root, STAGE2) / "first_stage.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        shutil.copy2(chosen, destination)
    return chosen


def plan_stage(
    stage: str, root: Path, corpus: Path, *, epochs: int, max_len: int, batch: int
) -> StagePlan:
    """One stage's plan, with its resume resolved against what is already on the volume."""
    logs = stage_logs(root, stage)
    logs.mkdir(parents=True, exist_ok=True)
    prefix = "epoch_1st" if stage == STAGE1 else "epoch_2nd"
    resumed = latest_checkpoint(logs, prefix)
    lists = corpus / (POOL if stage == STAGE1 else VOICE)
    if stage == STAGE1:
        pretrained = resumed if resumed is not None else root / "kokoro_base.pth"
    else:
        first = logs / "first_stage.pth"
        if not first.is_file():
            message = f"Stage 2 needs {first}; run stage1 first"
            raise PreconditionError(message)
        pretrained = resumed if resumed is not None else first
    return StagePlan(
        stage=stage,
        root=root,
        logs=logs,
        train_list=lists / "train.txt",
        val_list=lists / "dev.txt",
        pretrained=pretrained,
        resuming=resumed is not None,
        epochs=epochs,
        max_len=max_len or MAX_LEN[stage],
        batch=batch or BATCH[stage],
        final=logs / f"{prefix}_{epochs - 1:05d}.pth",
    )


# --------------------------------------------------------------------------- prepare


def check_lists(corpus: Path, vocabulary: Vocabulary) -> dict[str, object]:
    """Prove every filelist encodes, before a card is paid for.

    A phoneme with no row is corruption, not data: the front end covers everything the corpus
    can hold, so an unmapped symbol means the list is not what it claims.
    """
    from agbalu.tts import training

    out: dict[str, object] = {}
    for name in (VOICE, POOL):
        for split in ("train", "dev"):
            path = corpus / name / f"{split}.txt"
            selection = training.select(training.read_list(path), vocabulary)
            training.require_encodable(selection)
            if not selection.rows:
                message = f"{path} holds no rows the base can be trained on"
                raise PreconditionError(message)
            out[f"{name}/{split}"] = selection.as_dict()
    return out


@app.function(
    image=matoub_image,
    cpu=TRAIN_CPU,
    volumes={str(DATA_PATH): data_volume, "/checkpoints": checkpoint_volume},
    timeout=60 * 60,
)
def matoub_pro_prepare(*, limit: int = 0, force: bool = False) -> dict[str, object]:
    """Everything that can fail deterministically, on a container with no GPU attached.

    Both patches are applied and reverted, the symbol table is written, the base is converted
    and checked, every filelist is proved encodable and the OOD file is cut to a length
    `slmadv` can use. What reaches the GPU has already been shown to load.
    """
    from huggingface_hub import hf_hub_download

    from agbalu.tts.vocabulary import Vocabulary

    _configure_logging()
    data_volume.reload()
    root = work_root(limit=limit)
    marker = root / "prepared.json"
    if marker.is_file() and not force:
        _emit("prepared-already", root=str(root))
        cached: dict[str, object] = json.loads(marker.read_text(encoding="utf-8"))
        return cached

    corpus = corpus_root(limit=limit)
    if not (corpus / "corpus_pro.stats.json").is_file():
        return _refuse(f"no corpus at {corpus}; run `make modal-matoub-pro TASK=corpus`")
    audio = corpus / "audio"
    rendered = len(list(audio.glob("*.wav"))) if audio.is_dir() else 0
    if not rendered:
        return _refuse(f"no audio at {audio}; the corpus build wrote none")

    try:
        root.mkdir(parents=True, exist_ok=True)
        vocabulary = Vocabulary.load()
        assert_base_matches_vendored(
            Path(hf_hub_download(BASE_REPO, BASE_CONFIG, revision=BASE_REVISION)), vocabulary
        )
        symbols = write_symbol_table(Path(STYLETTS2_PATH) / "kokoro_symbols.py", vocabulary)
        scripts = assert_recipe_imports()
        applied = apply_patches(revert=True)
        weights = Path(hf_hub_download(BASE_REPO, BASE_WEIGHTS, revision=BASE_REVISION))
        tensors = convert_base(weights, root / "kokoro_base.pth")
        ood = select_ood(OOD_SOURCE, root / "OOD_texts.txt")
        lists = check_lists(corpus, vocabulary)
    except DeterministicError as refusal:
        return _refuse(str(refusal), root=str(root))

    payload: dict[str, object] = {
        "task": "MP.C2",
        "root": str(root),
        "corpus": str(corpus),
        "smoke": bool(limit),
        "base": {"repo": BASE_REPO, "revision": BASE_REVISION, "tensors": tensors},
        "symbol_rows": symbols,
        "assigned": dict(vocabulary.assigned),
        "slm": {"model": SLM_MODEL, "licence": SLM_LICENCE, "layers": SLM_LAYERS},
        "ood_lines": ood,
        "audio_files": rendered,
        "recipe_imports_verified": list(scripts),
        "patches_verified": applied,
        "lists": lists,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    _emit("prepared", root=str(root), ood_lines=ood, audio_files=rendered)
    return payload


# --------------------------------------------------------------------------- train


@app.function(
    image=matoub_image,
    gpu=GPU,
    cpu=TRAIN_CPU,
    volumes={str(DATA_PATH): data_volume, "/checkpoints": checkpoint_volume},
    timeout=TIMEOUT,
    retries=RETRIES,
)
def matoub_pro_train(
    *,
    stage: str = STAGE1,
    epochs: int = 10,
    batch: int = 0,
    max_len: int = 0,
    limit: int = 0,
    first_stage_epoch: int = -1,
) -> dict[str, object]:
    """One stage against a prepared directory. Spawned, resumable, and refusing rather than
    raising on anything a retry would repeat.

    `AGBALU_SLM_MIN_SHARE` is the guard that matters most: `SLMAdversarialLoss` skips every
    sample shorter than `slmadv_params.min_len` and `train_second.py` answers with a bare
    `continue`, so a joint epoch can run to the end with no adversary at all and a log that
    looks entirely healthy. A smoke sets the floor to zero, because a capped corpus is short
    by construction.
    """
    _configure_logging()
    if stage not in (STAGE1, STAGE2):
        return _refuse(f"unknown stage {stage!r}; expected {STAGE1!r} or {STAGE2!r}")
    data_volume.reload()
    root = work_root(limit=limit)
    corpus = corpus_root(limit=limit)
    if not (root / "prepared.json").is_file():
        return _refuse(f"{root} was never prepared; run `make modal-matoub-pro TASK=prepare`")

    started = time.monotonic()
    try:
        if stage == STAGE2:
            stage_two_setup(root, epoch=first_stage_epoch)
        plan = plan_stage(stage, root, corpus, epochs=epochs, max_len=max_len, batch=batch)
        if plan.final.is_file():
            _emit("already-trained", stage=stage, checkpoint=str(plan.final))
            return {"trained_this_run": False, "stage": stage, "checkpoint": str(plan.final)}
        apply_patches()
        config = render_config(plan, workers=int(TRAIN_CPU))
        os.environ["AGBALU_SLM_LAYERS"] = str(SLM_LAYERS)
        os.environ["AGBALU_SLM_MIN_SHARE"] = str(0.0 if limit else SLM_MIN_SHARE)
        os.environ["AGBALU_MEL_LEN_ST"] = "192"
        _emit("stage-plan", **plan.as_dict())
        script = "train_first.py" if stage == STAGE1 else "train_second.py"
        elapsed = run_stage(script, config, max_len=plan.max_len, batch=plan.batch)
    except CapacityError as refusal:
        return _refuse(str(refusal), stage=stage, max_len=max_len or MAX_LEN[stage], batch=batch)
    except DeterministicError as refusal:
        return _refuse(str(refusal), stage=stage)

    data_volume.commit()
    checkpoint_volume.commit()
    payload: dict[str, object] = {
        "task": "MP.D1" if stage == STAGE1 else "MP.D2",
        "trained_this_run": True,
        **plan.as_dict(),
        "checkpoints": sorted(path.name for path in plan.logs.glob("epoch_*.pth")),
        "seconds_this_run": round(elapsed, 1),
        "hours_this_run": round(elapsed / 3600, 3),
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    _emit("stage-done", **payload)
    return payload
