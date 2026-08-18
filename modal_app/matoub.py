"""Task 12.6: fine-tuning Kokoro-82M into a Kabyle voice.

The recipe is [`semidark/kikiri-tts`](https://github.com/semidark/kikiri-tts) (Apache-2.0)
over a patched StyleTTS2, pinned by commit in `modal_app.common`. Stage 1 adapts the base
to Kabyle across both voices; Stage 2 fine-tunes one voice out of it; the voicepack is what
inference loads. Stage 1 is not optional for a new language — Stage 2 loads its output.

**Every deterministic failure is made to happen on a CPU container.** `matoub_prepare`
asks for no GPU and no retries, and it is where the filelists are validated, the symbol
table is written, the base weights are converted and checked, and the audio is proved to
exist. What reaches the GPU has already been shown to load. The split exists because the
alternative was measured elsewhere in this project: a missing volume surfaced from inside
`from_pretrained` as a Hub lookup, five times, once per retry, with the card attached.

Six defects in the recipe are corrected here rather than worked around downstream. The
first three are Stage 1's and were found before it ran; the last three are Stage 2's, and
every one of them fails silently — a plausible checkpoint, a healthy loss curve, and a
number that is not measuring what its name says.

1. Its symbol table has Private Use Area placeholders on rows 7, 8 and 26 and no entry for
   `ħ`, `ʕ` or `ˤ`, and its `TextCleaner` drops what it cannot find. Unmodified it deletes
   three Kabyle consonants from every training target, silently, and the loss curve looks
   healthy throughout. `agbalu.tts.vocabulary` renders the table that replaces it.
2. `meldataset` casts the speaker column with `int()`. 12.4 writes the voice name there,
   so the lists are renumbered on the way in.
3. Its OOD text is German, and its sampler loops until it draws a line of 50+ characters.
   `make tts TASK=ood` builds the Kabyle replacement with that floor guaranteed.
4. `train_second.py` resolves `first_stage_path` against `log_dir` and ignores
   `pretrained_model` unless `second_stage_load_pretrained` is set, so naming a Stage 1
   epoch does nothing on its own. `stage_two_setup` copies the named checkpoint into the
   per-voice `log_dir` as `first_stage.pth`, which is the only path the recipe reads.
5. Resuming Stage 2 at all requires `second_stage_load_pretrained`, which was written
   `false` unconditionally — so `pretrained_model` and `load_only_params` were read by a
   branch that never ran, and a preempted Stage 2 silently reloaded `first_stage.pth` and
   restarted from epoch zero with a fresh optimizer. On a function that retries five times,
   that is a run reporting six epochs having trained the last one.
6. That resume branch re-seeds `predictor_encoder` from `style_encoder` after loading, which
   is right on the way out of Stage 1 — where the module is untrained — and discards several
   epochs of it on the way back into Stage 2.
7. It binds `ref` only under `if multispeaker and epoch >= diff_epoch`, and reads it under
   `if epoch >= joint_epoch`. With style diffusion off — which is this project's
   configuration, because Kokoro loads a fixed voicepack — the name never exists and the
   first joint-training step raises `UnboundLocalError`, hours into a paid run. `slmadv`
   already handles the case: its `ref_s` defaults to `None` and goes unread when diffusion
   is disabled, so the caller is bound to `None` rather than made to compute encoders whose
   output nothing reads.

Defects 6 and 7 are the two hunks of `resources/styletts2_stage2_resume.patch`, applied to
the working tree before `train_second.py` runs and `--check`ed on a CPU container first.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from agbalu.tts import training
from agbalu.tts.corpus import ARMS, DEV, TRAIN
from agbalu.tts.vocabulary import Vocabulary
from modal_app.common import (
    DATA_PATH,
    MATOUB_BATCH,
    MATOUB_CPU,
    MATOUB_GPU,
    MATOUB_TIMEOUT,
    RESOURCES,
    RETRIES,
    STYLETTS2_PATH,
    app,
    call_owner,
    checkpoint_volume,
    data_volume,
    matoub_image,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

BASE_REPO: Final = "hexgrad/Kokoro-82M"
BASE_WEIGHTS: Final = "kokoro-v1_0.pth"
BASE_CONFIG: Final = "config.json"
BASE_REVISION: Final = "f3ff3571791e39611d31c381e3a41a3af07b4987"
"""The revision `resources/kokoro_vocabulary.json` was vendored from. Pinned so the table
this repository asserts against is the table the weights were published with."""

VOICES: Final = ("kab_male", "kab_female")
CORPUS_ROOT: Final = Path(DATA_PATH) / "tts" / "voices"
WORK_ROOT: Final = Path(DATA_PATH) / "tts" / "matoub"
OOD_SOURCE: Final = Path(DATA_PATH) / "tts" / "ood_texts.txt"

STAGE1: Final = "stage1"
STAGE2: Final = "stage2"

STAGE2_PATCH: Final = RESOURCES / "styletts2_stage2_resume.patch"
"""Applied to `train_second.py` before it runs, and `--check`ed on a CPU container first.

Against the pinned commit, so it either applies exactly or the pin moved. Kept as a diff
rather than as a rewrite of the file because the fork is 954 lines of somebody else's code
and one guarded line is the whole change."""

PLAN_MARKER: Final = "stage2.json"
"""Which Stage 1 epoch a Stage 2 directory was started from.

A resumed Stage 2 loads its own `epoch_2nd_*.pth` and never reads `first_stage.pth` again,
so a `FROM=` that disagrees with the one the directory was started on would be accepted and
ignored. Recorded here, and refused on the next launch rather than silently dropped."""


class DeterministicError(RuntimeError):
    """Anything that will fail identically on all five retries.

    `matoub_train` returns these as refusals rather than letting them out: `retries` is five
    and cannot tell a bug from a preemption, so raising buys the same diagnosis five times,
    each on a container with an A10G already attached. It has now cost that twice.
    """


class PreconditionError(DeterministicError):
    """Something knowable before a step runs.

    A directory that was never prepared, a marker predating a check, a Stage 1 checkpoint
    that is not there, a `FROM=` contradicting the one a directory was started on, a recipe
    patch that will not apply.
    """


class CapacityError(DeterministicError):
    """The run does not fit the card it was given, at this `max_len` and batch size.

    Deterministic in the way that matters: the allocation that failed is the same one on
    every attempt. Detected by reading the recipe's own output, because the OOM is raised
    inside `train_second.py` and reaches this process only as an exit code.
    """


def _refuse(reason: str, **fields: object) -> dict[str, object]:
    """Report a precondition failure as this call's result, so Modal does not retry it."""
    payload: dict[str, object] = {
        "task": "12.6",
        "trained_this_run": False,
        "refused": reason,
        **fields,
    }
    _emit("refused", **payload)
    return payload


DIFF_EPOCH: Final = 999
"""Style diffusion off, matching `lambda_diff` and `lambda_sty` at 0.0. Kokoro's inference
path loads a fixed voicepack as the speaker and never samples one. The recipe's own config
writes the same number with the same comment."""

MAX_LEN: Final[dict[str, int]] = {STAGE1: 200, STAGE2: 100}
"""Frames of audio per training example, per stage. One frame is `hop_length / sr` = 12.5 ms.

Upstream names this as the lever to lower on an OOM, and the stages do not need the same
value: Stage 1 trained through six epochs at 200, while Stage 2 from `joint_epoch` onward
also runs the waveform decoder, MPD, MSD and WavLM over the decoded segment, and 200 does
not fit an A10G there. `mel_len` is `max_len // 2`, so 100 gives the decoder 0.625 s."""

COMPONENTS: Final = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")
"""What the converted checkpoint must carry. StyleTTS2 loads by component name, and a
missing one is not an error there — it simply trains that module from scratch."""

log: Final = logging.getLogger("agbalu.matoub")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _emit(event: str, **fields: object) -> None:
    log.info("%s %s", event, json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str))


def work_root(arm: str, *, limit: int) -> Path:
    """Where one arm's run lives, keyed on its cap as well as its arm.

    A capped run never shares a directory with the real one, and — the part that was wrong
    first time — no two caps share one either. `LIMIT=200` and `LIMIT=4000` both resolving
    to `<arm>-smoke` meant the second probe read the first's `epoch_1st_00000.pth`, decided
    it was already complete, and measured nothing, while its own prepare had already
    overwritten the lists the first had trained on.
    """
    return WORK_ROOT / (f"{arm}-smoke-{limit}" if limit else arm)


def assert_base_matches_vendored(config_path: Path, vocabulary: Vocabulary) -> None:
    """Refuse to train if the published table is not the one this repository assigned into.

    The three Kabyle rows are free rows of a specific revision. If upstream reassigns any of
    them, the fine-tune would overwrite a trained embedding and nothing downstream would say
    so — the model would simply be worse.
    """
    published: dict[str, int] = json.loads(config_path.read_text(encoding="utf-8"))["vocab"]
    ours = {s: i for s, i in vocabulary.symbols.items() if s not in vocabulary.assigned}
    ours.pop("$", None)
    if ours != published:
        differing = sorted(
            symbol
            for symbol in set(ours) | set(published)
            if ours.get(symbol) != published.get(symbol)
        )
        message = (
            f"{BASE_REPO}@{BASE_REVISION} does not carry the vocabulary vendored in "
            f"resources/kokoro_vocabulary.json; {len(differing)} symbols differ: {differing[:8]}"
        )
        raise RuntimeError(message)
    taken = set(published.values()) | {vocabulary.pad_index}
    collisions = sorted(set(vocabulary.assigned.values()) & taken)
    if collisions:
        message = f"rows {collisions} are no longer free in the published table"
        raise RuntimeError(message)


def write_symbol_table(path: Path, vocabulary: Vocabulary) -> int:
    """Render the table StyleTTS2 imports, with the Kabyle rows in it.

    A `TextCleaner` that raises rather than skipping: the lists are validated before they
    get here, so anything it cannot place means the two disagree, and silence there is what
    puts a mutilated target into the loss.
    """
    symbols = vocabulary.symbol_list()
    body = "\n".join(f"    {symbol!r}," for symbol in symbols)
    source = f'''"""Generated by modal_app.matoub from resources/kokoro_vocabulary.json.

Do not edit. Rows {sorted(vocabulary.assigned.values())} carry {"".join(vocabulary.assigned)},
which the recipe's own table leaves as Private Use Area placeholders.
"""

symbols = [
{body}
]

dicts = {{sym: i for i, sym in enumerate(symbols)}}


class TextCleaner:
    """Raises on an unmapped symbol instead of dropping it."""

    def __init__(self, dummy=0):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            index = self.word_index_dictionary.get(char)
            if index is None:
                raise KeyError(
                    "no embedding row for %r U+%04X in %r" % (char, ord(char), text)
                )
            indexes.append(index)
        return indexes


assert len(symbols) == {vocabulary.n_token}, len(symbols)
'''
    path.write_text(source, encoding="utf-8")
    return len(symbols)


RECIPE_SCRIPTS: Final = ("train_first", "train_second")


def assert_recipe_imports() -> tuple[str, ...]:
    """Import both training scripts here, on a container with no GPU, before spawning one.

    This is the check whose absence cost a card. `matoub_prepare` validated this project's
    inputs and not the recipe's own import closure, so a package missing from StyleTTS2's
    `requirements.txt` — `pandas`, which `meldataset` imports — surfaced only after an A10G
    had started, and surfaced five times because the training function retries.

    Both scripts are import-safe at module scope: a `torch.load` monkeypatch, a warnings
    filter, a logger, and a `__main__` guard. Importing them also proves the generated
    `kokoro_symbols.py` parses and that `text_utils` can read the table out of it, which no
    other check covers.
    """
    for script in RECIPE_SCRIPTS:
        finished = subprocess.run(  # noqa: S603
            [sys.executable, "-c", f"import {script}"],
            cwd=str(STYLETTS2_PATH),
            capture_output=True,
            text=True,
            check=False,
        )
        if finished.returncode != 0:
            tail = (finished.stderr or finished.stdout).strip().splitlines()[-4:]
            message = f"{script}.py cannot be imported in this image: " + " | ".join(tail)
            raise RuntimeError(message)
    return RECIPE_SCRIPTS


def convert_base(weights: Path, destination: Path) -> dict[str, int]:
    """Kokoro's checkpoint in the shape StyleTTS2 loads: `module.` stripped, wrapped in `net`."""
    import torch

    state = torch.load(weights, map_location="cpu", weights_only=False)
    net = {
        component: {key.removeprefix("module."): tensor for key, tensor in tensors.items()}
        for component, tensors in state.items()
    }
    missing = [component for component in COMPONENTS if component not in net]
    if missing:
        message = f"{BASE_WEIGHTS} carries no {missing}; StyleTTS2 would train them from scratch"
        raise RuntimeError(message)
    torch.save({"net": net}, destination)
    return {component: len(tensors) for component, tensors in net.items()}


def _lists_for(voice: str, arm: str, split: str) -> Path:
    return CORPUS_ROOT / voice / arm / f"{voice}.{split}.txt"


def build_lists(
    root: Path, arm: str, voices: Sequence[str], vocabulary: Vocabulary, *, limit: int
) -> dict[str, object]:
    """Validate 12.4's lists and write what both stages read, with integer speakers.

    Four lists per split, not one: the merged list Stage 1 trains on, and a list per voice
    for Stage 2, which is single-speaker. The speaker ids are assigned once over every voice
    and reused in both, so the integer a voice answers to in the fine-tune is the integer it
    was conditioned on in the base — renumbering a lone voice to 0 would address a different
    row of the speaker embedding Stage 1 spent eighteen hours training.
    """
    ids = training.assign_speaker_ids(voices)
    report: dict[str, object] = {"speaker_ids": ids, "splits": {}}
    splits: dict[str, object] = {}

    for name, source_split in ((training.TRAIN_LIST, TRAIN), (training.VAL_LIST, DEV)):
        selections = []
        for voice in voices:
            rows = training.read_list(_lists_for(voice, arm, source_split))
            selection = training.select(rows, vocabulary)
            training.require_encodable(selection)
            selections.append(selection)
        merged = training.merge(selections)
        if limit:
            merged = merged[:limit]
        written = training.write_list(root / name, training.renumber(merged, ids))
        per_voice: dict[str, object] = {}
        for voice, chosen in zip(voices, selections, strict=True):
            kept = chosen.rows[:limit] if limit else chosen.rows
            alone = training.write_list(
                root / training.voice_list(name, voice), training.renumber(kept, ids)
            )
            per_voice[voice] = {**chosen.as_dict(), "stage2_rows": alone}
        splits[name] = {"rows": written, "per_voice": per_voice}
    report["splits"] = splits
    return report


def assert_audio_present(root: Path) -> int:
    """Prove every path in the list resolves, before the loader discovers one does not.

    Every row, not a sample. A sample catches the failures that are total — a wrong arm, an
    unmounted volume — and misses the one that matters after moving a corpus between
    accounts: a transfer that dropped some fraction of 23,000 small files. That surfaces
    mid-epoch on a card, where a stat per row on a CPU container costs seconds.
    """
    rows = training.read_list(root / training.TRAIN_LIST)
    absent = [row.audio for row in rows if not Path(row.audio).is_file()]
    if absent:
        message = (
            f"{len(absent)} of {len(rows)} clips named by the training list are not on the "
            f"volume; first missing: {absent[0]}"
        )
        raise RuntimeError(message)
    return len(rows)


def render_config(root: Path, plan: StagePlan, *, batch: int, workers: int) -> Path:
    """The recipe's config, written from this repository rather than edited by hand.

    `multispeaker` stays `true` in both stages because that is what the published base
    declares, and the flag is architectural: flipping it between stages would leave Stage 2
    unable to load what Stage 1 wrote. Every schedule number comes off the plan, which is
    where the recipe's own epoch arithmetic is accounted for.
    """
    import yaml

    config = {
        "log_dir": str(plan.logs),
        "batch_size": batch,
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
            "train_data": str(root / plan.train_list),
            "val_data": str(root / plan.val_list),
            "root_path": str(root),
            "OOD_data": str(root / training.OOD_LIST),
            "min_length": training.MIN_OOD_PHONEMES,
            "num_workers": workers,
        },
        "preprocess_params": {
            "sr": 24000,
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
                "model": "microsoft/wavlm-base-plus",
                "sr": 16000,
                "hidden": 768,
                "nlayers": 13,
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
            "lambda_diff": 0.0,
            "lambda_sty": 0.0,
            "TMA_epoch": 0,
            "diff_epoch": plan.diff_epoch,
            "joint_epoch": plan.joint_epoch,
        },
        "optimizer_params": {"lr": 0.0001, "bert_lr": 0.00001, "ft_lr": 0.0001},
        "slmadv_params": {
            "min_len": 100,
            "max_len": 500,
            "batch_percentage": 0.5,
            "iter": 10,
            "thresh": 5,
            "scale": 0.01,
            "sig": 1.5,
        },
    }
    stem = f"{plan.stage}_{plan.voice}" if plan.voice else plan.stage
    path = root / f"config_{stem}.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def latest_checkpoint(logs: Path, prefix: str) -> Path | None:
    """The newest epoch checkpoint a stage wrote, so a restart resumes rather than repeats."""
    found = sorted(logs.glob(f"{prefix}_*.pth"))
    return found[-1] if found else None


def first_stage_for(logs: Path, *, epoch: int) -> tuple[Path, int]:
    """Which Stage 1 checkpoint Stage 2 continues from, and the epoch index it carries.

    Always a concrete `epoch_1st_NNNNN.pth` rather than the `first_stage.pth` copy, so the
    epoch is readable from the filename: Stage 2 is copied its choice under the recipe's own
    name, and what that copy was would otherwise be unrecoverable.

    `epoch` of -1 takes the newest. That is a default, not a recommendation: the recipe
    saves every epoch and prints a validation loss per epoch but records no best, and its
    users have reported Stage 2 overfitting from epoch 4 on a comparable corpus. Naming an
    epoch is how the operator acts on those numbers instead of being handed whichever ran
    last.
    """
    if epoch < 0:
        newest = latest_checkpoint(logs, "epoch_1st")
        if newest is None:
            message = f"Stage 2 needs a Stage 1 checkpoint in {logs}; run stage1 first"
            raise PreconditionError(message)
        chosen = newest
    else:
        chosen = logs / f"epoch_1st_{epoch:05d}.pth"
        if not chosen.is_file():
            available = sorted(path.name for path in logs.glob("epoch_1st_*.pth"))
            message = f"no Stage 1 checkpoint for epoch {epoch} in {logs}; have {available}"
            raise PreconditionError(message)
    return chosen, int(chosen.stem.rsplit("_", 1)[1])


def stage_logs(root: Path, *, stage: str, voice: str) -> Path:
    """Where a stage writes. Stage 2 gets a directory per voice.

    The recipe saves `epoch_2nd_%05d.pth` into `log_dir` and resolves `first_stage_path`
    against it, so two voices sharing one directory collide on both names: the second voice
    launched would resume from the first voice's checkpoint, and neither run would be the
    single-speaker model it was started for.
    """
    return root / "logs" if stage == STAGE1 else root / "logs" / voice


def recorded_first_stage(logs: Path) -> int | None:
    """The Stage 1 epoch this directory was started from, or `None` for a fresh one."""
    marker = logs / PLAN_MARKER
    if not marker.is_file():
        return None
    return int(json.loads(marker.read_text(encoding="utf-8"))["first_stage_epoch"])


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One launch, resolved against what is already on the volume.

    `resuming` is the whole of Stage 2's restart path: the recipe reads `pretrained_model`
    and `load_only_params` only inside the branch that key opens, so writing it `false`
    unconditionally left a preempted run reloading Stage 1 and starting over. `final` is the
    filename the last epoch writes, and is what "already complete" is allowed to mean.
    """

    stage: str
    voice: str
    logs: Path
    pretrained: Path
    train_list: str
    val_list: str
    epochs: int
    joint_epoch: int
    diff_epoch: int
    max_len: int
    load_only_params: bool
    resuming: bool
    final: Path
    first_stage: Path | None
    first_stage_epoch: int

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "voice": self.voice,
            "logs": str(self.logs),
            "pretrained": str(self.pretrained),
            "train_list": self.train_list,
            "epochs": self.epochs,
            "joint_epoch": self.joint_epoch,
            "max_len": self.max_len,
            "resuming": self.resuming,
            "final": str(self.final),
            "first_stage_epoch": self.first_stage_epoch,
        }


def stage_two_setup(plan: StagePlan, *, voice: str) -> None:
    """Put the per-voice directory into the state `train_second.py` reads it from."""
    if plan.first_stage is None:
        message = "stage2 resolved no Stage 1 checkpoint"
        raise PreconditionError(message)
    # `train_second.py` joins `first_stage_path` onto `log_dir` and never reads the absolute
    # `pretrained_model` on this path, so the named epoch has to be carried in.
    shutil.copyfile(plan.first_stage, plan.logs / "first_stage.pth")
    (plan.logs / PLAN_MARKER).write_text(
        json.dumps({"first_stage_epoch": plan.first_stage_epoch, "voice": voice}) + "\n",
        encoding="utf-8",
    )
    apply_stage2_patch()


def check_preconditions(
    root: Path,
    *,
    stage: str,
    arm: str,
    voice: str,
    limit: int,
    epochs: int,
    first_stage_epoch: int,
    max_len: int = 0,
) -> StagePlan:
    """Everything that must hold before a step runs, raising `PreconditionError` if it does not.

    Separate from the entrypoint so the whole class can be caught in one place, and so it is
    testable without a container.
    """
    if stage == STAGE2 and voice not in VOICES:
        message = f"stage2 fine-tunes one voice; pass VOICE=<{'|'.join(VOICES)}>, not {voice!r}"
        raise PreconditionError(message)
    if stage == STAGE1 and voice:
        message = f"stage1 is multi-speaker and trains every voice at once; drop VOICE={voice}"
        raise PreconditionError(message)

    marker = root / "prepared.json"
    capped = f" LIMIT={limit}" if limit else ""
    if not marker.is_file():
        message = (
            f"nothing prepared at {root}; run `make modal-matoub TASK=prepare "
            f"ARM={arm}{capped}` first"
        )
        raise PreconditionError(message)
    prepared = json.loads(marker.read_text(encoding="utf-8"))
    required = ("recipe_imports_verified", *(("stage2_patch_verified",) if stage == STAGE2 else ()))
    missing = [key for key in required if not prepared.get(key)]
    if missing:
        message = (
            f"{marker} predates {' and '.join(missing)}, so nothing has proved on a CPU "
            f"container what this run is about to depend on; re-run `make modal-matoub "
            f"TASK=prepare ARM={arm}{capped} FORCE=1`"
        )
        raise PreconditionError(message)
    return plan_stage(
        root,
        stage=stage,
        voice=voice,
        epochs=epochs,
        first_stage_epoch=first_stage_epoch,
        max_len=max_len,
    )


def plan_stage(
    root: Path,
    *,
    stage: str,
    voice: str,
    epochs: int,
    first_stage_epoch: int,
    max_len: int = 0,
) -> StagePlan:
    """Resolve one launch against what is already on the volume.

    `max_len` of 0 takes the stage's default; anything else is the operator overriding it,
    which is how an OOM is walked down without a code change.
    """
    logs = stage_logs(root, stage=stage, voice=voice)
    joint = max(1, epochs // 3)
    frames = max_len or MAX_LEN[stage]

    if stage == STAGE1:
        resumed = latest_checkpoint(logs, "epoch_1st")
        return StagePlan(
            stage=stage,
            voice="",
            logs=logs,
            pretrained=resumed if resumed is not None else root / "kokoro_base.pth",
            train_list=training.TRAIN_LIST,
            val_list=training.VAL_LIST,
            epochs=epochs,
            joint_epoch=joint,
            diff_epoch=DIFF_EPOCH,
            max_len=frames,
            load_only_params=resumed is None,
            resuming=False,
            final=logs / f"epoch_1st_{epochs - 1:05d}.pth",
            first_stage=None,
            first_stage_epoch=-1,
        )

    recorded = recorded_first_stage(logs)
    if recorded is not None and first_stage_epoch >= 0 and first_stage_epoch != recorded:
        message = (
            f"{logs} continues Stage 1 epoch {recorded}, and a resumed Stage 2 never reads "
            f"the first stage again — FROM={first_stage_epoch} would be accepted and have "
            f"no effect. Relaunch with FROM={recorded}, or start epoch {first_stage_epoch} "
            f"in a directory of its own"
        )
        raise PreconditionError(message)
    first, first_epoch = first_stage_for(
        root / "logs", epoch=recorded if recorded is not None else first_stage_epoch
    )
    resumed = latest_checkpoint(logs, "epoch_2nd")
    return StagePlan(
        stage=stage,
        voice=voice,
        logs=logs,
        pretrained=resumed if resumed is not None else first,
        train_list=training.voice_list(training.TRAIN_LIST, voice),
        val_list=training.voice_list(training.VAL_LIST, voice),
        epochs=epochs,
        joint_epoch=joint,
        diff_epoch=DIFF_EPOCH,
        max_len=frames,
        load_only_params=resumed is None,
        resuming=resumed is not None,
        final=logs / f"epoch_2nd_{epochs - 1:05d}.pth",
        first_stage=first,
        first_stage_epoch=first_epoch,
    )


def _git(*arguments: str) -> None:
    """A `git` in the vendored recipe. Failure is a `PreconditionError`, never a raw one.

    A patch that will not apply says the pin moved, and it says it identically on every
    retry — so it has to leave through the same door as the rest of the preconditions.
    """
    executable = shutil.which("git")
    if executable is None:
        message = "git is not on PATH in this image, so the recipe patch cannot be applied"
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


def _patch_path() -> str:
    if not STAGE2_PATCH.is_file():
        message = f"no recipe patch at {STAGE2_PATCH.resolve()}; `resources/` is not mounted"
        raise PreconditionError(message)
    return str(STAGE2_PATCH.resolve())


def check_stage2_patch() -> str:
    """Prove the diff still applies against the pinned commit, on a container with no GPU."""
    _git("apply", "--check", _patch_path())
    return STAGE2_PATCH.name


def apply_stage2_patch() -> None:
    """Guard the recipe's resume-time `predictor_encoder` reset.

    `checkout` first so a retried container patches a clean file: `git apply` refuses a diff
    that is already applied, and a preemption is the case this patch exists for.
    """
    _git("checkout", "--", "train_second.py")
    _git("apply", _patch_path())


OOM_MARKERS: Final = ("CUDA out of memory", "OutOfMemoryError")
"""What an exhausted card looks like in the recipe's output.

It raises inside `train_second.py`, so this process sees only an exit code; without reading
for it, an OOM is indistinguishable from a preemption and buys five containers."""


def run_stage(script: str, config: Path, *, max_len: int, batch: int) -> float:
    """One training stage, as its own process, with its output relayed line by line.

    A subprocess because `train_first.py` and `train_second.py` are scripts with a
    module-level `main(config_path)`, not a library. Output is streamed rather than
    captured so a container that stalls shows where.
    """
    started = time.monotonic()
    command = [sys.executable, script, "--config_path", str(config)]
    environment = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    exhausted = False
    _emit("stage-start", script=script, config=str(config), cwd=str(STYLETTS2_PATH))
    with subprocess.Popen(  # noqa: S603
        command,
        cwd=str(STYLETTS2_PATH),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            message = f"{script} produced no readable output stream"
            raise RuntimeError(message)
        for line in process.stdout:
            log.info("%s", line.rstrip())
            exhausted = exhausted or any(marker in line for marker in OOM_MARKERS)
        code = process.wait()
    elapsed = time.monotonic() - started
    if code != 0 and exhausted:
        message = (
            f"{script} ran out of memory after {elapsed:.0f}s at max_len={max_len} "
            f"batch={batch} on a {MATOUB_GPU}. Joint training adds the waveform decoder and "
            f"both discriminators, so an epoch before `joint_epoch` proves nothing about one "
            f"after it. Lower it — `MAXLEN=<n>`, halving the decoder's segment each time — "
            f"and re-smoke; the same allocation fails on every retry, so this run stops here"
        )
        raise CapacityError(message)
    if code != 0:
        message = f"{script} exited {code} after {elapsed:.0f}s; see the log above"
        raise RuntimeError(message)
    return elapsed


@app.function(
    image=matoub_image,
    cpu=MATOUB_CPU,
    volumes={str(DATA_PATH): data_volume},
    timeout=60 * 60,
)
def matoub_prepare(
    *, arm: str = "restored", limit: int = 0, force: bool = False
) -> dict[str, object]:
    """Everything that can fail deterministically, on a container with no GPU attached.

    No retries: every failure reachable here — a list that is not on the volume, a phoneme
    with no row, a base whose vocabulary moved, audio that was never rendered — returns the
    same answer five times, and the point of this function is that none of them cost a card.
    """
    from huggingface_hub import hf_hub_download

    _configure_logging()
    if arm not in ARMS:
        message = f"unknown arm {arm!r}; the corpus has {list(ARMS)}"
        raise RuntimeError(message)

    root = work_root(arm, limit=limit)
    marker = root / "prepared.json"
    if marker.is_file() and not force:
        _emit("prepared-already", root=str(root))
        payload: dict[str, object] = json.loads(marker.read_text(encoding="utf-8"))
        return payload
    root.mkdir(parents=True, exist_ok=True)

    vocabulary = Vocabulary.load()
    config_path = Path(hf_hub_download(BASE_REPO, BASE_CONFIG, revision=BASE_REVISION))
    assert_base_matches_vendored(config_path, vocabulary)
    rows = write_symbol_table(Path(STYLETTS2_PATH) / "kokoro_symbols.py", vocabulary)

    lists = build_lists(root, arm, VOICES, vocabulary, limit=limit)
    sampled = assert_audio_present(root)

    if not OOD_SOURCE.is_file():
        message = f"no OOD text at {OOD_SOURCE}; run `make tts TASK=ood` and upload it"
        raise RuntimeError(message)
    ood = [line for line in OOD_SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    training.write_ood(root / training.OOD_LIST, ood)

    weights = Path(hf_hub_download(BASE_REPO, BASE_WEIGHTS, revision=BASE_REVISION))
    tensors = convert_base(weights, root / "kokoro_base.pth")
    imported = assert_recipe_imports()
    patched = check_stage2_patch()

    payload = {
        "task": "12.6",
        "arm": arm,
        "smoke": bool(limit),
        "root": str(root),
        "base": {"repo": BASE_REPO, "revision": BASE_REVISION, "tensors": tensors},
        "symbol_rows": rows,
        "assigned": dict(vocabulary.assigned),
        "audio_sampled": sampled,
        "ood_lines": len(ood),
        "recipe_imports_verified": list(imported),
        "stage2_patch_verified": patched,
        **lists,
    }
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    _emit("prepared", **{k: v for k, v in payload.items() if not isinstance(v, (dict, list))})
    return payload


@app.function(
    image=matoub_image,
    gpu=MATOUB_GPU,
    cpu=MATOUB_CPU,
    volumes={str(DATA_PATH): data_volume, "/checkpoints": checkpoint_volume},
    timeout=MATOUB_TIMEOUT,
    retries=RETRIES,
)
def matoub_train(
    *,
    stage: str = STAGE1,
    arm: str = "restored",
    voice: str = "",
    epochs: int = 10,
    batch: int = MATOUB_BATCH,
    limit: int = 0,
    first_stage_epoch: int = -1,
    max_len: int = 0,
    run: str = "",
) -> dict[str, object]:
    """One stage against a prepared directory. Spawned, locked, and resumable.

    The lock is keyed on the call id rather than the task id: Modal gives a preempted
    container a new task id, so a task-keyed lock refuses its own retry and burns every one
    of them. A stage already at its last epoch returns instead of retraining — a resumed run
    that trains nothing otherwise reports the checkpoint's own counters as a measurement —
    and which file that is comes off `StagePlan`, because Stage 2's epochs are numbered from
    the Stage 1 checkpoint it continues rather than from zero.

    `voice` is required for Stage 2 and refused for Stage 1. Refused rather than ignored:
    the two stages read different filelists, and a Stage 1 that silently accepted a voice
    would report the fine-tune the operator asked for and train the base.
    """
    _configure_logging()
    root = work_root(arm, limit=limit)
    try:
        plan = check_preconditions(
            root,
            stage=stage,
            arm=arm,
            voice=voice,
            limit=limit,
            epochs=epochs,
            first_stage_epoch=first_stage_epoch,
            max_len=max_len,
        )
        plan.logs.mkdir(parents=True, exist_ok=True)
        if plan.final.is_file():
            _emit("already-complete", stage=stage, voice=voice, checkpoint=str(plan.final))
            return {
                "stage": stage,
                "arm": arm,
                "voice": voice,
                "trained_this_run": False,
                "checkpoint": str(plan.final),
            }
        if stage == STAGE2:
            stage_two_setup(plan, voice=voice)
    except PreconditionError as refusal:
        return _refuse(str(refusal), stage=stage, arm=arm, voice=voice)

    prefix = "epoch_1st" if stage == STAGE1 else "epoch_2nd"
    config = render_config(root, plan, batch=batch, workers=int(MATOUB_CPU))
    owner = call_owner()
    _emit(
        "start",
        arm=arm,
        batch=batch,
        owner=owner,
        run=run or "matoub",
        **plan.as_dict(),
    )

    script = "train_first.py" if stage == STAGE1 else "train_second.py"
    try:
        elapsed = run_stage(script, config, max_len=plan.max_len, batch=batch)
    except CapacityError as refusal:
        return _refuse(str(refusal), stage=stage, arm=arm, voice=voice, max_len=plan.max_len)

    written = latest_checkpoint(plan.logs, prefix)
    if written is None:
        message = f"{script} exited cleanly but wrote no {prefix}_*.pth into {plan.logs}"
        raise RuntimeError(message)
    if written != plan.final:
        message = (
            f"{script} exited cleanly but its newest checkpoint is {written.name}, not the "
            f"{plan.final.name} this plan expects; the recipe's epoch arithmetic has moved"
        )
        raise RuntimeError(message)
    if stage == STAGE1:
        shutil.copyfile(written, plan.logs / "first_stage.pth")

    data_volume.commit()
    checkpoint_volume.commit()
    payload = {
        "task": "12.6",
        "stage": stage,
        "arm": arm,
        "voice": voice,
        "trained_this_run": True,
        "seconds_this_run": round(elapsed, 1),
        "epochs": epochs,
        "batch": batch,
        "checkpoint": str(written),
    }
    _emit("done", **payload)
    return payload
