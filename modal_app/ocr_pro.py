"""Train and smoke-test Feraoun-Pro document OCR on Modal GPUs.

`ocr_pro_train` fits the hybrid CTC recogniser on rendered document lines, in the Latin
orthography and Neo-Tifinagh together. `ocr_pro_smoke` runs a short one on the same path
before any multi-epoch run is paid for.

The corpus is uploaded by `modal_app.ocr::upload_ocr`, which already puts both the Latin
JSONL and the Tifinagh split on the data volume — `make modal-upload TASK=ocr`.
"""

from __future__ import annotations

import logging
import random
import urllib.request
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import modal
import torch

if TYPE_CHECKING:
    # Only the helpers below are annotated with these, never a Modal entrypoint: the CLI
    # resolves an entrypoint's hints at call time and a `TYPE_CHECKING` name would break
    # `modal run` while leaving `spawn` working.
    from agbalu.ocr.model_pro import HybridCTCRecognizer
    from agbalu.ocr.synthetic_pro import LineRenderer

from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    OCR_CPU,
    OCR_GPU,
    OCR_TIMEOUT,
    RETRIES,
    VOLUMES,
    checkpoint_volume,
    data_volume,
    ocr_pro_image,
)

APP_NAME: Final = "agbalu-ocr"
"""Not the shared app: a subset deploy replaces the app's function set."""

app: Final = modal.App(APP_NAME)
log: Final = logging.getLogger("agbalu.modal.ocr_pro")

DEFAULT_RUN: Final = "feraoun-pro"
RUN_ROOT: Final = Path(CHECKPOINT_PATH)
"""Where the first run wrote, so `/<run>/best.pt` on the volume. Kept rather than moved
under a `feraoun-pro/` root: the four trained epochs are at the old path and a tidier
layout would orphan them."""

REMOTE_TEXT_CORPUS: Final = Path(DATA_PATH) / "processed" / "text" / "agbalu-text-v1.jsonl"
REMOTE_TIFINAGH_PARQUET: Final = (
    Path(DATA_PATH) / "tasks" / "tifinagh" / "script_conversion" / "train.parquet"
)

HUB_TIFINAGH_PARQUET_URL: Final = (
    "https://huggingface.co/datasets/agbalu/KabTifinagh/resolve/main/"
    "script_conversion/train.parquet"
)
"""The Tifinagh side is published and can be re-fetched. **The Latin corpus is not** —
AƔBALU-Text v1 has no mirror — so it is uploaded, never fetched. Any other dataset landing
under its filename would train the release on text whose provenance record says otherwise."""

HOLDOUT_FRACTION: Final = 0.05
SPLIT_SEED: Final = 1789
"""A seeded draw over the mixed line set. `data/raw` lands in source order, so a prefix is
one source's prose: the first 80,000 lines of this corpus are a single CC-BY-2.0 dataset,
not the 42-source mixture, and a validation curve over a suffix measures that source."""

MAX_LINES: Final = 600_000
TIFINAGH_RATIO: Final = 0.40
"""Both scripts in one mix. Latin is not held out of the continuation: rehearsal is what
stops a Latin-trained encoder forgetting Latin, and a gate that only *detects* regression
cannot prevent it."""

LATIN_PREVIEWS: Final[tuple[str, ...]] = (
    "Taqbaylit d tutlayt tayemmat nneɣ.",
    "Ḥemleɣ ad ɣreɣ idlisen n Dda Lmulud Mammeri d Belaïd At Ali.",
    "Tettmeslay ţ-ţidet ɣef tmaziɣt d yidles n lejdud.",
    "« Awal yettazzal, tira teqqim deg udlis. »",
    "1980 d aseggas n Tafsut Imaziɣen deg Tizi Uzzu d Vgayet.",
)

TIFINAGH_PREVIEWS: Final[tuple[str, ...]] = (
    "ⴰⵣⵓⵍ ⴼⵍⵍ-ⴰⵡⴻⵏ, ⴰⵎⴻⴽ ⵜⴻⵍⵍⴰⵎ?",
    "« ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ. »",
    "ⵎⴰⵛⵉ ⴷ ⵢⵉⵡⴻⵏ ⵡⴰⵡⴰⵍ ⵉ ⴷ-ⵢⴻⵏⵏⴰ ⵓⵎⴻⵖⵏⴰⵙ.",
    "ⵃⴻⵎⵍⴻⵖ ⴰⴷ ⵖⵔⴻⵖ ⵉⴷⵍⵉⵙⴻⵏ ⵏ ⴷⴷⴰ ⵍⵎⵓⵍⵓⴷ ⵎⴰⵎⵎⴻⵔⵉ.",
)


class PreconditionError(RuntimeError):
    """Something knowable before a step runs, so identical on all five retries.

    `retries` cannot tell a bug from a preemption, so these are returned as refusals: a
    missing corpus costs one container, not five with an A10G attached.
    """


def _configure_logging() -> None:
    warnings.filterwarnings("ignore")
    for noisy in ("urllib3", "httpx", "modal", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        force=True,
    )


def _emit(event: str, **fields: float | int | str | None) -> None:
    parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    log.info("%-12s %s", event, parts)


def _refuse(reason: str, **fields: object) -> dict[str, Any]:
    log.error("REFUSED      %s", reason)
    return {"trained_this_run": False, "refused": reason, **fields}


def _fetch_tifinagh_parquet() -> None:
    if REMOTE_TIFINAGH_PARQUET.is_file():
        return
    REMOTE_TIFINAGH_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    _emit("FETCH_START", url=HUB_TIFINAGH_PARQUET_URL)
    staging = REMOTE_TIFINAGH_PARQUET.with_suffix(".partial")
    urllib.request.urlretrieve(HUB_TIFINAGH_PARQUET_URL, staging)
    staging.replace(REMOTE_TIFINAGH_PARQUET)
    data_volume.commit()
    _emit("FETCH_DONE", bytes=REMOTE_TIFINAGH_PARQUET.stat().st_size)


def _load_mixed_corpus(max_lines: int, tifinagh_ratio: float) -> list[str]:
    """The mixed line set, from the volume.

    Raises on anything missing rather than falling back to a sample pool. A run over a
    handful of repeated sentences converges, reports a plausible CER, and writes a
    checkpoint that has read seven sentences.
    """
    from agbalu.ocr.dataset_pro import load_corpus_lines, mix_dual_script

    if not REMOTE_TEXT_CORPUS.is_file():
        message = f"no text corpus at {REMOTE_TEXT_CORPUS}; run `make modal-upload TASK=ocr` first"
        raise PreconditionError(message)

    latin = load_corpus_lines(REMOTE_TEXT_CORPUS)
    if not latin:
        message = f"{REMOTE_TEXT_CORPUS} yielded no sentences"
        raise PreconditionError(message)

    if tifinagh_ratio <= 0.0:
        lines = mix_dual_script(latin, [], tifinagh_ratio=0.0, max_lines=max_lines)
        _emit("CORPUS", lines=len(lines), latin=len(lines), tifinagh=0)
        return lines

    _fetch_tifinagh_parquet()
    tifinagh = load_corpus_lines(REMOTE_TIFINAGH_PARQUET)
    if not tifinagh:
        message = f"{REMOTE_TIFINAGH_PARQUET} yielded no Tifinagh sentences"
        raise PreconditionError(message)

    lines = mix_dual_script(latin, tifinagh, tifinagh_ratio=tifinagh_ratio, max_lines=max_lines)
    from agbalu.ocr.synthetic_pro import is_tifinagh_text

    tifinagh_lines = sum(1 for line in lines if is_tifinagh_text(line))
    _emit(
        "CORPUS",
        lines=len(lines),
        latin=len(lines) - tifinagh_lines,
        tifinagh=tifinagh_lines,
        tifinagh_pool=len(tifinagh),
        latin_pool=len(latin),
    )
    if tifinagh_lines == 0:
        message = f"mix at ratio {tifinagh_ratio} produced no Tifinagh lines"
        raise PreconditionError(message)
    return lines


def _split(lines: list[str]) -> tuple[list[str], list[str]]:
    """A seeded draw, so both sides carry the mix's script proportions."""
    held = max(1, int(len(lines) * HOLDOUT_FRACTION))
    indices = random.Random(SPLIT_SEED).sample(range(len(lines)), held)
    holdout = set(indices)
    validation = [lines[index] for index in indices]
    train = [line for index, line in enumerate(lines) if index not in holdout]
    return train, validation


def _resolve_resume(resume_from: str | None, run_name: str) -> Path | None:
    """The checkpoint to continue, or a refusal.

    A name that is not `best`/`latest` is read as another run's directory, which is how a
    continuation warm-starts from a different lineage: `RUN=feraoun-pro-tifinagh
    RESUME=feraoun-pro` writes somewhere new and leaves the source checkpoint intact.
    """
    if not resume_from:
        return None
    if resume_from in ("best", "latest"):
        own = RUN_ROOT / run_name / f"{resume_from}.pt"
        if own.is_file():
            return own
        fallback = RUN_ROOT / DEFAULT_RUN / f"{resume_from}.pt"
        if fallback.is_file():
            # Logged rather than silent: this is a *different lineage's* checkpoint, and a
            # run that continues one without saying so reports another run's training as
            # its own.
            _emit("RESUME_FALLBACK", requested=str(RUN_ROOT / run_name), using=str(fallback))
            return fallback
        message = f"no {resume_from}.pt under {RUN_ROOT / run_name} or {RUN_ROOT / DEFAULT_RUN}"
        raise PreconditionError(message)
    named = Path(resume_from)
    if named.is_file():
        return named
    inside_run = RUN_ROOT / resume_from / "best.pt"
    if inside_run.is_file():
        return inside_run
    message = f"resume checkpoint {resume_from} resolves to nothing on the volume"
    raise PreconditionError(message)


def _run_preview_eval(
    model: HybridCTCRecognizer, renderer: LineRenderer, device: torch.device, tag: str
) -> None:
    import editdistance
    import numpy as np

    from agbalu.ocr.vocabulary_pro import decode_ctc

    model.eval()
    for script, prompts in (("LATIN", LATIN_PREVIEWS), ("TIFINAGH", TIFINAGH_PREVIEWS)):
        log.info("PREVIEW      %s %s", tag, script)
        with torch.no_grad():
            for prompt in prompts:
                image, _ = renderer.render_line(prompt, apply_augmentation=False)
                array = (np.array(image, dtype=np.float32) / 255.0 - 0.5) / 0.5
                tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
                predicted = decode_ctc(
                    torch.argmax(model(pixel_values=tensor).ctc_logits, dim=-1)[0].cpu().tolist()
                )
                edits = editdistance.eval(predicted, prompt)
                status = (
                    "exact"
                    if predicted == prompt
                    else f"ed={edits} cer={edits / max(1, len(prompt)):.1%}"
                )
                log.info("  [%-16s] gold: %r", status, prompt)
                log.info("                     pred: %r", predicted)
    model.train()


@app.function(
    image=ocr_pro_image,
    gpu=OCR_GPU,
    cpu=OCR_CPU,
    volumes=VOLUMES,
    timeout=OCR_TIMEOUT,
    retries=RETRIES,
)
def ocr_pro_train(
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-5,
    eval_every: int = 500,
    max_lines: int = MAX_LINES,
    tifinagh_ratio: float = TIFINAGH_RATIO,
    resume_from: str | None = None,
    run_name: str = DEFAULT_RUN,
    log_every: int = 20,
) -> dict[str, Any]:
    """Train Feraoun-Pro on a Modal GPU.

    The defaults are the continuation's, not a fresh run's: one epoch at `lr=2e-5`, ten
    times below the 2e-4 the Latin-only run used. A continuation that re-enters at the
    original learning rate overwrites what it is meant to preserve.
    """
    _configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _emit(
        "INIT",
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        run=run_name,
        epochs=epochs,
        batch=batch_size,
        lr=lr,
        tifinagh_ratio=tifinagh_ratio,
        resume_from=resume_from,
    )

    from agbalu.ocr.model_pro import HybridCTCRecognizer
    from agbalu.ocr.synthetic_pro import LineRenderer
    from agbalu.ocr.trainer_pro import TrainingConfigOCRPro, train_feraoun_pro

    try:
        lines = _load_mixed_corpus(max_lines=max_lines, tifinagh_ratio=tifinagh_ratio)
        resolved_resume = _resolve_resume(resume_from, run_name)
    except PreconditionError as refusal:
        return _refuse(str(refusal), run=run_name)

    train_lines, val_lines = _split(lines)
    _emit("SPLIT", train=len(train_lines), val=len(val_lines))

    save_dir = RUN_ROOT / run_name
    config = TrainingConfigOCRPro(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        log_every=log_every,
        eval_every=eval_every,
        save_dir=save_dir,
    )

    model = HybridCTCRecognizer()
    inference_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("aux_decoder.")
    )
    _emit(
        "MODEL",
        parameters=f"{sum(p.numel() for p in model.parameters()):,}",
        # What a downloader gets: the auxiliary decoder supplies training gradients and is
        # not exported, so the trained figure is not the published one.
        inference_parameters=f"{inference_parameters:,}",
        architecture="HybridCTCRecognizer",
    )

    renderer = LineRenderer()
    results = train_feraoun_pro(
        model, train_lines, val_lines, config, device, resume_from=resolved_resume
    )
    checkpoint_volume.commit()
    _emit("CHECKPOINT_COMMITTED", path=str(save_dir / "best.pt"))

    # The invariants that decide whether the artifact is usable, printed by the run that
    # paid for it: a Latin CER that moved against its own baseline is the one number that
    # says the continuation cost more than it bought.
    _emit(
        "RESULT",
        steps_this_run=results["steps_this_run"],
        total_steps=results["total_steps"],
        latin_cer=results["best_latin_cer"],
        latin_baseline=results["baseline_latin_cer"],
        tifinagh_cer=results["best_tifinagh_cer"],
        tifinagh_baseline=results["baseline_tifinagh_cer"],
        non_finite_steps=results["non_finite_steps"],
    )
    _run_preview_eval(model, renderer, device, tag="final")
    return results


@app.function(
    image=ocr_pro_image,
    gpu=OCR_GPU,
    cpu=OCR_CPU,
    volumes=VOLUMES,
    timeout=30 * 60,
)
def ocr_pro_smoke(run_name: str = "feraoun-pro-smoke") -> dict[str, Any]:
    """A short run on the real path and the real corpus, before a multi-epoch one is paid for.

    Deliberately not a synthetic sentence list: the defect this exists to catch is a corpus
    that never reaches the model, and a hardcoded pool cannot see it.
    """
    return ocr_pro_train.local(
        epochs=1,
        batch_size=16,
        lr=2e-5,
        eval_every=50,
        log_every=5,
        max_lines=2_000,
        run_name=run_name,
    )
