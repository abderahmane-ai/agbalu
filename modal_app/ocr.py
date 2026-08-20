"""Train, smoke-test, and score Feraoun-36M document OCR on Modal GPUs.

`feraoun_train` fits the Vision-Encoder-Decoder on rendered document lines.
`feraoun_smoke` runs a 20-step smoke measuring throughput and loss descent before any
multi-epoch run.
`upload_ocr` pushes the Kabyle text corpus, the Tifinagh split and the Adlis book scans
onto the data volume.
"""

from __future__ import annotations

import logging
import random
import urllib.request
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import DataLoader

from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    OCR_CPU,
    OCR_GPU,
    OCR_TIMEOUT,
    RETRIES,
    VOLUMES,
    app,
    checkpoint_volume,
    data_volume,
    ocr_image,
)

log: Final = logging.getLogger("agbalu.modal.ocr")
DEFAULT_RUN: Final = "feraoun-36m-v1"
RUN_ROOT: Final = Path(CHECKPOINT_PATH) / "feraoun"

LOCAL_TEXT_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
LOCAL_TIFINAGH_PARQUET: Final = Path("data/tasks/tifinagh/script_conversion/train.parquet")
LOCAL_ADLIS_RAW: Final = Path("data/raw/hf.boffire.adlis-pdfs-ocr-kab")

REMOTE_TEXT_CORPUS: Final = Path(DATA_PATH) / "processed" / "text" / "agbalu-text-v1.jsonl"
REMOTE_TIFINAGH_PARQUET: Final = (
    Path(DATA_PATH) / "tasks" / "tifinagh" / "script_conversion" / "train.parquet"
)

HUB_TIFINAGH_PARQUET_URL: Final = (
    "https://huggingface.co/datasets/agbalu/KabTifinagh/resolve/main/"
    "script_conversion/train.parquet"
)
"""The Tifinagh side has a public mirror and can be re-fetched. **The Latin corpus has
none** — AƔBALU-Text v1 is not published — so it is uploaded, never fetched. Any other
dataset written under its filename would train the release on text whose provenance record
says something else."""

HOLDOUT_FRACTION: Final = 0.05
SPLIT_SEED: Final = 1789
"""A seeded draw over the whole line set. `data/raw` lands in source order, so a suffix is
one source's prose and a validation curve over it measures that source, not the corpus."""

PREVIEWS: Final[tuple[str, ...]] = (
    "Taqbaylit d tutlayt tayemmat nneɣ.",
    "Ḥemleɣ ad ɣreɣ idlisen n Dda Lmulud Mammeri d Belaïd At Ali.",
    "« Awal yettazzal, tira teqqim deg udlis. »",
    "1980 d aseggas n Tafsut Imaziɣen deg Tizi Uzzu d Vgayet.",
    "ⴰⵣⵓⵍ ⴼⵍⵍ-ⴰⵡⴻⵏ, ⴰⵎⴻⴽ ⵜⴻⵍⵍⴰⵎ?",
    "« ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ. »",
)


class PreconditionError(RuntimeError):
    """Something knowable before a step runs, so identical on all five retries.

    `retries` cannot tell a bug from a preemption, so these are returned as refusals: a
    missing corpus costs one container, not five with an A10G attached.
    """


def _configure_logging() -> None:
    import warnings

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
    log.info("%-10s %s", event, parts)


def _refuse(reason: str, **fields: object) -> dict[str, Any]:
    payload: dict[str, Any] = {"trained_this_run": False, "refused": reason, **fields}
    log.error("refused: %s", reason)
    return payload


def _fetch_tifinagh_parquet() -> None:
    if REMOTE_TIFINAGH_PARQUET.is_file():
        return
    REMOTE_TIFINAGH_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    _emit("FETCH_START", url=HUB_TIFINAGH_PARQUET_URL)
    staging = REMOTE_TIFINAGH_PARQUET.with_suffix(".partial")
    urllib.request.urlretrieve(HUB_TIFINAGH_PARQUET_URL, staging)
    staging.replace(REMOTE_TIFINAGH_PARQUET)
    data_volume.commit()
    _emit("FETCH_DONE", size_bytes=REMOTE_TIFINAGH_PARQUET.stat().st_size)


def _load_corpus(max_lines: int, tifinagh_ratio: float) -> list[str]:
    """Lines to render, from the volume.

    Raises on anything missing. Training on a handful of repeated sentences produces a
    beautiful loss curve, a plausible CER and a model that has read seven sentences.
    """
    from agbalu.ocr.dataset import (
        build_dual_script_lines,
        chunk_text_into_lines,
        load_corpus_sentences,
    )

    if not REMOTE_TEXT_CORPUS.is_file():
        message = f"no text corpus at {REMOTE_TEXT_CORPUS}; run `make modal-upload TASK=ocr` first"
        raise PreconditionError(message)

    latin = load_corpus_sentences(REMOTE_TEXT_CORPUS, max_sentences=max_lines)
    if not latin:
        message = f"{REMOTE_TEXT_CORPUS} yielded no sentences"
        raise PreconditionError(message)

    if tifinagh_ratio <= 0.0:
        lines = chunk_text_into_lines(latin)[:max_lines]
        _emit("CORPUS", source=str(REMOTE_TEXT_CORPUS), lines=len(lines), tifinagh_pct=0)
        return lines

    _fetch_tifinagh_parquet()
    tifinagh = load_corpus_sentences(REMOTE_TIFINAGH_PARQUET, max_sentences=max_lines)
    if not tifinagh:
        message = f"{REMOTE_TIFINAGH_PARQUET} yielded no Tifinagh sentences"
        raise PreconditionError(message)

    lines = build_dual_script_lines(
        latin,
        tifinagh,
        tifinagh_ratio=tifinagh_ratio,
        max_lines=max_lines,
    )
    _emit(
        "DUAL_CORPUS",
        lines=len(lines),
        tifinagh_pct=int(tifinagh_ratio * 100.0),
        latin_pct=100 - int(tifinagh_ratio * 100.0),
    )
    return lines


def _split(lines: list[str]) -> tuple[list[str], list[str]]:
    held = max(1, int(len(lines) * HOLDOUT_FRACTION))
    indices = random.Random(SPLIT_SEED).sample(range(len(lines)), held)
    holdout = set(indices)
    val = [lines[index] for index in indices]
    train = [line for index, line in enumerate(lines) if index not in holdout]
    return train, val


def _resolve_resume(resume_from: str | None, run_name: str) -> Path | None:
    if not resume_from:
        return None
    if resume_from in ("best", "latest", "final"):
        for root in (RUN_ROOT / run_name, RUN_ROOT / DEFAULT_RUN):
            candidate = root / f"{resume_from}.pt"
            if candidate.is_file():
                return candidate
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


@app.function(
    image=ocr_image,
    gpu=OCR_GPU,
    cpu=OCR_CPU,
    volumes=VOLUMES,
    timeout=OCR_TIMEOUT,
    retries=RETRIES,
)
def feraoun_train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_lines: int = 100_000,
    run_name: str = DEFAULT_RUN,
    use_pretrained: bool = True,
    resume_from: str | None = None,
    tifinagh_ratio: float = 0.0,
    smoke_steps: int | None = None,
) -> dict[str, Any]:
    """Train Feraoun OCR on a Modal GPU."""
    from agbalu.ocr.config import ModelConfig, TrainConfig
    from agbalu.ocr.dataset import SyntheticDataset, collate_lines
    from agbalu.ocr.infer import Recognizer
    from agbalu.ocr.models import VisionEncoderDecoder
    from agbalu.ocr.synthetic import render_text_line
    from agbalu.ocr.trainer import ResumeError, Trainer

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

    try:
        lines = _load_corpus(max_lines=max_lines, tifinagh_ratio=tifinagh_ratio)
        resolved_resume = _resolve_resume(resume_from, run_name)
    except PreconditionError as refusal:
        return _refuse(str(refusal), run=run_name)

    train_lines, val_lines = _split(lines)
    _emit("SPLIT", train=len(train_lines), val=len(val_lines))

    loaders = {
        name: DataLoader(
            SyntheticDataset(rows, augment=name == "train"),
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collate_lines,
            num_workers=4,
            pin_memory=True,
        )
        for name, rows in (("train", train_lines), ("val", val_lines))
    }

    config = ModelConfig()
    model = VisionEncoderDecoder(config=config, use_pretrained_encoder=use_pretrained)
    _emit(
        "MODEL",
        parameters=sum(p.numel() for p in model.parameters()),
        hidden=config.hidden_size,
        vocab=config.vocab_size,
    )

    output_dir = RUN_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        output_dir=output_dir,
        resume_checkpoint=resolved_resume,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=1 if smoke_steps else epochs,
        max_steps=smoke_steps,
        log_every_steps=5 if smoke_steps else 20,
        eval_every_steps=10 if smoke_steps else 250,
        save_every_steps=10 if smoke_steps else 500,
        tifinagh_ratio=tifinagh_ratio,
        device=device.type,
    )

    # Every checkpoint is committed as it lands: the container's filesystem is discarded on
    # preemption, and only a commit survives it.
    trainer = Trainer(
        model,
        loaders["train"],
        loaders["val"],
        train_config,
        on_checkpoint=lambda _: checkpoint_volume.commit(),
    )
    try:
        results = trainer.fit()
    except ResumeError as refusal:
        return _refuse(str(refusal), run=run_name)

    _emit("PREVIEW_START")
    previews = [render_text_line(prompt, augment=False) for prompt in PREVIEWS]
    for gold, pred in zip(
        PREVIEWS,
        Recognizer(model=model, device=device).recognize_lines(previews),
        strict=True,
    ):
        log.info("  GOLD: %s", gold)
        log.info("  PRED: %s", pred)

    checkpoint_volume.commit()
    _emit("DONE", best_cer=results["best_cer"], steps_this_run=results["steps_this_run"])
    return results


@app.function(
    image=ocr_image,
    gpu=OCR_GPU,
    cpu=OCR_CPU,
    volumes=VOLUMES,
    timeout=30 * 60,
)
def feraoun_smoke(run_name: str = "feraoun-smoke") -> dict[str, Any]:
    """20 steps on a real GPU, so the path is proved before a multi-epoch run is paid for."""
    return feraoun_train.local(
        epochs=1,
        batch_size=16,
        max_lines=500,
        run_name=run_name,
        use_pretrained=True,
        smoke_steps=20,
    )


@app.local_entrypoint()
def upload_ocr() -> None:
    """Upload the Kabyle text corpus, the Tifinagh split and the Adlis book scans."""
    if not LOCAL_TEXT_CORPUS.is_file():
        message = f"{LOCAL_TEXT_CORPUS} missing; run `make extract` first"
        raise SystemExit(message)

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(LOCAL_TEXT_CORPUS, "/processed/text/agbalu-text-v1.jsonl")
        if LOCAL_TIFINAGH_PARQUET.is_file():
            batch.put_file(
                LOCAL_TIFINAGH_PARQUET,
                "/tasks/tifinagh/script_conversion/train.parquet",
            )
        scans = 0
        if LOCAL_ADLIS_RAW.is_dir():
            for path in LOCAL_ADLIS_RAW.rglob("*"):
                if path.is_file() and not path.name.startswith("."):
                    relative = path.relative_to(LOCAL_ADLIS_RAW)
                    batch.put_file(path, f"/raw/{LOCAL_ADLIS_RAW.name}/{relative}")
                    scans += 1

    print(f"uploaded {LOCAL_TEXT_CORPUS} ({LOCAL_TEXT_CORPUS.stat().st_size / 1e6:.1f} MB)")
    if LOCAL_TIFINAGH_PARQUET.is_file():
        print(f"uploaded {LOCAL_TIFINAGH_PARQUET}")
    print(f"uploaded {scans} Adlis page scans")
