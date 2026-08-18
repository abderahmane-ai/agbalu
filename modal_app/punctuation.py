"""Fit and score the punctuation and casing heads on a GPU.

`punctuation_train` fine-tunes the pretrained encoder for both heads; `punctuation_evaluate`
scores a held-out split beside the rule baseline in the same pass, because a baseline
computed elsewhere on a different sample is not a comparison.

The splits are already decontaminated when they arrive: `agbalu.punctuation.corpus` excludes
every Common Voice clip whose text is in AƔBALU-Text v1, which is 58.2% of them, and the
encoder was pretrained on that same file — so the rows scored here are unseen by both the
head and the backbone. Nothing in this module may widen that set.

One image and one module: `modal deploy -m modal_app.punctuation` registers whatever this
file defines, and the training image already carries torch, numpy and sentencepiece.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    RETRIES,
    VOLUMES,
    app,
    call_owner,
    checkpoint_volume,
    data_volume,
    image,
)

log: Final = logging.getLogger("agbalu.punctuation")

GPU: Final = "A10G"
"""One A10G. The model is 31M parameters over sentences averaging 10.5 subwords; the run is
bounded by how many rows it reads, not by how much memory a step needs."""

DEFAULT_RUN: Final = "punctuation-v1"
RUN_ROOT: Final = Path(CHECKPOINT_PATH) / "punctuation"
SPLIT_ROOT: Final = Path(DATA_PATH) / "punctuation"
ENCODER_RUN: Final = Path(CHECKPOINT_PATH) / "agbalu-encoder-v1"
TOKENIZER: Final = Path(DATA_PATH) / "agbalu-tok-base-16k.model"

LOCAL_SPLITS: Final = Path("data/processed/punctuation")
LOCAL_TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
LOCAL_ENCODER: Final = Path("artifacts/runs/agbalu-encoder-v1")

PUNCTUATION_TIMEOUT: Final = 6 * 60 * 60
"""Well under Modal's ceiling. 1.3M sentences for a few epochs on a 31M encoder is a matter
of hours; the headroom is for a resumed retry, not for the first attempt."""

RESULTS: Final = Path(DATA_PATH) / "bench" / "punctuation.json"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _emit(event: str, **fields: object) -> None:
    log.info("%s %s", event, " ".join(f"{key}={value}" for key, value in fields.items()))


def _require(path: Path, remedy: str) -> None:
    if not path.exists():
        msg = f"{path} is not on the volume — run `{remedy}` first"
        raise FileNotFoundError(msg)


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    timeout=PUNCTUATION_TIMEOUT,
    retries=RETRIES,
)
def punctuation_train(
    run: str = DEFAULT_RUN,
    epochs: int = 3,
    batch_size: int = 256,
    encoder_lr: float = 5e-5,
    head_lr: float = 1e-3,
    max_length: int = 128,
    freeze_encoder: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Fine-tune both heads, resuming from `latest.pt` if the run directory holds one.

    Locked on the run directory: two trainers writing one directory corrupt nothing, because
    the checkpoint writes are atomic, and simply spend two GPU budgets reaching one step.
    """
    import torch

    from agbalu.model import lock
    from agbalu.model.trainer import select_device
    from agbalu.punctuation.corpus import read_split
    from agbalu.punctuation.dataset import encode_corpus
    from agbalu.punctuation.model import build
    from agbalu.punctuation.train import Trainer, TrainSettings
    from agbalu.tokenizer.evaluate import load

    _configure_logging()
    _require(SPLIT_ROOT / "train.jsonl", "make modal-upload TASK=punctuation")
    _require(TOKENIZER, "make modal-upload TASK=punctuation")
    _require(ENCODER_RUN / "best.pt", "make modal-upload TASK=encoder")

    directory = RUN_ROOT / run
    directory.mkdir(parents=True, exist_ok=True)
    owner = call_owner()
    lock.acquire(directory, owner, force=force)
    try:
        device = select_device()
        tokenizer = load(TOKENIZER)
        settings = TrainSettings(
            epochs=epochs,
            batch_size=batch_size,
            encoder_lr=encoder_lr,
            head_lr=head_lr,
            freeze_encoder=freeze_encoder,
        )
        torch.manual_seed(settings.seed)

        train = encode_corpus(read_split(SPLIT_ROOT / "train.jsonl"), tokenizer, max_length)
        dev = encode_corpus(read_split(SPLIT_ROOT / "dev.jsonl"), tokenizer, max_length)
        _emit("encoded", train=len(train), dev=len(dev), device=str(device))

        model = build(ENCODER_RUN, device=device)
        if settings.freeze_encoder:
            model.freeze_encoder()

        trainer = Trainer(model, train, dev, settings, directory, device)
        if trainer.maybe_resume():
            _emit("resumed", step=trainer.state.step, epoch=trainer.state.epoch)
        summary = trainer.run()
        checkpoint_volume.commit()

        best = summary.best
        return {
            "run": run,
            "total_steps": summary.total_steps,
            "steps_this_run": summary.steps_this_run,
            "seconds_this_run": round(summary.seconds_this_run, 1),
            "labels_per_second": summary.labels_per_second,
            "best_loss": best.loss if best else None,
            "best_punctuation_macro_f1": best.punctuation_macro_f1 if best else None,
            "best_case_noninitial_f1": best.case_noninitial_f1 if best else None,
            "history": summary.history,
        }
    finally:
        lock.release(directory, owner)
        checkpoint_volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    timeout=PUNCTUATION_TIMEOUT,
)
def punctuation_evaluate(
    run: str = DEFAULT_RUN,
    split: str = "test",
    name: str = "best.pt",
    max_length: int = 128,
) -> dict[str, object]:
    """Score a checkpoint and the rule baseline on the same rows, in the same pass.

    `split="ood"` is the generalisation probe: long-form prose from a source held out of
    training entirely, where `dev` and `test` are Common Voice and share their shape with
    half the training text.
    """
    import torch

    from agbalu.model.trainer import select_device
    from agbalu.punctuation.corpus import read_split
    from agbalu.punctuation.evaluate import Prediction, render, score, trivial_baseline
    from agbalu.punctuation.infer import predict, truncated_gold
    from agbalu.punctuation.labels import annotate
    from agbalu.punctuation.model import build
    from agbalu.tokenizer.evaluate import load

    _configure_logging()
    checkpoint = RUN_ROOT / run / name
    _require(checkpoint, "make modal-punctuation TASK=train")
    _require(SPLIT_ROOT / f"{split}.jsonl", "make modal-upload TASK=punctuation")

    device = select_device()
    tokenizer = load(TOKENIZER)
    rows = read_split(SPLIT_ROOT / f"{split}.jsonl")

    model = build(ENCODER_RUN, device=device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])

    restorations = predict(
        model, tokenizer, [row.text for row in rows], device=device, max_length=max_length
    )
    predictions = [
        Prediction(
            truncated_gold(row.text, len(restoration.words)),
            restoration.punctuation,
            restoration.case,
        )
        for row, restoration in zip(rows, restorations, strict=True)
    ]
    model_report = score(predictions)
    baseline_report = score([trivial_baseline(annotate(row.text)) for row in rows])

    print(render(baseline_report, f"BASELINE  capitalise-first + final period  [{split}]"))
    print()
    print(render(model_report, f"MODEL     {run}/{name}  [{split}]"))

    payload: dict[str, object] = {
        "run": run,
        "split": split,
        "checkpoint": name,
        "model": model_report,
        "baseline": baseline_report,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    _emit(
        "scored",
        split=split,
        sentences=model_report["sentences"],
        macro_f1=round(model_report["punctuation_macro_f1"], 4),
        baseline_macro_f1=round(baseline_report["punctuation_macro_f1"], 4),
    )
    return payload


@app.local_entrypoint()
def upload_punctuation() -> None:
    """Put every decontaminated split and the tokenizer on the data volume.

    The tokenizer ships here rather than being taken from `modal_app.train.upload_corpus`.
    That entrypoint refuses to run until `artifacts/model-data/train.bin` exists — the
    encoder's tokenised pretraining corpus, gigabytes of it — so depending on it makes a
    400 kB file conditional on rebuilding an artifact this run never reads. It is immutable
    and checksummed in the released config, so two writers of the path cannot disagree.
    """
    from agbalu.punctuation.corpus import WRITTEN_SPLITS

    if not LOCAL_SPLITS.is_dir():
        msg = f"{LOCAL_SPLITS} does not exist — run `make punctuation TASK=corpus` first"
        raise FileNotFoundError(msg)
    if not LOCAL_TOKENIZER.is_file():
        msg = f"{LOCAL_TOKENIZER} is missing — run `make tokenizer STAGE=build` first"
        raise FileNotFoundError(msg)

    with data_volume.batch_upload(force=True) as batch:
        for split in WRITTEN_SPLITS:
            source = LOCAL_SPLITS / f"{split}.jsonl"
            if not source.is_file():
                msg = f"{source} is missing — run `make punctuation TASK=corpus` first"
                raise FileNotFoundError(msg)
            batch.put_file(source, f"/punctuation/{split}.jsonl")
            print(f"{source} -> /punctuation/{split}.jsonl ({source.stat().st_size / 1e6:.1f} MB)")
        batch.put_file(LOCAL_TOKENIZER, f"/{LOCAL_TOKENIZER.name}")
        print(f"{LOCAL_TOKENIZER} -> /{LOCAL_TOKENIZER.name}")


@app.local_entrypoint()
def upload_encoder() -> None:
    """Put the pretrained encoder on the checkpoint volume, for an account that lacks it.

    Separate from `upload_punctuation` and out of the default `modal-upload` sweep, because
    it is 398 MB that only has to move once. The `.sha256` sidecar goes with it: without the
    sidecar `checkpoint.load` cannot verify what it restores, and a truncated 398 MB download
    has already cost this project a staging cycle once.
    """
    if not (LOCAL_ENCODER / "best.pt").is_file():
        msg = f"{LOCAL_ENCODER}/best.pt is missing — this is the Masinissa training run"
        raise FileNotFoundError(msg)

    with checkpoint_volume.batch_upload(force=True) as batch:
        for name in ("best.pt", "best.pt.sha256"):
            source = LOCAL_ENCODER / name
            if not source.is_file():
                print(f"skipping {source}: not present")
                continue
            batch.put_file(source, f"/{LOCAL_ENCODER.name}/{name}")
            print(
                f"{source} -> /{LOCAL_ENCODER.name}/{name} ({source.stat().st_size / 1e6:.1f} MB)"
            )
