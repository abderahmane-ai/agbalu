"""Training Dihya-5M, the Berber language identifier, on a Modal GPU.

Short enough for `modal run`: 5.19M parameters over 77,756 rows is minutes, not hours,
so the spawn-against-a-deployed-app rule that governs the long trainers does not apply.

The function returns the numbers that decide whether the artifact is usable — worst
group recall beside macro-F1, and the calibration's realised coverage — rather than
leaving them on the volume for someone to read afterwards.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import torch

from agbalu.lid.calibrate import calibrate, save
from agbalu.lid.config import ModelConfig, TrainConfig
from agbalu.lid.dataset import Corpus, Record
from agbalu.lid.infer import exported_weights
from agbalu.lid.model import CharClassifier
from agbalu.lid.train import build_corpora, held_out_report, train_identifier
from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    DATA_VOLUME_NAME,
    VOLUMES,
    app,
    checkpoint_volume,
    data_volume,
    image,
)

log: Final = logging.getLogger("agbalu.dihya")

GPU: Final = "A10G"
CPU: Final = 4.0
"""Requested explicitly. Modal's default is 0.125 of a core, and on this model that
shows up as a slow step rather than as a CPU error: the batch is built row by row on the
host, so the collate is the step's floor."""

TIMEOUT: Final = 2 * 60 * 60

LOCAL_DATA_DIR: Final = Path("data/processed/bench/lid-dataset")
LOCAL_EVAL_PATH: Final = Path("data/processed/bench/lid-eval.jsonl")

REMOTE_DATA_DIR: Final = Path(DATA_PATH) / "processed" / "bench" / "lid-dataset"
REMOTE_EVAL_PATH: Final = Path(DATA_PATH) / "processed" / "bench" / "lid-eval.jsonl"
REMOTE_CHECKPOINT_DIR: Final = Path(CHECKPOINT_PATH) / "dihya"

SPLITS: Final[tuple[str, ...]] = ("train.jsonl", "dev.jsonl")
REMOTE_SPLIT_ROOT: Final = "/processed/bench/lid-dataset"


def load_exported(directory: Path, config: ModelConfig, device: torch.device) -> CharClassifier:
    """`best.pt` back into a bare classifier, without the training-only submodules."""
    payload = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    model = CharClassifier(config)
    model.load_state_dict(exported_weights(payload["model"]))
    return model.to(device)


def read_held_out(path: Path) -> list[Record]:
    """The held-out set as corpus rows, keeping the domain the report groups on."""
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        language = row["language"]
        domain = row.get("domain", "unknown")
        records.append(
            Record(
                text=row["text"],
                language=language,
                domain=domain,
                group=f"{language}/{domain}",
                source=row.get("source", "unknown"),
            )
        )
    if not records:
        msg = f"{path} carried no rows"
        raise ValueError(msg)
    return records


@app.function(image=image, gpu=GPU, cpu=CPU, volumes=VOLUMES, timeout=TIMEOUT)
def dihya_train(
    epochs: int = 8,
    learning_rate: float = 3e-4,
    contrastive_weight: float = 0.5,
    group_dro_step: float = 0.01,
    coverage: float = 0.99,
    seed: int = 20260823,
) -> dict[str, float]:
    """Train, calibrate on dev, then score the held-out set once."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    device = torch.device("cuda")
    log.info("device=%s", torch.cuda.get_device_name(0))

    model_config = ModelConfig()
    settings = TrainConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        contrastive_weight=contrastive_weight,
        group_dro_step=group_dro_step,
        seed=seed,
    )
    summary = train_identifier(
        model_config=model_config,
        settings=settings,
        data_dir=REMOTE_DATA_DIR,
        checkpoint_dir=REMOTE_CHECKPOINT_DIR,
        device=device,
    )
    if summary.best is None:
        msg = "training completed no validation, so no checkpoint was selected"
        raise RuntimeError(msg)

    train_corpus, dev_corpus = build_corpora(REMOTE_DATA_DIR, model_config, settings)
    model = load_exported(REMOTE_CHECKPOINT_DIR, model_config, device)

    calibration = calibrate(
        model, dev_corpus, settings=settings, target_coverage=coverage, device=str(device)
    )
    save(calibration, REMOTE_CHECKPOINT_DIR / "calibration.json")

    report = held_out_report(
        model,
        read_held_out(REMOTE_EVAL_PATH),
        Corpus.observed_groups(train_corpus.records),
        batch_size=settings.eval_batch_size,
        device=device,
    )
    (REMOTE_CHECKPOINT_DIR / "held-out-report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    checkpoint_volume.commit()

    worst_group, worst_recall = report.worst_group
    log.info(
        "held-out accuracy=%.4f macro_f1=%.4f worst=%s worst_recall=%.4f "
        "coverage=%.4f mean_set=%.2f",
        report.accuracy,
        report.macro_f1,
        worst_group,
        worst_recall,
        calibration.empirical_coverage,
        calibration.mean_set_size,
    )
    headline = {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "worst_group_recall": worst_recall,
        "dev_score": summary.best.score,
        "q_hat": calibration.q_hat,
        "empirical_coverage": calibration.empirical_coverage,
        "mean_set_size": calibration.mean_set_size,
        "steps": float(summary.steps_this_run),
    }
    per_class = {f"f1_{score.label}": score.f1 for score in report.classes}
    return {**headline, **per_class}


@app.local_entrypoint()
def upload(force: bool = False) -> None:
    """Put the built splits and the held-out set on the data volume."""
    missing = [name for name in SPLITS if not (LOCAL_DATA_DIR / name).is_file()]
    if missing or not LOCAL_EVAL_PATH.is_file():
        msg = f"missing {missing or [str(LOCAL_EVAL_PATH)]} — run `make lid-dataset` first"
        raise FileNotFoundError(msg)

    print(f"uploading LID splits to {DATA_VOLUME_NAME}")
    with data_volume.batch_upload(force=force) as batch:
        for name in SPLITS:
            local = LOCAL_DATA_DIR / name
            batch.put_file(local, f"{REMOTE_SPLIT_ROOT}/{name}")
            print(f"  {name} {local.stat().st_size / 1e6:.1f} MB")
        batch.put_file(LOCAL_EVAL_PATH, "/processed/bench/lid-eval.jsonl")
        print(f"  {LOCAL_EVAL_PATH.name} {LOCAL_EVAL_PATH.stat().st_size / 1e6:.1f} MB")


@app.local_entrypoint()
def train(
    epochs: int = 8,
    learning_rate: float = 3e-4,
    contrastive_weight: float = 0.5,
    group_dro_step: float = 0.01,
    coverage: float = 0.99,
    seed: int = 20260823,
    sync: bool = True,
) -> None:
    """Upload the splits, train, and print the held-out headline."""
    if sync:
        upload(force=True)
    result = dihya_train.remote(
        epochs=epochs,
        learning_rate=learning_rate,
        contrastive_weight=contrastive_weight,
        group_dro_step=group_dro_step,
        coverage=coverage,
        seed=seed,
    )
    for key, value in result.items():
        print(f"  {key:26s} {value:.4f}")
