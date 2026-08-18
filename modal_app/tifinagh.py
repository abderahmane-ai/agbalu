"""Train and score the script-conversion model on a GPU.

`train` fits it on `agbalu/KabTifinagh`'s script-conversion split; `evaluate` scores a
held-out split two ways, and reporting both is the point of the module.

**Teacher forcing and free running are different measurements and the gap is the result.**
A teacher-forced pass hands the decoder the gold prefix at every position, so a model that
would derail after one wrong character still scores every later position correctly. A
free-running pass feeds the model its own output, which is what a caller gets. The first
evaluation of this model reported the teacher-forced number under a beam-search heading;
`Evaluation` carries both, named.

One image and one module, because `modal deploy -m modal_app.tifinagh` registers whatever
this file defines and a second file on another image cannot be deployed with it.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final

from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    RETRIES,
    TRAIN_TIMEOUT,
    VOLUMES,
    app,
    call_owner,
    checkpoint_volume,
    data_volume,
    tifinagh_image,
)

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from agbalu.bench.tifinagh import Pair
    from agbalu.tifinagh.model import CharTransformer
    from agbalu.tifinagh.tokenizer import CharTokenizer

log: Final = logging.getLogger("agbalu.tifinagh")

GPU: Final = "A10G"
DEFAULT_RUN: Final = "juba-27m-v1"
RUN_ROOT: Final = Path(CHECKPOINT_PATH) / "tifinagh"
SPLIT_ROOT: Final = Path(DATA_PATH) / "tasks" / "tifinagh" / "script_conversion"

HUB_SPLITS: Final = (
    "https://huggingface.co/datasets/agbalu/KabTifinagh/resolve/main/script_conversion"
)
"""The published dataset. Fetched onto the volume rather than uploaded from the laptop:
it is 60 MB, it is immutable once released, and a container that can rebuild its own
inputs is one the operator does not have to prepare."""

MAX_PIECES: Final = 256
"""Longest sequence in characters. The builder caps a row at 35 words, so this is slack."""

LOG_EVERY: Final = 200
PREVIEW_LIMIT: Final = 256
"""Sentences decoded free-running at each validation. Enough for the exact-match rate to
move visibly between checkpoints, small enough that scoring does not outweigh training."""

PREVIEWS: Final[tuple[str, ...]] = (
    "ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?",
    "ⵢⴼⴽⴰ ⵢⴰⵙ ⴷ ⵜ.",
    "ⴰⴽⴽⴰ ⴷ ⴰⴽⴽⴰ.",
)
"""Logged verbatim at every validation, so a reader can watch the schwa appear rather than
infer it from a loss curve."""


def _configure_logging() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    for noisy in ("urllib3", "httpx", "modal", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", force=True
    )


def _emit(event: str, **fields: float | int | str | None) -> None:
    parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    log.info("%-10s %s", event, parts)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One validation pass. Both exact-match rates, because they answer different questions.

    `teacher_forced_exact` is the share of sentences every position of which is argmax-correct
    given the gold prefix. `free_running_exact` is the share the model reproduces when fed
    its own output. The second is the one a card may quote without qualification.
    """

    loss: float
    token_accuracy: float
    teacher_forced_exact: float
    free_running_exact: float
    free_running_sentences: int


def _ensure_splits(directory: Path, splits: tuple[str, ...]) -> None:
    """Fetch any missing split of the published dataset onto the volume."""
    directory.mkdir(parents=True, exist_ok=True)
    for split in splits:
        target = directory / f"{split}.parquet"
        if target.is_file():
            continue
        url = f"{HUB_SPLITS}/{split}.parquet"
        _emit("fetch", split=split, url=url)
        staging = target.with_suffix(".partial")
        urllib.request.urlretrieve(url, staging)  # noqa: S310
        staging.replace(target)
        data_volume.commit()
        _emit("fetched", split=split, megabytes=round(target.stat().st_size / 1e6, 1))


def _batches(
    pairs: Sequence[Pair],
    tokenizer: CharTokenizer,
    batch_size: int,
    *,
    order: Sequence[int],
    device: torch.device,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Padded source, decoder prefix and labels, in a caller-supplied order.

    Written here rather than through `DataLoader` because a character lookup is not work
    a worker process can usefully be spawned for, and the order being the caller's makes
    the shuffle a seeded decision rather than a hidden one.
    """
    import torch

    for start in range(0, len(order), batch_size):
        window = [pairs[index] for index in order[start : start + batch_size]]
        sources = [tokenizer.encode(pair.tifinagh)[:MAX_PIECES] for pair in window]
        targets = [tokenizer.encode(pair.latin)[:MAX_PIECES] for pair in window]
        yield (
            _pad(sources, torch=torch, device=device),
            _pad([target[:-1] for target in targets], torch=torch, device=device),
            _pad([target[1:] for target in targets], torch=torch, device=device),
        )


def _pad(rows: Sequence[Sequence[int]], *, torch: ModuleType, device: torch.device) -> Tensor:
    """Right-pad to the longest row with `[PAD]`, which is zero and the loss ignores it."""
    width = max(len(row) for row in rows)
    padded = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    moved: Tensor = padded.to(device, non_blocking=True)
    return moved


def _validate(
    model: CharTransformer,
    tokenizer: CharTokenizer,
    pairs: Sequence[Pair],
    device: torch.device,
    batch_size: int,
) -> Evaluation:
    """Loss and both exact-match rates over the dev split.

    The teacher-forced half reuses the logits the loss was computed from. Calling the
    model a second time for the predictions runs it twice over the same batch, which is
    what the first test-split pass did.
    """
    import torch

    from agbalu.tifinagh.infer import Transliterator

    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    tokens = torch.zeros((), dtype=torch.long, device=device)
    exact = torch.zeros((), dtype=torch.long, device=device)

    with torch.no_grad():
        for source, prefix, labels in _batches(
            pairs, tokenizer, batch_size, order=range(len(pairs)), device=device
        ):
            output, logits = model.loss_with_logits(source, prefix, labels)
            scored = labels != model.config.pad_token_id
            loss_sum += output.loss.detach() * output.tokens
            correct += output.correct
            tokens += output.tokens
            exact += ((logits.argmax(dim=-1) == labels) | ~scored).all(dim=-1).sum()

    engine = Transliterator(model, tokenizer, device)
    sample = pairs[:PREVIEW_LIMIT]
    hypotheses = engine.greedy_batch([pair.tifinagh for pair in sample], MAX_PIECES)
    free_running = sum(
        pair.latin.lower() == hypothesis
        for pair, hypothesis in zip(sample, hypotheses, strict=True)
    )
    for source_text, hypothesis in zip(PREVIEWS, engine.greedy_batch(PREVIEWS), strict=True):
        _emit("preview", source=source_text, output=hypothesis)

    model.train()
    total = int(tokens)
    return Evaluation(
        loss=float(loss_sum) / max(1, total),
        token_accuracy=int(correct) / max(1, total),
        teacher_forced_exact=int(exact) / max(1, len(pairs)),
        free_running_exact=free_running / max(1, len(sample)),
        free_running_sentences=len(sample),
    )


def _save(directory: Path, payload: dict[str, object], name: str) -> None:
    """Write through a staging path, so a kill mid-write cannot destroy the last good file."""
    import torch

    target = directory / name
    staging = target.with_suffix(".partial")
    torch.save(payload, staging)
    staging.replace(target)


@app.function(
    image=tifinagh_image,
    gpu=GPU,
    volumes=VOLUMES,
    timeout=TRAIN_TIMEOUT,
    retries=RETRIES,
)
def tifinagh_train(  # noqa: PLR0915
    run: str = DEFAULT_RUN, max_steps: int = 0, force: bool = False
) -> dict[str, object]:
    """Fit the model, resuming from `latest.pt` if the run directory holds one.

    Locked on the run directory: two trainers writing one directory corrupt nothing, since
    the writes are atomic, and simply spend two GPU budgets reaching the same step.
    """
    import torch

    from agbalu.bench.tifinagh import read_split
    from agbalu.model import lock
    from agbalu.tifinagh.config import ModelConfig, TrainConfig
    from agbalu.tifinagh.model import CharTransformer
    from agbalu.tifinagh.tokenizer import CharTokenizer

    _configure_logging()
    directory = RUN_ROOT / run
    directory.mkdir(parents=True, exist_ok=True)
    owner = call_owner()
    lock.acquire(directory, owner, force=force)
    try:
        _ensure_splits(SPLIT_ROOT, ("train", "dev"))
        model_config, train_config = ModelConfig(), TrainConfig(run_name=run)
        total = max_steps or train_config.max_steps

        torch.manual_seed(train_config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = CharTokenizer()
        model = CharTransformer(model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total, eta_min=train_config.min_learning_rate
        )

        step, epoch, best = 0, 0, -1.0
        latest = directory / "latest.pt"
        if latest.is_file():
            state = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step, epoch, best = state["step"], state["epoch"], state.get("best", -1.0)

        train_pairs = read_split("train", directory=SPLIT_ROOT)
        dev_pairs = read_split("dev", directory=SPLIT_ROOT)
        _emit(
            "start",
            run=run,
            device=str(device),
            parameters=model_config.parameters,
            rows=len(train_pairs),
            steps=total,
            resumed_at=step or None,
        )
        if step >= total:
            log.warning(
                "nothing to train: resumed at step %d of %d. Raise --max-steps, or use a "
                "fresh --run — every number below would be the checkpoint's, not this run's",
                step,
                total,
            )
            return {"run": run, "step": step, "trained_this_run": 0}

        entered, started = step, time.monotonic()
        model.train()
        accumulated = 0
        order = list(range(len(train_pairs)))
        shuffler = random.Random(train_config.seed)
        while step < total:
            epoch += 1
            # Reshuffled per epoch, and seeded: the corpus is written source by source, so
            # file order makes the loss track which source the batch came from.
            shuffler.shuffle(order)
            for source, prefix, labels in _batches(
                train_pairs,
                tokenizer,
                train_config.micro_batch_size,
                order=order,
                device=device,
            ):
                output, _ = model.loss_with_logits(source, prefix, labels)
                scaled = output.loss / train_config.gradient_accumulation_steps
                scaled.backward()  # type: ignore[no-untyped-call]
                accumulated += 1
                if accumulated % train_config.gradient_accumulation_steps:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config.max_gradient
                )
                if not torch.isfinite(grad_norm):
                    # Checked before the step, not after: stepping on a non-finite
                    # gradient writes NaN into every weight and no later batch recovers.
                    optimizer.zero_grad(set_to_none=True)
                    _emit("skipped", step=step, reason="non-finite gradient")
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % LOG_EVERY == 0 or step in (entered + 1, total):
                    _emit(
                        "step",
                        step=step,
                        total=total,
                        loss=round(float(output.loss.detach()), 4),
                        accuracy=round(int(output.correct) / max(1, int(output.tokens)), 4),
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        seconds_per_step=round(
                            (time.monotonic() - started) / max(1, step - entered), 3
                        ),
                    )

                if step % train_config.eval_interval == 0 or step >= total:
                    evaluation = _validate(
                        model, tokenizer, dev_pairs, device, train_config.micro_batch_size
                    )
                    _emit("validate", step=step, **asdict(evaluation))
                    payload = {
                        "step": step,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": asdict(model_config),
                        "evaluation": asdict(evaluation),
                        "best": max(best, evaluation.free_running_exact),
                    }
                    _save(directory, payload, "latest.pt")
                    if evaluation.free_running_exact > best:
                        best = evaluation.free_running_exact
                        _save(directory, payload, "best.pt")
                        _emit("best", step=step, free_running_exact=round(best, 4))
                    checkpoint_volume.commit()

                if step >= total:
                    break

        return {
            "run": run,
            "step": step,
            "trained_this_run": step - entered,
            "best_free_running_exact": best,
            "seconds_per_step": round((time.monotonic() - started) / max(1, step - entered), 3),
        }
    finally:
        lock.release(directory, owner)


@app.function(
    image=tifinagh_image,
    gpu=GPU,
    volumes=VOLUMES,
    timeout=TRAIN_TIMEOUT,
    retries=RETRIES,
)
def tifinagh_evaluate(
    run: str = DEFAULT_RUN, split: str = "test", limit: int = 0, name: str = "best.pt"
) -> dict[str, object]:
    """Score a held-out split free-running, against the deterministic character table.

    The table is scored on the same rows in the same pass. A baseline computed elsewhere,
    on a different sample, is not a comparison.
    """
    import torch

    from agbalu.bench.tifinagh import read_split, score_conversion, score_schwa, to_latin
    from agbalu.tifinagh.infer import Transliterator

    _configure_logging()
    checkpoint = RUN_ROOT / run / name
    if not checkpoint.is_file():
        message = f"no checkpoint at {checkpoint}"
        raise FileNotFoundError(message)

    _ensure_splits(SPLIT_ROOT, (split,))
    pairs = read_split(split, directory=SPLIT_ROOT, limit=limit or None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Transliterator.load(checkpoint, device=device)
    _emit("start", run=run, split=split, sentences=len(pairs), device=device)

    batch_size = 256
    hypotheses: list[str] = []
    started = time.monotonic()
    for start in range(0, len(pairs), batch_size):
        window = pairs[start : start + batch_size]
        hypotheses.extend(engine.greedy_batch([pair.tifinagh for pair in window], MAX_PIECES))
        _emit("decoded", done=len(hypotheses), of=len(pairs))

    references = [pair.latin.lower() for pair in pairs]
    table = [to_latin(pair.tifinagh) for pair in pairs]
    report = {
        "run": run,
        "split": split,
        "checkpoint": name,
        "sentences": len(pairs),
        "decoding": "greedy, free-running",
        "seconds": round(time.monotonic() - started, 1),
        "model": {
            "conversion": score_conversion(references, hypotheses).as_dict(),
            "schwa": score_schwa(references, hypotheses).as_dict(),
        },
        "character_table": {
            "conversion": score_conversion(references, table).as_dict(),
            "schwa": score_schwa(references, table).as_dict(),
        },
    }
    results = Path(DATA_PATH) / "bench" / f"tifinagh-{run}-{split}.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    return report
