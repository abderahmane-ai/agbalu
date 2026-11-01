"""Fine-tune restoration heads with measured GPU execution and durable recovery."""

import functools
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from agbalu.encoder.corpus import read_json, write_json
from agbalu.encoder.training import EventLog, StopRequest
from agbalu.llm.progress import Heartbeat, Tracker, stage
from agbalu.llm.provenance import code_stamp, config_hash, file_digest
from agbalu.restoration import checkpoint
from agbalu.restoration.calibration import Calibration, fit
from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.corpus import verify
from agbalu.restoration.dataset import Batch, Example, Windows, collate_update
from agbalu.restoration.evaluation import evaluate
from agbalu.restoration.model import Restorer
from agbalu.restoration.objective import loss_sums, target_counts

MATRIX_DIMENSIONS = 2
ETA_UPDATES = 20


class Log(EventLog):
    def write(self, event: str, **fields: object) -> None:
        super().write(event, **{**fields, "family": "restoration"})


class Epoch(Dataset[Example]):
    def __init__(
        self, offline: Windows, streaming: Windows, settings: Settings, epoch: int
    ) -> None:
        self.offline, self.streaming = offline, streaming
        rng = np.random.default_rng(settings.seed + epoch)
        self.order = rng.permutation(len(offline))
        extra = round(
            len(offline) * settings.streaming_fraction / max(1 - settings.streaming_fraction, 1e-9)
        )
        if settings.streaming_fraction == 1:
            extra = len(streaming)
            self.order = np.empty(0, np.int64)
        self.extra = rng.integers(0, len(streaming), size=extra)
        self.mixed = rng.permutation(len(self.order) + len(self.extra))

    def __len__(self) -> int:
        return len(self.mixed)

    def __getitem__(self, index: int) -> Example:
        chosen = int(self.mixed[index])
        return (
            self.offline[int(self.order[chosen])]
            if chosen < len(self.order)
            else self.streaming[int(self.extra[chosen - len(self.order)])]
        )


class Remaining(Sampler[int]):
    def __init__(self, cursor: int, length: int) -> None:
        self.cursor, self.length = cursor, length

    def __iter__(self) -> Iterator[int]:
        yield from range(self.cursor, self.length)

    def __len__(self) -> int:
        return self.length - self.cursor


def loader(
    dataset: Epoch, settings: Settings, cursor: int, device: torch.device
) -> Iterator[list[Batch]]:
    batches = DataLoader(
        dataset,
        batch_size=settings.batch_windows,
        sampler=Remaining(cursor, len(dataset)),
        num_workers=settings.workers,
        collate_fn=functools.partial(
            collate_update,
            context_tokens=settings.geometry.context_tokens,
            micro_windows=settings.micro_windows,
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=settings.workers > 0,
        prefetch_factor=2 if settings.workers else None,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    for value in batches:
        result: object = value
        if not isinstance(result, list) or not all(isinstance(row, Batch) for row in result):
            message = f"restoration collator returned {type(result).__name__}, expected Batch list"
            raise TypeError(message)
        yield [row for row in result if isinstance(row, Batch)]


def optimizer_for(
    model: Restorer, settings: Settings, device: torch.device
) -> torch.optim.Optimizer:
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        is_encoder = name.startswith("encoder.")
        rate = settings.encoder_lr if is_encoder else settings.head_lr
        if name.startswith("encoder.layers."):
            depth = int(name.split(".")[2])
            rate *= settings.layer_decay ** (settings.geometry.layers - 1 - depth)
        elif is_encoder:
            rate *= settings.layer_decay**settings.geometry.layers
        decay = (
            settings.weight_decay
            if parameter.ndim >= MATRIX_DIMENSIONS and "document_edges" not in name
            else 0.0
        )
        groups.setdefault((rate, decay), []).append(parameter)
    unique = [parameter for values in groups.values() for parameter in values]
    if len({id(parameter) for parameter in unique}) != len(list(model.parameters())):
        msg = "optimizer groups do not cover parameters exactly once"
        raise ValueError(msg)
    return torch.optim.AdamW(
        [
            {"params": values, "lr": rate, "peak_lr": rate, "weight_decay": decay}
            for (rate, decay), values in groups.items()
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
        fused=device.type == "cuda",
    )


def rate_multiplier(step: int, total: int, settings: Settings) -> float:
    progress = min((step + 1) / total, 1.0)
    if progress < settings.warmup_fraction:
        return progress / settings.warmup_fraction
    cosine = (
        1
        + math.cos(math.pi * (progress - settings.warmup_fraction) / (1 - settings.warmup_fraction))
    ) / 2
    return settings.final_lr_fraction + (1 - settings.final_lr_fraction) * cosine


def update(
    model: Restorer,
    optimizer: torch.optim.Optimizer,
    batches: list[Batch],
    settings: Settings,
    device: torch.device,
) -> dict[str, float]:
    balanced = settings.objective == "balanced"
    counts = sum(
        (target_counts(batch, balanced=balanced) for batch in batches),
        torch.zeros(16 if balanced else 3, dtype=torch.long),
    )
    if int(counts[0]) == 0:
        msg = "optimizer update contains no punctuation targets"
        raise ValueError(msg)
    denominators = counts.to(device)
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=device)
    contributions = torch.zeros(14, device=device) if balanced else None
    progress = Tracker("restoration micro-batches", len(batches))
    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scores = model.forward(batch)
            losses = loss_sums(
                scores,
                batch,
                denominators=denominators,
                balanced=balanced,
            )
        torch.autograd.backward(losses.normalized(denominators))
        totals += torch.stack((losses.punctuation, losses.casing, losses.structure)).detach()
        if contributions is not None and losses.contributions is not None:
            contributions += losses.contributions.detach()
        progress.advance()
    primary = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith(("casing.", "structure."))
    ]
    norm = torch.nn.utils.clip_grad_norm_(primary, settings.gradient_clip)
    case_norm = torch.nn.utils.clip_grad_norm_(model.casing.parameters(), settings.gradient_clip)
    torch.nn.utils.clip_grad_norm_(model.structure.parameters(), settings.gradient_clip)
    observed = torch.cat((totals, norm.reshape(1), case_norm.reshape(1))).tolist()
    if not all(math.isfinite(value) for value in observed):
        optimizer.zero_grad(set_to_none=True)
        msg = f"non-finite restoration losses/gradient: {observed}"
        raise FloatingPointError(msg)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    normalized = totals if balanced else totals / denominators.clamp(min=1)
    values = normalized.tolist()
    metrics = {
        "loss": sum(values),
        "punctuation_loss": values[0],
        "casing_loss": values[1],
        "structure_loss": values[2],
        "grad_norm": observed[3],
        "case_grad_norm": observed[4],
        "supervised_gaps": float(counts[0]),
        "supervised_case_components": float(counts[1]),
        "owned_words": float(sum(int(batch.owned_words.sum()) for batch in batches)),
    }
    if contributions is not None:
        names = (
            "presence_positive",
            "presence_none",
            *(f"type_{i}" for i in range(1, 9)),
            *(f"case_{i}" for i in range(3)),
            "structure",
        )
        metrics.update(
            (f"loss_{name}", value)
            for name, value in zip(names, contributions.tolist(), strict=True)
        )
        metrics.update(
            (f"targets_{name}", float(value))
            for name, value in zip(names[:-1], counts[3:].tolist(), strict=True)
        )
    return metrics


def smoke_identity(settings: Settings, locations: Locations) -> str:
    directory = Path(__file__).parent
    return config_hash(
        {
            "settings": settings.model_dump(),
            "data": file_digest(locations.data / "corpus.stats.json"),
            "source": {path.name: file_digest(path) for path in directory.glob("*.py")},
            "encoder_source": {
                path.name: file_digest(path)
                for path in directory.parent.joinpath("encoder").glob("*.py")
            },
            "initial_checkpoint": file_digest(locations.encoder / "best_mlm.json"),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    )


class Trainer:
    def __init__(
        self, settings: Settings, locations: Locations, run: str, device: torch.device
    ) -> None:
        self.settings, self.locations, self.device = settings, locations, device
        self.directory = locations.runs / run
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.directory)
        self.data = verify(locations.data)
        templates = self.data["templates"]
        if not isinstance(templates, list) or not all(isinstance(name, str) for name in templates):
            msg = "data manifest does not contain structural templates"
            raise ValueError(msg)
        self.identity = config_hash(
            {
                "settings": settings.model_dump(),
                "data": self.data,
                "initial_checkpoint": file_digest(locations.encoder / "best_mlm.json"),
            }
        )
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)
        random.seed(settings.seed)
        self.model = Restorer.from_settings(settings, tuple(str(name) for name in templates)).to(
            device
        )
        self.initialized_from = self.model.load_encoder(locations.encoder)
        self.optimizer = optimizer_for(self.model, settings, device)
        self.offline = Windows(locations.data, settings, 0)
        self.streaming = Windows(locations.data, settings, 0, streaming=True)
        self.dev = Windows(locations.data, settings, 1)
        self.dev_stream = Windows(locations.data, settings, 1, streaming=True)
        epoch = Epoch(self.offline, self.streaming, settings, 0)
        self.steps_per_epoch = math.ceil(len(epoch) / settings.batch_windows)
        self.total_steps = self.steps_per_epoch * settings.epochs
        self.state = checkpoint.State()
        self.stop = StopRequest()
        self.last_saved = time.monotonic()
        self.phase = "initialization"
        if settings.compile and device.type == "cuda":
            self.model.enable_compile()

    def snapshot(self) -> Path:
        with stage("restoration checkpoint", step=self.state.step):
            path = checkpoint.save(
                self.directory, self.model, self.optimizer, self.state, self.identity
            )
        self.last_saved = time.monotonic()
        self.log.write("checkpoint", step=self.state.step, path=str(path))
        return path

    def validate(self) -> dict[str, object]:
        self.phase = "validation"
        started = time.monotonic()
        proof = self.locations.data / "selection.stats.json"
        selected = read_json(proof).get("chosen") if proof.exists() else None
        correction: Calibration | None = None
        if isinstance(selected, dict) and selected.get("variant") == "calibrated":
            correction, _ = fit(self.model, self.dev, self.device)
        calibrated = correction.apply if correction is not None else None
        fold: Literal["selection", "diagnostic"] = (
            "selection" if isinstance(selected, dict) else "diagnostic"
        )
        limit = 512 if isinstance(selected, dict) else self.settings.validation_windows
        with torch.random.fork_rng(
            devices=[self.device.index or 0] if self.device.type == "cuda" else []
        ):
            offline = evaluate(
                self.model, self.dev, self.device, limit, fold=fold, calibration=calibrated
            )
            streaming = evaluate(
                self.model, self.dev_stream, self.device, limit, fold=fold, calibration=calibrated
            )
        score = (
            0.4 * float(str(offline["punctuation_macro_f1"]))
            + 0.3 * float(str(streaming["punctuation_macro_f1"]))
            + 0.3 * float(str(offline["casing_macro_f1"]))
        )
        result = {
            "offline": offline,
            "streaming": streaming,
            "score": score,
            "step": self.state.step,
            "seconds": time.monotonic() - started,
            "calibration": correction.model_dump() if correction is not None else {},
        }
        write_json(
            self.directory / "evaluations" / f"step-{self.state.step:07d}.stats.json", result
        )
        self.log.write("eval", **result)
        if self.state.best_score is None or score > self.state.best_score:
            self.state.best_score = score
            self.state.stale_evaluations = 0
            snapshot = self.snapshot()
            checkpoint.promote(self.directory, snapshot, result)
            self.log.write("best", step=self.state.step, score=score)
        else:
            self.state.stale_evaluations += 1
        checkpoint.prune(self.directory)
        return result

    def run(self, *, stop_after: int | None = None, smoke: bool = False) -> dict[str, object]:
        with checkpoint.run_lock(self.directory):
            return self.run_locked(stop_after=stop_after, smoke=smoke)

    def run_locked(self, *, stop_after: int | None, smoke: bool) -> dict[str, object]:
        quality = self.locations.data / "selection.stats.json"
        if (
            self.settings.require_pilot
            and not smoke
            and self.device.type == "cuda"
            and (
                not quality.exists()
                or read_json(quality).get("identity")
                != smoke_identity(self.settings, self.locations)
                or read_json(quality).get("passed") is not True
            )
        ):
            msg = "full restoration training requires a selected, measured pilot for this code/data"
            raise ValueError(msg)
        proof = self.locations.data / "smoke.stats.json"
        if self.settings.require_smoke and not smoke and self.device.type == "cuda":
            if not proof.exists():
                msg = (
                    f"{proof}: missing smoke proof; "
                    "run make smoke FAMILY=restoration RUN=restoration-smoke"
                )
                raise ValueError(msg)
            stored_identity = read_json(proof).get("identity")
            current_identity = smoke_identity(self.settings, self.locations)
            if stored_identity != current_identity:
                msg = (
                    f"{proof}: code/config changed since last smoke proof "
                    f"(expected {current_identity[:12]}, found {str(stored_identity)[:12]}); "
                    "run make smoke FAMILY=restoration RUN=restoration-smoke"
                )
                raise ValueError(msg)
        config = self.directory / "config.json"
        if config.exists() and read_json(config).get("identity") != self.identity:
            msg = f"{config}: configuration changed on resume"
            raise ValueError(msg)
        write_json(
            config,
            {
                "family": "restoration",
                "settings": self.settings.model_dump(),
                "locations": self.locations.model_dump(mode="json"),
                "templates": list(self.model.templates),
                "identity": self.identity,
                "code": code_stamp(),
                "initialized_from": self.initialized_from,
            },
        )
        saved = checkpoint.latest(self.directory)
        if saved:
            self.state = checkpoint.load(saved, self.model, self.optimizer, self.identity)
        first_step, first_tokens = self.state.step, self.state.tokens
        self.stop.install()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        seconds: list[float] = []
        sizes: list[int] = []
        display_steps = (
            min(self.total_steps, stop_after) if stop_after is not None else self.total_steps
        )
        self.log.write(
            "start",
            step=self.state.step,
            total_steps=display_steps,
            schedule_steps=self.total_steps,
            parameters=sum(parameter.numel() for parameter in self.model.parameters()),
            cpu_threads=torch.get_num_threads(),
            torch_version=torch.__version__,
            precision="bfloat16 autocast" if self.device.type == "cuda" else "float32",
            resumed_from=str(saved) if saved else None,
            settings=self.settings.model_dump(),
            train_windows=len(self.offline),
            train_stream_windows=len(self.streaming),
            data_composition=self.data.get("by_source"),
        )
        ended = "schedule"
        self.model.train()
        if first_step == 0 and not smoke:
            write_json(
                self.directory / "baseline.stats.json",
                evaluate(
                    self.model,
                    self.dev,
                    self.device,
                    self.settings.validation_windows,
                    baseline=True,
                ),
            )
        with Heartbeat(
            "restoration training", status=lambda: {"step": self.state.step, "phase": self.phase}
        ):
            while self.state.epoch < self.settings.epochs:
                if self.state.ended == "development_patience":
                    ended = self.state.ended
                    break
                dataset = Epoch(self.offline, self.streaming, self.settings, self.state.epoch)
                iterator = iter(loader(dataset, self.settings, self.state.cursor, self.device))
                while self.state.cursor < len(dataset):
                    if self.stop.number is not None or (
                        stop_after is not None and self.state.step >= stop_after
                    ):
                        ended = f"signal {self.stop.number}" if self.stop.number else "stop_after"
                        break
                    self.phase = "loading data"
                    tick = time.monotonic()
                    try:
                        batches = next(iterator)
                    except StopIteration:
                        break
                    if not batches:
                        break
                    wait = time.monotonic() - tick
                    self.phase = "training"
                    multiplier = rate_multiplier(self.state.step, self.total_steps, self.settings)
                    for group in self.optimizer.param_groups:
                        group["lr"] = group["peak_lr"] * multiplier
                    metrics = update(
                        self.model, self.optimizer, batches, self.settings, self.device
                    )
                    elapsed = time.monotonic() - tick
                    count = sum(int(batch.attention.sum()) for batch in batches)
                    self.state.cursor += sum(batch.words.size(0) for batch in batches)
                    self.state.step += 1
                    self.state.tokens += count
                    self.state.words += int(metrics["supervised_case_components"])
                    seconds.append(elapsed)
                    sizes.append(count)
                    recent = (
                        seconds[-ETA_UPDATES:]
                        if len(seconds) > ETA_UPDATES
                        else seconds[1:] or seconds
                    )
                    self.log.write(
                        "step",
                        step=self.state.step,
                        total_steps=display_steps,
                        **metrics,
                        tokens=self.state.tokens,
                        tokens_per_s=count / elapsed,
                        lr=self.settings.encoder_lr * multiplier,
                        taken=True,
                        mfu=None,
                        memory=torch.cuda.max_memory_allocated(self.device)
                        if self.device.type == "cuda"
                        else None,
                        step_s=elapsed,
                        wait_s=wait,
                        encoder_blocks=sum(batch.ids.size(0) for batch in batches),
                        padding_fraction=1 - count / sum(batch.ids.numel() for batch in batches),
                        epoch=self.state.epoch,
                        eta_h=(display_steps - self.state.step) * sum(recent) / len(recent) / 3600,
                    )
                    if not smoke and self.state.step >= self.state.next_evaluation:
                        self.state.next_evaluation = self.state.step + max(
                            1,
                            math.ceil(
                                self.steps_per_epoch / self.settings.validation_checks_per_epoch
                            ),
                        )
                        self.validate()
                    if (
                        self.state.step % self.settings.checkpoint_updates == 0
                        or time.monotonic() - self.last_saved >= self.settings.checkpoint_seconds
                    ):
                        self.snapshot()
                        checkpoint.prune(self.directory)
                    if self.state.stale_evaluations >= self.settings.patience:
                        ended = "development_patience"
                        break
                if ended != "schedule":
                    break
                self.state.epoch += 1
                self.state.cursor = 0
            if self.state.step > first_step and not smoke:
                self.validate()
            self.state.ended = ended
            path = self.snapshot()
            checkpoint.prune(self.directory)
        steady = seconds[1:] or seconds
        steady_sizes = sizes[1:] if len(seconds) > 1 else sizes
        result: dict[str, object] = {
            "family": "restoration",
            "ended": ended,
            "step": self.state.step,
            "steps_this_run": self.state.step - first_step,
            "tokens_total": self.state.tokens,
            "tokens_this_run": self.state.tokens - first_tokens,
            "tokens_per_second": sum(steady_sizes) / sum(steady) if steady else None,
            "step_seconds_mean": sum(steady) / len(steady) if steady else None,
            "wall_seconds": time.monotonic() - started,
            "best_score": self.state.best_score,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
            "latest": path.name,
            "code": code_stamp(),
            "config_hash": self.identity,
        }
        write_json(self.directory / "summary.stats.json", result)
        self.log.write("end", **result)
        return result
