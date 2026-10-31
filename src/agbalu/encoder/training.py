"""Train the encoder with target-weighted updates and verified recovery."""

from __future__ import annotations

import json
import math
import random
import signal
import statistics
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from agbalu.encoder import checkpoint, evaluation
from agbalu.encoder.attention import make_layout
from agbalu.encoder.batches import Batches, Packed, trim_to_budget
from agbalu.encoder.configuration import MATRIX_AXES, SPECIAL_TOKENS, Locations, Recipe
from agbalu.encoder.corpus import read_json, verify_data, write_json
from agbalu.encoder.masking import Masked, mask_groups
from agbalu.encoder.model import Encoder
from agbalu.encoder.schedule import context_length, learning_rate, mask_probability
from agbalu.encoder.selection import choose
from agbalu.llm.progress import Heartbeat, Tracker, stage
from agbalu.llm.provenance import code_stamp, config_hash, file_digest


class EventLog:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "log.jsonl"

    def write(self, event: str, **fields: object) -> None:
        line = json.dumps(
            {
                "event": event,
                "time": time.time(),
                "family": "encoder",
                "schema_version": 1,
                **fields,
            },
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class StopRequest:
    def __init__(self) -> None:
        self.number: int | None = None

    def receive(self, number: int, frame: FrameType | None) -> None:
        del frame
        self.number = number

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.receive)
        signal.signal(signal.SIGINT, self.receive)


class Order(Sampler[int]):
    def __init__(self, length: int, cursor: int, seed: int) -> None:
        self.length, self.cursor, self.seed = length, cursor, seed

    def __iter__(self) -> Iterator[int]:
        epoch, offset = divmod(self.cursor, self.length)
        while True:
            order = np.random.default_rng(self.seed + epoch).permutation(self.length)
            yield from (int(index) for index in order[offset:])
            epoch, offset = epoch + 1, 0


def optimizer_for(model: Encoder, recipe: Recipe, device: torch.device) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    exempt: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        target = (
            decay
            if parameter.ndim >= MATRIX_AXES and name not in {"embedding.weight", "relative"}
            else exempt
        )
        target.append(parameter)
    if len({id(p) for p in [*decay, *exempt]}) != len(list(model.parameters())):
        message = "optimizer parameter groups do not cover trainable parameters exactly once"
        raise ValueError(message)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": recipe.weight_decay},
            {"params": exempt, "weight_decay": 0.0},
        ],
        lr=recipe.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )


def smoke_identity(recipe: Recipe, device: torch.device) -> str:
    root = Path(__file__).parent
    sources = {
        name: file_digest(root / name)
        for name in (
            "attention.py",
            "model.py",
            "routing.py",
            "training.py",
            "batches.py",
            "masking.py",
        )
    }
    return config_hash(
        {
            "geometry": recipe.geometry.model_dump(),
            "micro_tokens": recipe.micro_tokens,
            "batch_tokens": recipe.batch_tokens,
            "peak_lr": recipe.peak_lr,
            "compile": recipe.compile,
            "torch": torch.__version__,
            "sources": sources,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        }
    )


def stream(batches: Batches, cursor: int, recipe: Recipe, device: torch.device) -> Iterator[Packed]:
    loader = DataLoader(
        batches,
        batch_size=None,
        sampler=Order(len(batches), cursor, recipe.seed),
        num_workers=recipe.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=recipe.workers > 0,
        prefetch_factor=2 if recipe.workers else None,
        generator=torch.Generator().manual_seed(recipe.seed),
    )
    yield from loader


def update(
    model: Encoder,
    optimizer: torch.optim.Optimizer,
    batch: list[Packed],
    recipe: Recipe,
    state: checkpoint.State,
    device: torch.device,
) -> dict[str, float]:
    masked: list[tuple[Masked, Packed]] = []
    targets = torch.zeros((), device=device, dtype=torch.long)
    for index, row in enumerate(batch):
        ids, groups = (
            row.ids.to(device, non_blocking=True),
            row.groups.to(device, non_blocking=True),
        )
        chosen = mask_groups(
            ids,
            groups,
            mask_probability(state.tokens, recipe.total_tokens),
            recipe.seed
            + (state.cursor - len(batch) + index) * 1009
            + state.context_tokens * 1000003,
            recipe.geometry.vocab_size,
        )
        masked.append((chosen, row))
        targets += chosen.targets
    sums = torch.zeros(2, device=device, dtype=torch.float32)
    tracker = Tracker(f"step {state.step + 1} micro-batches", len(batch))
    optimizer.zero_grad(set_to_none=True)
    for chosen, row in masked:
        layout = make_layout(
            row.documents.to(device, non_blocking=True),
            row.positions.to(device, non_blocking=True),
            row.valid.to(device, non_blocking=True),
            flex=recipe.geometry.backend == "flex",
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, correct = model.forward(chosen.ids, layout, chosen.labels)
        torch.autograd.backward(loss / targets.clamp(min=1))
        sums += torch.stack((loss.detach(), correct.detach().float()))
        tracker.advance()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
    loss_sum, correct_sum, count, gradient = torch.stack((sums[0], sums[1], targets, norm)).tolist()
    if not count or not all(math.isfinite(value) for value in (loss_sum, gradient)):
        optimizer.zero_grad(set_to_none=True)
        message = (
            f"update {state.step + 1}: loss_sum={loss_sum}, grad_norm={gradient}, targets={count}"
        )
        raise FloatingPointError(message)
    rate = learning_rate(state.tokens + sum(row.tokens for row in batch), recipe)
    diagnostics: dict[str, float] = {}
    if (state.step + 1) % 100 == 0:
        for index, layer in enumerate(model.layers):
            squares = [
                torch.linalg.vector_norm(p.grad, dtype=torch.float32).square()
                for p in layer.parameters()
                if p.grad is not None
            ]
            diagnostics[f"layer_{index:02d}_clipped_grad_norm"] = float(
                torch.stack(squares).sum().sqrt()
            )
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "loss": loss_sum / count,
        "accuracy": correct_sum / count,
        "targets": count,
        "grad_norm": gradient,
        "lr": rate,
        "clip_coefficient": min(1.0, recipe.gradient_clip / max(gradient, 1e-12)),
        **diagnostics,
    }


class Trainer:
    def __init__(
        self, recipe: Recipe, locations: Locations, run: str, device: torch.device
    ) -> None:
        self.recipe, self.locations, self.device = recipe, locations, device
        self.directory = locations.runs / run
        self.log = EventLog(self.directory)
        self.data = verify_data(locations.data)
        self.identity = checkpoint.run_identity(recipe, self.data)
        self.state = checkpoint.State()
        torch.manual_seed(recipe.seed)
        random.seed(recipe.seed)
        np.random.seed(recipe.seed)
        self.model = Encoder(recipe.geometry).to(device)
        self.optimizer = optimizer_for(self.model, recipe, device)
        if recipe.compile and device.type == "cuda":
            self.model.compiled_features = torch.compile(
                self.model.contextualise_packed, dynamic=False
            )
        self.validation = Batches(
            locations.data,
            512,
            recipe.micro_tokens,
            recipe.seed,
            validation=True,
            limit_documents=recipe.validation_documents,
        )
        self.last_saved = time.monotonic()
        self.phase = "starting"
        self.stop = StopRequest()

    def snapshot(self) -> Path:
        with stage("encoder checkpoint", step=self.state.step):
            path = checkpoint.save(
                self.directory, self.model, self.optimizer, self.state, self.identity
            )
        self.last_saved = time.monotonic()
        self.log.write("checkpoint", step=self.state.step, path=str(path))
        return path

    def evaluate(self, *, tasks: bool) -> None:
        self.phase = "evaluating"
        started = time.monotonic()
        with torch.random.fork_rng(
            devices=[self.device.index or 0] if self.device.type == "cuda" else []
        ):
            health = evaluation.fixed_mlm(self.model, self.validation, self.device)
            texts, diagnostic = evaluation.preview(
                self.model, self.validation, self.locations, self.device
            )
            self.log.write("probe", step=self.state.step, texts=texts)
            self.log.write("diagnostics", step=self.state.step, **diagnostic)
            ce = float(str(health["ce"]))
            improved_mlm = self.state.best_ce is None or ce < self.state.best_ce
            self.state.fast_improved = self.state.fast_improved or improved_mlm
            if improved_mlm:
                self.state.best_ce = ce
            scores: dict[str, object] = {}
            deployment_improved = False
            if tasks:
                baseline_path = self.locations.data / "baseline.stats.json"
                if baseline_path.is_file():
                    reference = read_json(baseline_path)
                else:
                    with stage("published encoder development baseline"):
                        reference = evaluation.baseline(self.locations, self.device)
                    write_json(baseline_path, reference)
                clean, corrupted = evaluation.primary_scores(reference)
                self.state.reference_clean, self.state.reference_corrupted = clean, corrupted
                with stage("frozen encoder POS development probe"):
                    scores = evaluation.downstream(self.model, self.locations, self.device)
                clean, corrupted = evaluation.primary_scores(scores)
                decision = choose(clean, corrupted, self.state, self.recipe)
                deployment_improved = decision.improved
                scores["selection"] = {"eligible": decision.eligible, "reason": decision.reason}
                if deployment_improved:
                    self.state.best_score, self.state.best_corrupted = clean, corrupted
                    self.state.stale_evaluations = 0
                elif not self.state.fast_improved:
                    self.state.stale_evaluations += 1
                self.state.fast_improved = False
        report = {
            "step": self.state.step,
            "tokens": self.state.tokens,
            "mlm": health,
            "downstream": scores,
            "seconds": time.monotonic() - started,
        }
        write_json(
            self.directory / "evaluations" / f"step-{self.state.step:07d}.stats.json", report
        )
        self.log.write("eval", **report)
        if improved_mlm or deployment_improved:
            snapshot = self.snapshot()
            if improved_mlm:
                checkpoint.promote(self.directory, snapshot, "best_mlm", health)
            if deployment_improved:
                checkpoint.promote(self.directory, snapshot, "best_deployment", scores)
                self.log.write("best", step=self.state.step, score=self.state.best_score)
            checkpoint.prune(self.directory)
        self.model.train()

    def run(
        self, *, stop_after: int | None = None, install_signals: bool = True
    ) -> dict[str, object]:
        with checkpoint.run_lock(self.directory):
            return self.run_locked(stop_after=stop_after, install_signals=install_signals)

    def run_locked(self, *, stop_after: int | None, install_signals: bool) -> dict[str, object]:
        if self.recipe.require_smoke and self.device.type == "cuda":
            proof = self.locations.data / "smoke.stats.json"
            if not proof.is_file() or read_json(proof).get("identity") != smoke_identity(
                self.recipe, self.device
            ):
                message = (
                    "current model/batch/environment needs "
                    "make smoke FAMILY=encoder PART=throughput"
                )
                raise ValueError(message)
        config_path = self.directory / "config.json"
        if config_path.is_file() and read_json(config_path).get("identity") != self.identity:
            message = f"{config_path}: refusing changed recipe/data on resume"
            raise ValueError(message)
        write_json(
            config_path,
            {
                "family": "encoder",
                "identity": self.identity,
                "recipe": self.recipe.model_dump(),
                "code": code_stamp(),
            },
        )
        found = checkpoint.latest(self.directory)
        if found:
            self.state = checkpoint.restore(found, self.model, self.optimizer, self.identity)
        if install_signals:
            self.stop.install()
        first_tokens, first_step = self.state.tokens, self.state.step
        interval = max(1, math.ceil(int(str(self.data["train_text_tokens"])) / 4))
        if self.state.next_fast_tokens == 0:
            self.state.next_fast_tokens, self.state.next_task_tokens = interval, interval * 4
        self.log.write(
            "start",
            step=self.state.step,
            total_steps=math.ceil(self.recipe.total_tokens / self.recipe.batch_tokens),
            parameters=self.recipe.geometry.parameters,
            resumed_from=str(found) if found else None,
            config_hash=self.identity,
            device=str(self.device),
            recipe=self.recipe.model_dump(),
        )
        times: list[float] = []
        sizes: list[int] = []
        started = time.monotonic()
        ended = "schedule"
        self.model.train()
        with Heartbeat(
            "encoder train", status=lambda: {"phase": self.phase, "step": self.state.step}
        ):
            if not found and self.state.tokens < self.recipe.total_tokens:
                self.evaluate(tasks=self.recipe.task_evaluation)
            iterator: Iterator[Packed] | None = None
            while self.state.tokens < self.recipe.total_tokens:
                if self.stop.number is not None:
                    ended = f"signal {self.stop.number}"
                    break
                if stop_after is not None and self.state.step >= stop_after:
                    ended = "stop_after"
                    break
                context = context_length(self.state.tokens, self.recipe)
                if context != self.state.context_tokens or iterator is None:
                    if context != self.state.context_tokens:
                        checkpoint.promote(self.directory, self.snapshot(), "transition", {})
                        self.state.context_tokens, self.state.cursor = context, 0
                    with stage("build encoder batches", context_tokens=context):
                        dataset = Batches(
                            self.locations.data, context, self.recipe.micro_tokens, self.recipe.seed
                        )
                    iterator = stream(dataset, self.state.cursor, self.recipe, self.device)
                self.phase = "loading data"
                batch: list[Packed] = []
                count = 0
                tick = time.monotonic()
                wanted = min(self.recipe.batch_tokens, self.recipe.total_tokens - self.state.tokens)
                while count < wanted:
                    row = next(iterator)
                    remaining = self.recipe.total_tokens - self.state.tokens - count
                    if row.tokens > remaining:
                        self.state.budget_tail_tokens += row.tokens - remaining
                        row = trim_to_budget(row, remaining)
                    batch.append(row)
                    count += row.tokens
                    self.state.cursor += 1
                wait = time.monotonic() - tick
                self.phase = "training"
                metrics = update(
                    self.model, self.optimizer, batch, self.recipe, self.state, self.device
                )
                elapsed = time.monotonic() - tick
                self.state.step += 1
                self.state.tokens += count
                text_count = sum(
                    int(((row.ids >= SPECIAL_TOKENS) & row.valid).sum()) for row in batch
                )
                self.state.text_tokens += text_count
                self.state.targets += int(metrics["targets"])
                times.append(elapsed)
                sizes.append(count)
                if self.state.step % 100 == 0:
                    self.model.eval()
                    texts, diagnostic = evaluation.preview(
                        self.model, self.validation, self.locations, self.device
                    )
                    self.log.write("probe", step=self.state.step, texts=texts)
                    self.log.write("diagnostics", step=self.state.step, **diagnostic)
                    self.model.train()
                self.log.write(
                    "step",
                    step=self.state.step,
                    total_steps=math.ceil(self.recipe.total_tokens / self.recipe.batch_tokens),
                    tokens=self.state.tokens,
                    text_tokens=self.state.text_tokens,
                    context_tokens=self.state.context_tokens,
                    **metrics,
                    taken=True,
                    step_s=elapsed,
                    wait_s=wait,
                    tokens_per_s=count / elapsed,
                    mask_probability=metrics["targets"] / text_count,
                    memory=torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else None,
                    mfu=None,
                    eta_h=max(0, self.recipe.total_tokens - self.state.tokens)
                    * statistics.median(
                        seconds / tokens
                        for seconds, tokens in zip(times[-20:], sizes[-20:], strict=True)
                    )
                    / 3600,
                )
                if self.state.tokens >= self.state.next_fast_tokens:
                    tasks = (
                        self.recipe.task_evaluation
                        and self.state.tokens >= self.state.next_task_tokens
                    )
                    self.state.next_fast_tokens = (self.state.tokens // interval + 1) * interval
                    if tasks:
                        self.state.next_task_tokens = (
                            (self.state.tokens // (interval * 4) + 1) * interval * 4
                        )
                    self.evaluate(tasks=tasks)
                if (
                    self.state.step % self.recipe.checkpoint_updates == 0
                    or time.monotonic() - self.last_saved >= self.recipe.checkpoint_seconds
                ):
                    self.snapshot()
                    checkpoint.prune(self.directory)
                exposure = self.state.text_tokens / int(str(self.data["train_text_tokens"]))
                if (
                    exposure >= self.recipe.minimum_passes
                    and self.state.stale_evaluations >= self.recipe.patience
                ):
                    ended = "development_patience"
                    break
            if self.state.step > first_step:
                self.evaluate(tasks=self.recipe.task_evaluation)
            snapshot = self.snapshot()
            checkpoint.prune(self.directory)
        steady = times[1:] if len(times) > 1 else times
        rate = sum(sizes[1:] if len(sizes) > 1 else sizes) / sum(steady) if steady else None
        result: dict[str, object] = {
            "ended": ended,
            "step": self.state.step,
            "steps_this_run": self.state.step - first_step,
            "tokens_total": self.state.tokens,
            "tokens_this_run": self.state.tokens - first_tokens,
            "tokens_per_second": rate,
            "step_seconds_median": statistics.median(steady) if steady else None,
            "wall_seconds": time.monotonic() - started,
            "best_ce": self.state.best_ce,
            "best_pos": self.state.best_score,
            "latest": snapshot.name,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
            "budget_excess_tokens": max(0, self.state.tokens - self.recipe.total_tokens),
            "budget_tail_tokens": self.state.budget_tail_tokens,
        }
        write_json(self.directory / "summary.stats.json", result)
        self.log.write("end", **result)
        return result
