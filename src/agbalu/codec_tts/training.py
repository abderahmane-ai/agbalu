"""Train compact speech adapters through deterministic optimizer windows."""

from __future__ import annotations

import hashlib
import json
import math
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import FrameType

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from agbalu.codec_tts.adaptation import disabled_adapter
from agbalu.codec_tts.artifacts import event_log, provenance, run_lock, write_json
from agbalu.codec_tts.backbone import Generator
from agbalu.codec_tts.checkpoint import (
    Pointer,
    RunState,
    latest_snapshot,
    preserved_rng,
    prune_snapshots,
    restore_snapshot,
    retain_candidate,
    save_snapshot,
)
from agbalu.codec_tts.configuration import Recipe
from agbalu.codec_tts.dataset import SpeechDataset, collate_cpu
from agbalu.codec_tts.evaluation import heldout_loss
from agbalu.codec_tts.layout import IGNORE
from agbalu.codec_tts.objective import codec_loss
from agbalu.llm.progress import Heartbeat

Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
SpeechEvaluation = Callable[[Generator, Path], dict[str, object]]
BATCH_FIELDS = 5
MIN_ETA_UPDATES = 2


def scheduled_rate(step: int, total: int, recipe: Recipe) -> float:
    warmup = max(1, math.ceil(total * recipe.warmup_fraction))
    if step < warmup:
        return recipe.peak_lr * (step + 1) / warmup
    fraction = min(1, (step + 1 - warmup) / max(1, total - warmup))
    return (
        recipe.minimum_lr
        + (recipe.peak_lr - recipe.minimum_lr) * (1 + math.cos(math.pi * fraction)) / 2
    )


def validated_batch(value: object) -> Batch:
    if (
        not isinstance(value, tuple)
        or len(value) != BATCH_FIELDS
        or not all(isinstance(item, Tensor) for item in value)
    ):
        message = f"invalid collated speech batch type={type(value).__name__}"
        raise TypeError(message)
    return value[0], value[1], value[2], value[3], value[4]


def perform_update(
    model: Generator, optimizer: torch.optim.Optimizer, batches: Sequence[Batch], recipe: Recipe
) -> dict[str, object]:
    denominator = sum(int(batch[4].ne(IGNORE).sum().item()) for batch in batches)
    if denominator <= 0:
        message = "optimizer window has no supervised codec decisions"
        raise ValueError(message)
    device = model.audio_embedding.weight.device
    optimizer.zero_grad(set_to_none=True)
    nats = torch.zeros(8, dtype=torch.float64, device=device)
    counts = torch.zeros(8, dtype=torch.long, device=device)
    mixed = 0
    for cpu_text, cpu_codes, cpu_audio, cpu_valid, cpu_labels in batches:
        indices = cpu_labels.ne(IGNORE).any(dim=-1).reshape(-1).nonzero().flatten().to(device)
        moved = [
            value.to(device, non_blocking=device.type == "cuda")
            for value in (cpu_text, cpu_codes, cpu_audio, cpu_valid, cpu_labels)
        ]
        text, codes, audio, valid, labels = moved
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.hidden(text, codes, audio, valid).reshape(-1, model.geometry.hidden_size)
            selected = hidden.index_select(0, indices)
            selected_labels = labels.reshape(-1, 8).index_select(0, indices)
            loss = codec_loss(model.classify(selected), selected_labels)
        torch.autograd.backward(loss.numerator / denominator)
        nats += loss.stream_nats.detach().double()
        counts += loss.stream_counts.detach()
        mixed += int(cpu_valid.sum().item())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, recipe.gradient_clip)
    if not bool((torch.isfinite(norm) & torch.isfinite(nats).all()).item()):
        optimizer.zero_grad(set_to_none=True)
        message = f"non-finite speech update: norm={norm.item()}, stream_nats={nats.cpu().tolist()}"
        raise FloatingPointError(message)
    optimizer.step()
    metrics: dict[str, object] = {
        "audio_ce_nats": float((nats.sum() / denominator).item()),
        "loss_numerator": float(nats.sum().item()),
        "supervised_tokens": denominator,
        "stream_nats": nats.cpu().tolist(),
        "stream_counts": counts.cpu().tolist(),
        "grad_norm": float(norm.item()),
        "mixed_positions": mixed,
    }
    optimizer.zero_grad(set_to_none=True)
    return metrics


class StopRequest:
    def __init__(self) -> None:
        self.number: int | None = None

    def receive(self, number: int, frame: FrameType | None) -> None:
        del frame
        self.number = number


@dataclass
class TrainingRun:
    model: Generator
    recipe: Recipe
    training: SpeechDataset
    validation: SpeechDataset
    directory: Path
    speech_evaluation: SpeechEvaluation
    signal_number: int | None = field(default=None, init=False)

    @property
    def identity(self) -> str:
        content = {
            "recipe": self.recipe.digest,
            "data": self.training.identity,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(self.model.audio_embedding.weight.dtype),
            "transformers": version("transformers"),
            "source_hash": hashlib.sha256(
                b"".join(
                    path.name.encode() + path.read_bytes()
                    for path in sorted(Path(__file__).parent.glob("*.py"))
                )
            ).hexdigest(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def train(self, stop_after: int | None = None) -> RunState:
        windows = tuple(self.training.windows())
        if not windows:
            message = "empty speech optimizer schedule"
            raise ValueError(message)
        identity = self.identity
        device = self.model.audio_embedding.weight.device
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.recipe.peak_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            fused=device.type == "cuda",
        )
        state = RunState()
        with run_lock(self.directory), event_log(self.directory, identity) as log:
            identity_path = self.directory / "identity.json"
            if (
                identity_path.is_file()
                and json.loads(identity_path.read_bytes()).get("identity") != identity
            ):
                message = f"run identity changed at {identity_path}; choose a new run ID"
                raise ValueError(message)
            write_json(
                identity_path, {"identity": identity, "recipe": self.recipe.model_dump(mode="json")}
            )
            latest = latest_snapshot(self.directory, identity)
            if latest is not None:
                state = restore_snapshot(latest, self.model, optimizer, identity)
            if state.cursor != state.step or state.step > len(windows):
                message = f"invalid recovered cursor={state.cursor}, step={state.step}"
                message += f", total={len(windows)}"
                raise ValueError(message)
            if state.step >= len(windows):
                log.emit("end", step=state.step, ended="already_complete", updates_this_run=0)
                return state
            if self.recipe.activation_checkpointing:
                self.model.enable_activation_checkpointing()
            stop = StopRequest()
            previous_signals = {
                number: signal.signal(number, stop.receive)
                for number in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                with Heartbeat(
                    "codec TTS", status=lambda: {"family": "codec_tts", "step": state.step}
                ):
                    self.run_updates(
                        windows, optimizer, state, log.emit, stop, identity, stop_after
                    )
                self.signal_number = stop.number
            except BaseException as error:
                log.emit("failure", step=state.step, phase="failed", error=str(error))
                raise
            finally:
                for number, handler in previous_signals.items():
                    signal.signal(number, handler)
            return state

    def run_updates(
        self,
        windows: Sequence[tuple[int, ...]],
        optimizer: torch.optim.Optimizer,
        state: RunState,
        emit: Callable[..., None],
        stop: StopRequest,
        identity: str,
        stop_after: int | None,
    ) -> None:
        device = self.model.audio_embedding.weight.device
        started_step, invocation_start = state.step, time.monotonic()
        started_targets = state.targets
        emit(
            "start",
            step=state.step,
            total_steps=len(windows),
            parameters=sum(p.numel() for p in self.model.parameters()),
            resumed_from=state.step if state.step else None,
            phase="training",
        )
        base_path = self.directory / "base" / "evaluation.stats.json"
        if not base_path.is_file():
            with disabled_adapter(self.model), preserved_rng():
                self.speech_evaluation(self.model, self.directory / "base")
        invocation_start = time.monotonic()
        self.model.train()
        micro = self.recipe.micro_examples
        remaining = windows[state.cursor :]
        batch_indices = [
            list(window[start : start + micro])
            for window in remaining
            for start in range(0, len(window), micro)
        ]
        loader = DataLoader(
            self.training,
            batch_sampler=batch_indices,
            collate_fn=collate_cpu,
            num_workers=self.recipe.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=self.recipe.workers > 0,
            prefetch_factor=2 if self.recipe.workers else None,
            multiprocessing_context="spawn" if self.recipe.workers else None,
        )
        batches = iter(loader)
        cadence = max(1, math.ceil(len(windows) / 8))
        last_save = time.monotonic()
        for window in remaining:
            if stop.number is not None or (stop_after is not None and state.step >= stop_after):
                break
            begun = time.monotonic()
            ready = [validated_batch(next(batches)) for _ in range(math.ceil(len(window) / micro))]
            wait_s = time.monotonic() - begun
            rate = scheduled_rate(state.step, len(windows), self.recipe)
            for group in optimizer.param_groups:
                group["lr"] = rate
            metrics = perform_update(self.model, optimizer, ready, self.recipe)
            state.step += 1
            state.cursor += 1
            state.targets += int(str(metrics["supervised_tokens"]))
            state.examples += len(window)
            elapsed = time.monotonic() - begun
            completed = state.step - started_step
            mean_s = (time.monotonic() - invocation_start) / completed
            eta_s = mean_s * (len(windows) - state.step) if completed >= MIN_ETA_UPDATES else None
            emit(
                "step",
                step=state.step,
                total_steps=len(windows),
                **metrics,
                lr=rate,
                step_s=elapsed,
                wait_s=wait_s,
                eta_s=eta_s,
                supervised_tokens_per_s=int(str(metrics["supervised_tokens"])) / elapsed,
                target_audio_s_per_s=sum(
                    self.training.entries[self.training.pairs[index].target].duration_s
                    for index in window
                )
                / elapsed,
                targets=state.targets,
                targets_this_run=state.targets - started_targets,
                memory=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                phase="training",
            )
            evaluated = state.step % cadence == 0 or state.step == len(windows)
            is_best = False
            candidate_score: float | None = None
            if evaluated:
                with preserved_rng():
                    scores = heldout_loss(
                        self.model, self.validation, self.recipe.evaluation_examples
                    )
                value = float(str(scores["audio_ce_nats"]))
                is_best = state.best_ce is None or value < state.best_ce
                if is_best:
                    state.best_ce = value
                emit("eval", step=state.step, **scores)
                if state.step % (2 * cadence) == 0 or state.step == len(windows):
                    with preserved_rng():
                        speech = self.speech_evaluation(
                            self.model, self.directory / "speech" / f"step-{state.step:08d}"
                        )
                    selection = speech.get("selection")
                    if isinstance(selection, dict):
                        score = selection.get("cer_excess")
                        if (
                            isinstance(score, int | float)
                            and selection.get("termination_failures") == 0
                        ):
                            candidate_score = float(score)
            if evaluated or time.monotonic() - last_save >= self.recipe.checkpoint_seconds:
                path = save_snapshot(self.directory, self.model, optimizer, state, identity)
                if candidate_score is not None:
                    retain_candidate(self.directory, path, candidate_score)
                if is_best:
                    write_json(
                        self.directory / "best_ce.json", Pointer(snapshot=path.name).model_dump()
                    )
                    emit("best", step=state.step, role="best_ce", audio_ce_nats=state.best_ce)
                emit("checkpoint", step=state.step, path=str(path))
                prune_snapshots(self.directory)
                last_save = time.monotonic()
        path = save_snapshot(self.directory, self.model, optimizer, state, identity)
        reason = (
            f"signal {stop.number}"
            if stop.number
            else "schedule"
            if state.step == len(windows)
            else "stop_after"
        )
        report = {
            **provenance(identity),
            "step": state.step,
            "updates_this_run": state.step - started_step,
            "ended": reason,
            "best_ce": state.best_ce,
            "latest": str(path),
            "best_speech": state.best_speech,
            "phase": "completed" if state.step == len(windows) else "stopped",
        }
        write_json(self.directory / "training.stats.json", report)
        emit("end", **report)
