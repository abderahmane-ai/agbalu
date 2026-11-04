"""Qualify native adapter execution through bounded reproducible probes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from agbalu.codec_tts.adaptation import adapter_state, disabled_adapter, restore_adapter
from agbalu.codec_tts.artifacts import atomic_binary, provenance, write_json
from agbalu.codec_tts.checkpoint import RunState, restore_snapshot, save_snapshot, verify_snapshot
from agbalu.codec_tts.configuration import SMOKE_PARTS
from agbalu.codec_tts.dataset import collate_cpu
from agbalu.codec_tts.evaluation import heldout_loss
from agbalu.codec_tts.training import TrainingRun, perform_update

PARTS = SMOKE_PARTS
SMOKE_UPDATES = 20


def interrupt_writer() -> None:
    message = "deliberate interrupted writer"
    raise InterruptedError(message)


def proof_path(run: TrainingRun, part: str) -> Path:
    return Path("artifacts/runs/codec-tts-qualification") / run.identity / f"{part}.stats.json"


def admit_training(run: TrainingRun) -> None:
    for part in PARTS:
        path = proof_path(run, part)
        payload: object = json.loads(path.read_bytes())
        if (
            not isinstance(payload, dict)
            or payload.get("identity") != run.identity
            or payload.get("passed") is not True
        ):
            message = f"missing or incompatible {part} qualification at {path}"
            raise ValueError(message)


def replay_probe(run: TrainingRun) -> dict[str, object]:
    model, recipe = run.model, run.recipe
    device = model.audio_embedding.weight.device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe.peak_lr,
        fused=device.type == "cuda",
        weight_decay=0,
    )
    windows = tuple(run.training.windows())[:SMOKE_UPDATES]
    if len(windows) != SMOKE_UPDATES:
        message = f"resume probe needs {SMOKE_UPDATES} real update windows, got {len(windows)}"
        raise ValueError(message)
    state = RunState()
    root = run.directory / "resume-control"
    initial = save_snapshot(root, model, optimizer, state, run.identity)
    for window in windows:
        batches = [
            collate_cpu(
                [
                    run.training.layout(index)
                    for index in window[start : start + recipe.micro_examples]
                ]
            )
            for start in range(0, len(window), recipe.micro_examples)
        ]
        perform_update(model, optimizer, batches, recipe)
    expected = adapter_state(model)
    restore_snapshot(initial, model, optimizer, run.identity)
    split = SMOKE_UPDATES // 2
    interrupted: Path | None = None
    for step, window in enumerate(windows, start=1):
        batches = [
            collate_cpu(
                [
                    run.training.layout(index)
                    for index in window[start : start + recipe.micro_examples]
                ]
            )
            for start in range(0, len(window), recipe.micro_examples)
        ]
        perform_update(model, optimizer, batches, recipe)
        state.step = state.cursor = step
        if step == split:
            interrupted = save_snapshot(root, model, optimizer, state, run.identity)
            restore_snapshot(interrupted, model, optimizer, run.identity)
    differences = []
    tolerance = 2e-3 if device.type == "cuda" else 1e-6
    for name, value in adapter_state(model).items():
        torch.testing.assert_close(value, expected[name], atol=tolerance, rtol=tolerance)
        differences.append(float((value - expected[name]).abs().max().item()))
    return {
        "updates_per_trajectory": SMOKE_UPDATES,
        "interrupted_snapshot": str(interrupted),
        "max_adapter_difference": max(differences),
        "real_data": True,
    }


def failure_probe(run: TrainingRun) -> dict[str, object]:
    model, recipe = run.model, run.recipe
    saved = adapter_state(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=recipe.peak_lr
    )
    path = save_snapshot(
        run.directory / "failure-control", model, optimizer, RunState(), run.identity
    )
    rejected_identity = rejected_corruption = rejected_nonfinite = False
    try:
        verify_snapshot(path, "wrong identity")
    except ValueError:
        rejected_identity = True
    staged = run.directory / "failure-control" / "atomic.txt"
    write_json(staged, {"old": True})
    try:
        with atomic_binary(staged) as stream:
            stream.write(b"partial")
            interrupt_writer()
    except InterruptedError:
        pass
    if json.loads(staged.read_bytes()) != {"old": True}:
        message = "interrupted writer replaced a committed artifact"
        raise AssertionError(message)
    (path / "training.pt").write_bytes(b"deliberate corruption")
    try:
        verify_snapshot(path, run.identity)
    except ValueError:
        rejected_corruption = True
    try:
        with torch.no_grad():
            parameter = next(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            parameter.fill_(float("nan"))
        perform_update(model, optimizer, [collate_cpu([run.training.layout(0)])], recipe)
    except FloatingPointError:
        rejected_nonfinite = True
    finally:
        restore_adapter(model, saved)
    if not (rejected_identity and rejected_corruption and rejected_nonfinite):
        message = "a deliberate training failure was not detected"
        raise AssertionError(message)
    return {
        "identity_control": True,
        "corruption_control": True,
        "nonfinite_control": True,
        "interrupted_write_control": True,
    }


def run_smoke(run: TrainingRun, part: str) -> dict[str, object]:
    if part not in PARTS:
        message = f"unknown codec smoke part={part!r}"
        raise ValueError(message)
    if run.model.audio_embedding.weight.device.type != "cuda":
        message = "production smoke qualification requires the actual CUDA target"
        raise ValueError(message)
    started = time.monotonic()
    if part == "throughput":
        before = time.monotonic()
        state = run.train(stop_after=SMOKE_UPDATES)
        report = json.loads((run.directory / "training.stats.json").read_bytes())
        if report.get("updates_this_run") != SMOKE_UPDATES or state.step != SMOKE_UPDATES:
            message = "twenty-update smoke must be a fresh run with twenty completed updates"
            raise ValueError(message)
        measured: dict[str, object] = {
            "updates": state.step,
            "elapsed_s": time.monotonic() - before,
        }
    elif part == "parity":
        # Native Transformers owns the Qwen3 layers; test our multimodal wrapper independently.
        longest = max(range(len(run.training)), key=run.training.positions)
        layout = run.training.layout(longest)
        text, codes, audio, valid, _ = collate_cpu([layout])
        device = run.model.audio_embedding.weight.device
        text, codes, audio, valid = (value.to(device) for value in (text, codes, audio, valid))
        run.model.train(False)
        with torch.no_grad(), disabled_adapter(run.model):
            expected = run.model.hidden(text, codes, audio, valid)
        with torch.no_grad():
            actual = run.model.hidden(text, codes, audio, valid)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        initial_adapter = adapter_state(run.model)
        run.model.train()
        if run.recipe.activation_checkpointing:
            run.model.enable_activation_checkpointing()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in run.model.parameters() if parameter.requires_grad],
            lr=run.recipe.peak_lr,
            fused=True,
            weight_decay=0,
        )
        longest_metrics = perform_update(run.model, optimizer, [collate_cpu([layout])], run.recipe)
        restore_adapter(run.model, initial_adapter)
        measured = {
            "zero_adapter_native_parity": True,
            "longest_positions": layout.positions,
            "longest_backward_update": longest_metrics,
            "parameters": sum(p.numel() for p in run.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in run.model.parameters() if p.requires_grad
            ),
        }
    elif part == "resume":
        measured = replay_probe(run)
    elif part == "evaluate":
        measured = {
            "teacher_forced": heldout_loss(
                run.model, run.validation, run.recipe.evaluation_examples
            ),
            "speech": run.speech_evaluation(run.model, run.directory / "evaluation"),
        }
    else:
        measured = failure_probe(run)
    free, total = torch.cuda.mem_get_info()
    if free < total * run.recipe.memory_reserve_fraction:
        message = f"GPU reserve failed after smoke: free_bytes={free}, total_bytes={total}"
        raise MemoryError(message)
    report = {
        **provenance(run.identity),
        "identity": run.identity,
        "part": part,
        "passed": True,
        "elapsed_s": time.monotonic() - started,
        "memory_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "memory_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
        "gpu_utilization": None,
        "gpu_utilization_reason": "not sampled",
        "measurements": measured,
    }
    write_json(proof_path(run, part), report)
    write_json(run.directory / f"smoke-{part}.stats.json", report)
    return report
