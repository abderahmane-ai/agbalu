"""Verify accelerated gradients, optimizer recovery, and measured training throughput."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Literal

import torch

from agbalu.encoder import checkpoint
from agbalu.encoder.attention import make_layout
from agbalu.encoder.configuration import SMOKE_UPDATES, Geometry, Locations, Recipe
from agbalu.encoder.corpus import write_json
from agbalu.encoder.model import Encoder
from agbalu.encoder.training import Trainer, optimizer_for, smoke_identity
from agbalu.llm.progress import Heartbeat, progress, stage

Part = Literal["parity", "resume", "throughput", "evaluate", "failure"]


def compare_gradient(
    reference: torch.Tensor, observed: torch.Tensor, high_precision: torch.Tensor | None
) -> tuple[float, float]:
    error = float((reference - observed).float().norm())
    scale = float(reference.float().norm())
    noise = 0.0 if high_precision is None else float((reference - high_precision).float().norm())
    # Two reduced-precision computations may differ by their combined rounding error.
    allowance = scale * 3e-4 + 1e-6 if high_precision is None else 2 * noise + scale * 1e-5 + 1e-6
    if not math.isfinite(error) or error > allowance:
        message = (
            f"gradient L2 error {error} exceeds {allowance}; "
            f"reference norm={scale}, precision noise={noise}"
        )
        raise ValueError(message)
    return error, noise


def compare_model_gradients(
    exact: Encoder,
    fast: Encoder,
    fp32: dict[str, torch.Tensor],
    *,
    is_bf16: bool,
    per_parameter: bool,
) -> tuple[float, float]:
    device = next(exact.parameters()).device
    total_error = torch.zeros((), device=device)
    total_noise = torch.zeros((), device=device)
    total_scale = torch.zeros((), device=device)
    maximum = 0.0
    for (name, left), (_, right) in zip(
        exact.named_parameters(), fast.named_parameters(), strict=True
    ):
        if left.grad is None or right.grad is None:
            message = f"missing gradient in reference/accelerated path: {name}"
            raise ValueError(message)
        difference = (left.grad - right.grad).float()
        noise = float((left.grad - fp32[name]).float().norm()) if is_bf16 else 0.0
        if per_parameter:
            try:
                compare_gradient(left.grad, right.grad, fp32.get(name))
            except ValueError as failure:
                message = f"bf16={is_bf16} {name}: {failure}"
                raise ValueError(message) from failure
        total_error += difference.square().sum()
        total_noise += noise**2
        total_scale += left.grad.float().square().sum()
        if not is_bf16:
            fp32[name] = left.grad.detach().clone()
        maximum = max(maximum, float(difference.abs().max()))
    error, noise, scale = torch.stack(
        (total_error.sqrt(), total_noise.sqrt(), total_scale.sqrt())
    ).tolist()
    allowance = 2 * noise + scale * 1e-5 + 1e-6 if is_bf16 else scale * 3e-4 + 1e-6
    if not math.isfinite(error) or error > allowance:
        message = f"bf16={is_bf16} complete gradient error {error} exceeds {allowance}"
        raise ValueError(message)
    progress("parity_complete_gradient", bf16=is_bf16, error_l2=error, noise_l2=noise)
    return maximum, noise


def model_parity(
    geometry: Geometry, device: torch.device, *, per_parameter_bf16: bool
) -> dict[str, object]:
    if device.type != "cuda":
        message = "accelerated attention parity requires the actual CUDA target"
        raise ValueError(message)
    torch.manual_seed(7)
    exact = Encoder(geometry.model_copy(update={"backend": "reference", "dropout": 0.0})).to(device)
    fast = Encoder(geometry.model_copy(update={"backend": "flex", "dropout": 0.0})).to(device)
    fast.load_state_dict(exact.state_dict())
    ids = torch.randint(5, geometry.vocab_size, (256,), device=device)
    documents = torch.arange(2, device=device).repeat_interleave(128)
    positions = torch.arange(128, device=device).repeat(2)
    valid = torch.ones(256, dtype=torch.bool, device=device)
    labels = ids.clone()
    labels[1::2] = -100
    errors: list[float] = []
    fp32: dict[str, torch.Tensor] = {}
    precision_noise: list[float] = []
    for dtype in (torch.float32, torch.bfloat16):
        progress("parity_precision", precision=str(dtype))
        exact.zero_grad(set_to_none=True)
        fast.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            expected, _ = exact.forward(
                ids, make_layout(documents, positions, valid, flex=False), labels
            )
            observed, _ = fast.forward(
                ids, make_layout(documents, positions, valid, flex=True), labels
            )
        torch.autograd.backward(expected)
        torch.autograd.backward(observed)
        torch.testing.assert_close(expected, observed, atol=0.02, rtol=0.002)
        error, noise = compare_model_gradients(
            exact,
            fast,
            fp32,
            is_bf16=dtype == torch.bfloat16,
            per_parameter=dtype == torch.float32 or per_parameter_bf16,
        )
        errors.append(error)
        precision_noise.append(noise)
    first = fast.contextualise_packed(ids, make_layout(documents, positions, valid, flex=True))
    changed = ids.clone()
    changed[129:] = 6
    second = fast.contextualise_packed(changed, make_layout(documents, positions, valid, flex=True))
    torch.testing.assert_close(first[:128], second[:128], atol=1e-5, rtol=1e-5)
    return {
        "passed": True,
        "parameters": geometry.parameters,
        "max_gradient_absolute_error": max(errors),
        "max_reference_bf16_gradient_noise_l2": max(precision_noise),
        "precisions": ["fp32", "bf16"],
        "document_isolation": True,
    }


def parity(geometry: Geometry, device: torch.device) -> dict[str, object]:
    # A one-layer control compares each BF16 path before depth compounds rounding differences.
    component = model_parity(
        geometry.model_copy(update={"layers": 1, "block_layers": 1}),
        device,
        per_parameter_bf16=True,
    )
    complete = model_parity(geometry, device, per_parameter_bf16=False)
    return {**complete, "single_layer_bf16": component}


def recovery(directory: Path, device: torch.device, *, corrupt: bool = False) -> dict[str, object]:
    geometry = Geometry(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        layers=2,
        block_layers=1,
        heads=2,
        dropout=0,
        backend="reference",
    )
    recipe = Recipe(geometry=geometry, require_smoke=False)
    torch.manual_seed(4)
    model = Encoder(geometry).to(device)
    optimizer = optimizer_for(model, recipe, device)
    ids = torch.tensor([2, 6, 7, 8, 3], device=device)
    layout = make_layout(
        torch.zeros_like(ids),
        torch.arange(5, device=device),
        torch.ones(5, dtype=torch.bool, device=device),
        flex=False,
    )

    def advance() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.forward(ids, layout, torch.tensor([-100, 6, 7, 8, -100], device=device))
        torch.autograd.backward(loss)
        optimizer.step()
        return float(loss.detach())

    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        root = Path(temporary)
        state = checkpoint.State(step=1)
        advance()
        saved = checkpoint.save(root, model, optimizer, state, "recovery-probe")
        expected = advance()
        weights = {name: value.detach().clone() for name, value in model.named_parameters()}
        checkpoint.restore(saved, model, optimizer, "recovery-probe")
        observed = advance()
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, weights[name], atol=1e-7, rtol=1e-6)
        if corrupt:
            (saved / "training.pt").write_bytes(b"deliberately corrupted positive control")
            try:
                checkpoint.verify_snapshot(saved)
            except ValueError:
                return {"passed": True, "corruption_detected": True}
            message = "checkpoint corruption positive control failed to fire"
            raise ValueError(message)
    return {
        "passed": True,
        "continuous_loss": expected,
        "resumed_loss": observed,
        "absolute_loss_difference": abs(expected - observed),
    }


def run(
    part: Part, recipe: Recipe, locations: Locations, name: str, device: torch.device
) -> dict[str, object]:
    directory = locations.runs / name
    directory.mkdir(parents=True, exist_ok=True)
    with stage(f"encoder smoke {part}"), Heartbeat(f"encoder smoke {part}"):
        if part == "parity":
            result = parity(recipe.geometry, device)
        elif part in {"resume", "failure"}:
            result = recovery(directory, device, corrupt=part == "failure")
        else:
            proof = recipe.model_copy(
                update={
                    "require_smoke": False,
                    "task_evaluation": False,
                    "total_tokens": recipe.batch_tokens * 20,
                }
            )
            trainer = Trainer(proof, locations, name, device)
            result = trainer.run(stop_after=20)
            if int(str(result["steps_this_run"])) < SMOKE_UPDATES:
                message = f"smoke requires 20 new optimizer updates, got {result['steps_this_run']}"
                raise ValueError(message)
            result["passed"] = True
            if part == "evaluate":
                trainer.evaluate(tasks=True)
            if part in {"throughput", "evaluate"}:
                write_json(
                    locations.data / "smoke.stats.json",
                    {
                        "identity": smoke_identity(recipe, device),
                        "run": name,
                        **result,
                    },
                )
    write_json(directory / f"{part}.stats.json", result)
    return result
