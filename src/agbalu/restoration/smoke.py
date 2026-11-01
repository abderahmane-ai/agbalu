"""Verify GPU gradients, complete optimizer updates and deterministic recovery."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from agbalu.encoder.corpus import write_json
from agbalu.llm.progress import Heartbeat, stage
from agbalu.restoration import checkpoint, pilot
from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.dataset import collate, collate_update
from agbalu.restoration.model import Restorer
from agbalu.restoration.objective import loss_sums, target_counts
from agbalu.restoration.training import Trainer, smoke_identity, update

RESUME_TOLERANCE = 1e-6
SMOKE_UPDATES = 20


def gradient_probe(trainer: Trainer) -> dict[str, object]:
    indices = np.random.default_rng(trainer.settings.seed).choice(
        len(trainer.offline), size=min(2, len(trainer.offline)), replace=False
    )
    batch = collate(
        [trainer.offline[int(index)] for index in indices],
        trainer.settings.geometry.context_tokens,
    ).to(trainer.device)
    trainer.model.train()
    scores = trainer.model.forward(batch)
    balanced = trainer.settings.objective == "balanced"
    counts = target_counts(batch, balanced=balanced).clamp(min=1)
    losses = loss_sums(
        scores,
        batch,
        denominators=counts,
        balanced=balanced,
    )
    terms = [
        losses.punctuation if balanced else losses.punctuation / counts[0],
        losses.casing if balanced else losses.casing / counts[1],
        losses.structure if balanced else losses.structure / counts[2],
    ]
    gradients: list[Tensor] = []
    for term in terms:
        gradient = None
        if term.requires_grad:
            (gradient,) = torch.autograd.grad(
                term, trainer.model.encoder.embedding.weight, retain_graph=True, allow_unused=True
            )
        gradients.append(
            torch.zeros_like(trainer.model.encoder.embedding.weight).flatten()
            if gradient is None
            else gradient.detach().float().flatten()
        )
    norms = [float(value.norm()) for value in gradients]
    cosines = [
        [
            float(torch.nn.functional.cosine_similarity(left, right, dim=0))
            if norms[i] > 0 and norms[j] > 0
            else None
            for j, right in enumerate(gradients)
        ]
        for i, left in enumerate(gradients)
    ]
    if not all(torch.isfinite(value).all() for value in gradients):
        msg = "fixed restoration probe produced non-finite gradients"
        raise FloatingPointError(msg)
    return {
        "loss_terms": ["punctuation", "casing", "structure"],
        "gradient_norms": norms,
        "gradient_cosines": cosines,
        "targets": counts.tolist(),
        "window_indices": indices.tolist(),
        "seed": trainer.settings.seed,
    }


def recovery(trainer: Trainer) -> dict[str, object]:
    model = trainer.model
    indices = np.random.default_rng(trainer.settings.seed).choice(
        len(trainer.offline), size=trainer.settings.batch_windows, replace=True
    )
    batches = collate_update(
        [trainer.offline[int(index)] for index in indices],
        trainer.settings.geometry.context_tokens,
        trainer.settings.micro_windows,
    )
    with tempfile.TemporaryDirectory(dir=trainer.directory) as temporary:
        directory = Path(temporary)
        state = checkpoint.State(step=trainer.state.step)
        saved = checkpoint.save(directory, model, trainer.optimizer, state, trainer.identity)
        update(model, trainer.optimizer, batches, trainer.settings, trainer.device)
        expected = {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        }
        checkpoint.load(saved, model, trainer.optimizer, trainer.identity)
        update(model, trainer.optimizer, batches, trainer.settings, trainer.device)
        error = max(
            float((parameter.detach().cpu() - expected[name]).abs().max())
            for name, parameter in model.named_parameters()
        )
        if error > RESUME_TOLERANCE:
            msg = f"continuous/resumed parameter difference {error} exceeds 1e-6"
            raise ValueError(msg)
        checkpoint.load(saved, model, trainer.optimizer, trainer.identity)
        with (saved / "training.pt").open("ab") as handle:
            handle.write(b"corrupt-checksum-positive-control")
        detected = False
        try:
            checkpoint.latest(directory)
        except ValueError:
            detected = True
        if not detected:
            msg = "checkpoint corruption control did not fire"
            raise ValueError(msg)
    return {"passed": True, "maximum_parameter_difference": error, "corruption_detected": detected}


def parity(trainer: Trainer) -> dict[str, object]:
    geometry = trainer.settings.geometry.model_copy(update={"dropout": 0.0, "backend": "reference"})
    reference = (
        Restorer.from_settings(
            trainer.settings.model_copy(update={"geometry": geometry}), trainer.model.templates
        )
        .to(trainer.device)
        .eval()
    )
    reference.load_state_dict(trainer.model.state_dict(), strict=True)
    fast = (
        Restorer.from_settings(
            trainer.settings.model_copy(
                update={"geometry": geometry.model_copy(update={"backend": "flex"})}
            ),
            trainer.model.templates,
        )
        .to(trainer.device)
        .eval()
    )
    fast.load_state_dict(reference.state_dict(), strict=True)
    batch = collate(
        [trainer.offline[index] for index in range(min(2, len(trainer.offline)))],
        trainer.settings.geometry.context_tokens,
    ).to(trainer.device)
    first, second = reference.forward(batch), fast.forward(batch)
    errors: list[float] = []
    for left, right in zip(
        (first.presence, first.punctuation, first.casing, first.structure),
        (second.presence, second.punctuation, second.casing, second.structure),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=2e-3, rtol=2e-3)
        errors.append(float((left - right).detach().abs().max()))
    left_loss = loss_sums(first, batch).normalized(target_counts(batch))
    right_loss = loss_sums(second, batch).normalized(target_counts(batch))
    torch.autograd.backward(left_loss)
    torch.autograd.backward(right_loss)
    for (name, left), (_, right) in zip(
        reference.named_parameters(), fast.named_parameters(), strict=True
    ):
        if left.grad is not None and right.grad is not None:

            def mismatch(message: str, parameter: str = name) -> str:
                return f"{parameter}: {message}"

            torch.testing.assert_close(
                left.grad,
                right.grad,
                atol=3e-3,
                rtol=3e-3,
                msg=mismatch,
            )
    neighbor_difference = None
    if batch.words.size(0) > 1:
        valid = batch.case_spans[1, :, 1] > batch.case_spans[1, :, 0]
        baseline = second.casing[1, valid].detach().clone()
        batch.ids.masked_fill_(batch.documents == 0, 6)
        with torch.no_grad():
            changed = fast.forward(batch).casing[1, valid]
        torch.testing.assert_close(baseline, changed, atol=2e-5, rtol=2e-5)
        neighbor_difference = float((baseline - changed).abs().max())
    return {
        "passed": True,
        "maximum_output_difference": max(errors),
        "neighbor_maximum_difference": neighbor_difference,
    }


def run(
    settings: Settings, locations: Locations, name: str, part: str, device: torch.device
) -> dict[str, object]:
    directory = locations.runs / name
    if part == "throughput" and directory.exists():
        for path in directory.iterdir():
            if path.name != "log.txt":
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    trainer = Trainer(settings, locations, name, device)
    with stage(f"restoration smoke {part}"), Heartbeat(f"restoration smoke {part}"):
        if part == "assessment":
            result = pilot.assess(trainer)
        elif part == "selection":
            result = pilot.select(trainer)
        elif part == "parity":
            result = parity(trainer)
        elif part in ("resume", "failure"):
            result = recovery(trainer)
        else:
            gradient_report = gradient_probe(trainer)
            trainer.optimizer.zero_grad(set_to_none=True)
            result = trainer.run(stop_after=SMOKE_UPDATES, smoke=True)
            if int(str(result["steps_this_run"])) != SMOKE_UPDATES:
                msg = f"smoke needs 20 new optimizer updates: {result}"
                raise ValueError(msg)
            result.update(
                {
                    "passed": True,
                    "gradients": gradient_report,
                    "recovery": recovery(trainer),
                    "identity": smoke_identity(settings, locations),
                    "gpu_utilization": None,
                    "projected_training_seconds": float(str(result["step_seconds_mean"]))
                    * trainer.total_steps,
                }
            )
            write_json(locations.data / "smoke.stats.json", result)
            write_json(trainer.directory / "throughput.stats.json", result)
            if part == "pilot":
                result = pilot.run(trainer)
    write_json(trainer.directory / f"{part}.stats.json", result)
    return result
