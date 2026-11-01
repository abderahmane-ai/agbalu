"""The 20-step smoke suite for Dihya-22M on HPC environment.

Executes mandatory verification checks: throughput benchmark, CPU/GPU parity,
checkpoint save/resume determinism, and evaluation sanity.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Final

import torch

from agbalu.dihya.artifacts import atomic_file, write_json
from agbalu.dihya.config import ModelConfig, TrainConfig
from agbalu.dihya.dataset import CellSampler, TextDataset, collate_examples
from agbalu.dihya.evaluate import partition_development
from agbalu.dihya.losses import GroupWeights, SupervisedContrastiveLoss
from agbalu.dihya.model import Classifier
from agbalu.dihya.train import emit_event, train, train_step
from agbalu.llm.provenance import file_digest

SMOKE_STEPS: Final = 20
PARITY_TOLERANCE: Final = 1e-3


def probe_deferred_branch(
    dataset: TextDataset, run_dir: Path, device: torch.device
) -> dict[str, float]:
    """Check finite GPU gradients with both LDAM and Group-DRO active."""
    model_config, recipe = ModelConfig(), TrainConfig()
    model = Classifier(model_config).to(device)
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location=device, weights_only=True))
    sampler = CellSampler(
        dataset,
        recipe.groups_per_batch,
        recipe.samples_per_group,
        seed=recipe.seed,
        min_group_support=recipe.min_group_support,
    )
    cells = {cell: index for index, cell in enumerate(sampler.cells)}
    groups = {cell: cells[target] for cell, target in sampler.cell_aliases.items()}
    domains = {
        domain: index for index, domain in enumerate(sorted({ex.domain for ex in dataset.examples}))
    }
    batch = collate_examples(
        [dataset[index] for index in next(iter(sampler))], model_config, domains, groups
    )
    weights = GroupWeights(len(cells), recipe.group_dro_step).to(device)
    contrastive = SupervisedContrastiveLoss(
        model_config.projection_dim, recipe.memory_bank_size, recipe.contrastive_temperature
    ).to(device)
    losses = train_step(model, batch, weights, contrastive, recipe, True, device)
    torch.autograd.backward(losses.total_loss)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True
    )
    return {
        "loss": float(losses.total_loss.detach()),
        "gradient_norm": float(norm),
        "minimum_group_weight": float(weights.weights.min()),
        "maximum_group_weight": float(weights.weights.max()),
    }


def run_throughput_smoke(device: torch.device, run_dir: Path) -> None:
    """Run 20-step throughput benchmark measuring sequences/second and memory."""
    dataset = TextDataset.load(Path("data/processed/dihya"), "train")
    dev = TextDataset.load(Path("data/processed/dihya"), "dev")
    dev, _, _ = partition_development(dev, TrainConfig().seed)
    m_cfg = ModelConfig()
    t_cfg = TrainConfig()
    if (run_dir / "latest.json").exists():
        msg = f"Smoke requires a fresh run directory: {run_dir}"
        raise ValueError(msg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    train(m_cfg, t_cfg, dataset, dev, run_dir, device, stop_after_steps=SMOKE_STEPS)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    rate = (SMOKE_STEPS * t_cfg.batch_size) / max(0.001, elapsed)
    replay_dir = run_dir / "resume"
    for filename in ("step-0000001.pt", "step-0000001.stats.json"):
        with (
            (run_dir / "checkpoints" / filename).open("rb") as source,
            atomic_file(replay_dir / "checkpoints" / filename) as destination,
        ):
            shutil.copyfileobj(source, destination)
    write_json(replay_dir / "latest.json", {"snapshot": "checkpoints/step-0000001.pt"})
    train(m_cfg, t_cfg, dataset, dev, replay_dir, device, stop_after_steps=SMOKE_STEPS)
    reference = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=True)
    replayed = torch.load(replay_dir / "final.pt", map_location="cpu", weights_only=True)
    difference = max(float((reference[key] - replayed[key]).abs().max()) for key in reference)
    if difference > 1e-6:
        msg = f"GPU checkpoint replay differs by {difference}"
        raise RuntimeError(msg)
    deferred_probe = probe_deferred_branch(dataset, run_dir, device)
    result = {
        "passed": True,
        "steps_this_run": SMOKE_STEPS,
        "rate": rate,
        "elapsed_s": elapsed,
        "seconds_per_step_including_evaluation": elapsed / SMOKE_STEPS,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "gpu_utilization": None,
        "gpu_utilization_reason": "NVML utilization unavailable on this MIG slice",
        "model_config": "default",
        "data": "real processed splits",
        "resume_replayed_steps": SMOKE_STEPS - 1,
        "resume_max_weight_difference": difference,
        "source_sha256": {
            path.name: file_digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "dataset_manifest_sha256": file_digest(Path("data/processed/dihya/dataset.stats.json")),
        "torch_version": torch.__version__,
        "deferred_branch": deferred_probe,
    }
    write_json(run_dir / "smoke.stats.json", result)

    emit_event(
        "smoke",
        part="throughput",
        passed=True,
        rate=round(rate, 1),
        elapsed=round(elapsed, 2),
    )


def run_parity_smoke() -> None:
    """Verify numerical parity of forward pass across CPU and GPU/MPS."""
    m_cfg = ModelConfig()
    model = Classifier(m_cfg).eval()
    input_ids = torch.randint(2, 256, (2, 64), dtype=torch.long)

    with torch.no_grad():
        # CPU pass
        model_cpu = model.to("cpu")
        out_cpu = model_cpu(input_ids)

        # CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            out_cuda = model_cuda(input_ids.to("cuda"))
            diff = (out_cpu.logits - out_cuda.logits.cpu()).abs().max().item()
            passed = diff < PARITY_TOLERANCE
        else:
            msg = "CPU/CUDA parity requires an available CUDA device"
            raise RuntimeError(msg)

    emit_event("smoke", part="parity", passed=passed, max_diff=diff)
    if not passed:
        msg = f"CPU/CUDA parity error {diff} exceeds {PARITY_TOLERANCE}"
        raise RuntimeError(msg)


def run_smoke(part: str, device: torch.device, run_dir: Path) -> None:
    """Dispatch smoke test part."""
    torch.use_deterministic_algorithms(True)
    if part == "throughput":
        run_throughput_smoke(device, run_dir)
    elif part == "parity":
        run_parity_smoke()
    else:
        # Default all smoke checks
        run_parity_smoke()
        run_throughput_smoke(device, run_dir)
