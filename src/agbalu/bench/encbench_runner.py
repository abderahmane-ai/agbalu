"""PyTorch training and evaluation runner for KabEncBench across all 13 encoders.

Supports both native AƔBALU encoders (`agbalu/Masinissa-31M`, `agbalu/Masinissa-73M`)
and Hugging Face `AutoModel` checkpoints (`XLMRoberta`, `Electra`, `ModernBert`, `Bert`,
`RemBert`), executing Track A (`frozen` scalar-mix probe) and Track B (`finetune` with
differential head LR), locking the learning rate on `dev` at seed 42, and running
multi-seed evaluation on `test`.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn

from agbalu.bench.encbench import (
    DEFAULT_SEEDS,
    FINETUNE_LR_GRID,
    FROZEN_LR_GRID,
    PROBE_DIM,
    ScalarLayerMix,
    TaskRunResult,
    Track,
    macro_f1_from_pairs,
)

NER_TAGS: Final[tuple[str, ...]] = (
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-MISC",
    "I-MISC",
)
NER_TAG_TO_ID: Final[dict[str, int]] = {t: i for i, t in enumerate(NER_TAGS)}

PUNCT_TAGS: Final[tuple[str, ...]] = ("NONE", "PERIOD", "COMMA", "QUESTION", "COLON")
PUNCT_TAG_TO_ID: Final[dict[str, int]] = {t: i for i, t in enumerate(PUNCT_TAGS)}

CASE_TAGS: Final[tuple[str, ...]] = ("LOWER", "CAP", "ALL_CAP")
CASE_TAG_TO_ID: Final[dict[str, int]] = {t: i for i, t in enumerate(CASE_TAGS)}

MORPH_ATTR_KEYS: Final[tuple[str, ...]] = (
    "Aspect",
    "Mood",
    "Person",
    "Number",
    "Gender",
    "Polarity",
    "VerbForm",
)


@dataclass(frozen=True, slots=True)
class FitSchedule:
    """Optimization schedule for one task x track configuration."""

    epochs: int
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_fraction: float = 0.10
    max_gradient: float = 1.0
    head_lr_multiplier: float = 5.0


@dataclass(frozen=True, slots=True)
class CachedClassificationData:
    """Cached per-layer pooled representations for Track A (`frozen`) classification."""

    task: str
    num_classes: int
    train_layers: Sequence[Tensor]
    train_labels: Tensor
    dev_layers: Sequence[Tensor]
    dev_labels: Tensor
    test_layers: Sequence[Tensor]
    test_labels: Tensor


class BiaffineDependencyHead(nn.Module):
    """Dozat & Manning (2017) Biaffine Dependency Parser head for `KabSyntax`."""

    def __init__(self, in_dim: int, num_rels: int, arc_dim: int = 128, rel_dim: int = 64) -> None:
        super().__init__()
        self.mlp_arc_dep = nn.Sequential(nn.Linear(in_dim, arc_dim), nn.GELU(), nn.Dropout(0.1))
        self.mlp_arc_head = nn.Sequential(nn.Linear(in_dim, arc_dim), nn.GELU(), nn.Dropout(0.1))
        self.mlp_rel_dep = nn.Sequential(nn.Linear(in_dim, rel_dim), nn.GELU(), nn.Dropout(0.1))
        self.mlp_rel_head = nn.Sequential(nn.Linear(in_dim, rel_dim), nn.GELU(), nn.Dropout(0.1))
        self.arc_weight = nn.Parameter(torch.zeros(arc_dim + 1, arc_dim))
        self.rel_linear = nn.Linear(rel_dim * 2, num_rels)
        nn.init.xavier_uniform_(self.arc_weight)

    def forward(self, word_repr: Tensor) -> tuple[Tensor, Tensor]:
        """`word_repr` has shape `[batch, num_words + 1, in_dim]` where index 0 is `<root>`."""
        arc_d = self.mlp_arc_dep(word_repr)
        arc_h = self.mlp_arc_head(word_repr)
        ones = torch.ones(
            (arc_d.shape[0], arc_d.shape[1], 1), device=arc_d.device, dtype=arc_d.dtype
        )
        arc_d_bias = torch.cat([arc_d, ones], dim=-1)
        arc_scores = torch.matmul(torch.matmul(arc_d_bias, self.arc_weight), arc_h.transpose(1, 2))
        return arc_scores, self.mlp_rel_dep(word_repr)

    def score_relations(self, rel_dep: Tensor, word_repr: Tensor, head_indices: Tensor) -> Tensor:
        """Score dependency relation labels at given `head_indices` (`[batch, num_words + 1]`)."""
        rel_h = self.mlp_rel_head(word_repr)
        safe_heads = head_indices.clamp(min=0, max=rel_h.shape[1] - 1)
        gather_idx = safe_heads.unsqueeze(-1).expand(-1, -1, rel_h.shape[-1])
        picked_h = torch.gather(rel_h, 1, gather_idx)
        rel_logits: Tensor = self.rel_linear(torch.cat([rel_dep, picked_h], dim=-1))
        return rel_logits


class SyntaxJointHead(nn.Module):
    """Joint UPOS + Core-FEATS + Biaffine Dependency head for Task 1 (`KabSyntax`)."""

    def __init__(self, in_dim: int, num_upos: int, num_feats: int, num_rels: int) -> None:
        super().__init__()
        self.root_emb = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        self.upos_proj = nn.Linear(in_dim, num_upos)
        self.feats_proj = nn.Linear(in_dim, num_feats)
        self.dep_head = BiaffineDependencyHead(in_dim, num_rels)

    def forward(self, word_repr: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute UPOS logits, FEATS logits, Biaffine arc scores, and relation dep embeddings."""
        upos_logits = self.upos_proj(word_repr)
        feats_logits = self.feats_proj(word_repr)
        root = self.root_emb.expand(word_repr.shape[0], 1, -1)
        with_root = torch.cat([root, word_repr], dim=1)
        arc_scores, rel_dep = self.dep_head(with_root)
        return upos_logits, feats_logits, arc_scores, rel_dep


class FactoredMorphHead(nn.Module):
    """Seven-attribute factored morphological classification head for Task 2 (`KabMorph`)."""

    def __init__(self, in_dim: int, attr_sizes: Mapping[str, int]) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({k: nn.Linear(in_dim, n) for k, n in attr_sizes.items()})

    def forward(self, pooled: Tensor) -> dict[str, Tensor]:
        h = self.mlp(pooled)
        return {k: head(h) for k, head in self.heads.items()}


class DualPunctCaseHead(nn.Module):
    """Joint punctuation (5 classes) and truecasing (3 classes) head for Task 6 (`KabPunct`)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.punct_proj = nn.Linear(in_dim, len(PUNCT_TAGS))
        self.case_proj = nn.Linear(in_dim, len(CASE_TAGS))

    def forward(self, word_repr: Tensor) -> tuple[Tensor, Tensor]:
        h = self.dropout(word_repr)
        return self.punct_proj(h), self.case_proj(h)


def _cosine_schedule(warmup_frac: float, total_steps: int) -> Callable[[int], float]:
    warmup = max(1, int(total_steps * warmup_fraction)) if (warmup_fraction := warmup_frac) else 1

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return factor


def select_lr_and_run_seeds(
    task: str,
    track: Track,
    fit_and_eval_fn: Callable[[float, int], tuple[float, float, dict[str, float], list[float]]],
    *,
    lr_grid: Sequence[float] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> TaskRunResult:
    """Sweep `lr_grid` on `dev` at `seeds[0]`, then evaluate `test` across all `seeds`."""
    grid = lr_grid or (FROZEN_LR_GRID if track == "frozen" else FINETUNE_LR_GRID)
    best_lr = grid[0]
    best_dev = -1.0
    first_seed = seeds[0]
    first_result: tuple[float, float, dict[str, float], list[float]] | None = None

    for candidate_lr in grid:
        dev_score, test_score, sec_metrics, item_scores = fit_and_eval_fn(candidate_lr, first_seed)
        if dev_score > best_dev or first_result is None:
            best_dev = dev_score
            best_lr = candidate_lr
            first_result = (dev_score, test_score, sec_metrics, item_scores)

    if first_result is None:
        msg = "empty lr_grid in select_lr_and_run_seeds"
        raise ValueError(msg)
    seed_primary: list[float] = [first_result[1]]
    seed_sec: list[dict[str, float]] = [first_result[2]]
    seed_items: list[tuple[float, ...]] = [tuple(first_result[3])]

    for seed in seeds[1:]:
        _, t_score, s_metrics, i_scores = fit_and_eval_fn(best_lr, seed)
        seed_primary.append(t_score)
        seed_sec.append(s_metrics)
        seed_items.append(tuple(i_scores))

    mean_primary = sum(seed_primary) / len(seed_primary)
    var_primary = sum((x - mean_primary) ** 2 for x in seed_primary) / max(1, len(seed_primary))
    avg_sec: dict[str, float] = {"selected_dev_score": best_dev}
    for k in seed_sec[0]:
        avg_sec[k] = sum(s[k] for s in seed_sec) / len(seed_sec)

    return TaskRunResult(
        task=task,
        track=track,
        selected_lr=best_lr,
        primary_metric=mean_primary,
        primary_std=math.sqrt(var_primary),
        secondary_metrics=avg_sec,
        seed_item_scores=tuple(seed_items),
    )


def evaluate_cached_classification(
    data: CachedClassificationData,
    *,
    lr_grid: Sequence[float] = FROZEN_LR_GRID,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    schedule: FitSchedule | None = None,
    device: torch.device | None = None,
) -> TaskRunResult:
    """Fast Track A (`frozen`) evaluator over cached per-layer pooled representations."""
    sched_cfg = schedule or FitSchedule(epochs=20, batch_size=64)
    dev_target = device or torch.device("cpu")
    tr_stack = torch.stack(list(data.train_layers), dim=0).to(dev_target)
    dv_stack = torch.stack(list(data.dev_layers), dim=0).to(dev_target)
    te_stack = torch.stack(list(data.test_layers), dim=0).to(dev_target)
    tr_y = data.train_labels.to(dev_target)
    dv_y, te_y = data.dev_labels.tolist(), data.test_labels.tolist()

    def _fit_one(lr: float, seed: int) -> tuple[float, float, dict[str, float], list[float]]:
        torch.manual_seed(seed)
        mix = ScalarLayerMix(tr_stack.shape[0] - 1, tr_stack.shape[-1], PROBE_DIM).to(dev_target)
        head = nn.Sequential(nn.GELU(), nn.Dropout(0.1), nn.Linear(PROBE_DIM, data.num_classes)).to(
            dev_target
        )
        params = list(mix.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=sched_cfg.weight_decay)
        n_train = tr_y.shape[0]
        steps = max(1, math.ceil(n_train / sched_cfg.batch_size) * sched_cfg.epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, _cosine_schedule(sched_cfg.warmup_fraction, steps)
        )
        crit = nn.CrossEntropyLoss()
        rng = random.Random(seed)  # noqa: S311

        best_dev = -1.0
        best_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None
        for _ in range(sched_cfg.epochs):
            mix.train()
            head.train()
            order = list(range(n_train))
            rng.shuffle(order)
            for start in range(0, n_train, sched_cfg.batch_size):
                idx = torch.tensor(
                    order[start : start + sched_cfg.batch_size],
                    dtype=torch.long,
                    device=dev_target,
                )
                loss = crit(head(mix(tr_stack[:, idx])), tr_y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, sched_cfg.max_gradient)
                opt.step()
                sched.step()

            mix.eval()
            head.eval()
            with torch.no_grad():
                dv_pred = head(mix(dv_stack)).argmax(dim=-1).tolist()
            if (dv_f1 := macro_f1_from_pairs(dv_y, dv_pred, data.num_classes)) > best_dev:
                best_dev = dv_f1
                best_state = (copy.deepcopy(mix.state_dict()), copy.deepcopy(head.state_dict()))

        if best_state is not None:
            mix.load_state_dict(best_state[0])
            head.load_state_dict(best_state[1])
        mix.eval()
        head.eval()
        with torch.no_grad():
            te_pred = head(mix(te_stack)).argmax(dim=-1).tolist()
        te_f1 = macro_f1_from_pairs(te_y, te_pred, data.num_classes)
        items = [1.0 if g == p else 0.0 for g, p in zip(te_y, te_pred, strict=True)]
        return best_dev, te_f1, {"accuracy": sum(items) / len(items), "macro_f1": te_f1}, items

    return select_lr_and_run_seeds(data.task, "frozen", _fit_one, lr_grid=lr_grid, seeds=seeds)
