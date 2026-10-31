"""KabEncBench multi-task evaluation harness, scale-normalized probes, and diagnostics.

Implements the 2026 two-track evaluation protocol:
- Track A (`frozen`): Scale-normalized scalar layer mix + fixed-width probe (d_p = 256),
  sweeping a 4-point logarithmic learning-rate grid on `dev` before multi-seed `test` runs.
- Track B (`finetune`): End-to-end fine-tuning with differential head LR (5x trunk LR) and
  layer-wise decay (0.95), sweeping a 4-point logarithmic trunk LR grid on `dev`.
- Diagnostics: `diag_clitic`, `diag_para` (zero-tune layer-wise TaPaCo retrieval + isotropy),
  and `diag_spell` (zero-leakage minimal pairs).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import torch
from torch import Tensor, nn

from agbalu.bench.elo import Leaderboard, compute_leaderboard
from agbalu.bench.encbench_data import CORE_TASKS, DEFAULT_OUT_DIR
from agbalu.bench.sts import check_isotropic_collapse

Track = Literal["frozen", "finetune"]
FROZEN_LR_GRID: Final[tuple[float, ...]] = (3e-4, 1e-3, 3e-3, 1e-2)
FINETUNE_LR_GRID: Final[tuple[float, ...]] = (1e-5, 2e-5, 5e-5, 1e-4)
DEFAULT_SEEDS: Final[tuple[int, ...]] = (42, 43, 44, 45, 46)
PROBE_DIM: Final[int] = 256
IGNORE_INDEX: Final[int] = -100


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Verified architectural and parameter metadata for the 13 benchmarked encoders."""

    model_id: str
    family: str
    architecture: str
    layers: int
    hidden_size: int
    vocab_size: int
    max_positions: int
    total_params_m: float
    trunk_params_m: float


ENCODER_REGISTRY: Final[tuple[EncoderSpec, ...]] = (
    EncoderSpec(
        "agbalu/Masinissa-31M",
        "kabyle-mono",
        "roberta-preln",
        12,
        384,
        16000,
        512,
        31.12,
        24.98,
    ),
    EncoderSpec(
        "agbalu/Masinissa-73M",
        "kabyle-mono",
        "modernbert-attnres",
        20,
        512,
        16000,
        1024,
        73.32,
        65.13,
    ),
    EncoderSpec(
        "UBC-NLP/serengeti-E110",
        "pan-african",
        "electra",
        12,
        768,
        110000,
        512,
        170.14,
        85.66,
    ),
    EncoderSpec(
        "UBC-NLP/serengeti-E250",
        "pan-african",
        "electra",
        12,
        768,
        250000,
        512,
        277.66,
        85.66,
    ),
    EncoderSpec(
        "UBC-NLP/serengeti",
        "pan-african",
        "xlm-roberta",
        12,
        768,
        250004,
        514,
        278.05,
        86.04,
    ),
    EncoderSpec(
        "Davlan/afro-xlmr-base-76L_script",
        "pan-african",
        "xlm-roberta",
        12,
        768,
        270002,
        514,
        293.40,
        86.04,
    ),
    EncoderSpec(
        "Davlan/afro-xlmr-large-76L",
        "pan-african",
        "xlm-roberta",
        24,
        1024,
        250002,
        514,
        559.89,
        303.89,
    ),
    EncoderSpec(
        "jhu-clsp/mmBERT-small",
        "massively-multilingual",
        "modernbert",
        22,
        384,
        256000,
        8192,
        140.76,
        42.45,
    ),
    EncoderSpec(
        "jhu-clsp/mmBERT-base",
        "massively-multilingual",
        "modernbert",
        22,
        768,
        256000,
        8192,
        307.53,
        110.93,
    ),
    EncoderSpec(
        "FacebookAI/xlm-roberta-base",
        "global-multilingual",
        "xlm-roberta",
        12,
        768,
        250002,
        514,
        278.04,
        86.04,
    ),
    EncoderSpec(
        "FacebookAI/xlm-roberta-large",
        "global-multilingual",
        "xlm-roberta",
        24,
        1024,
        250002,
        514,
        559.89,
        303.89,
    ),
    EncoderSpec(
        "google-bert/bert-base-multilingual-cased",
        "global-multilingual",
        "bert",
        12,
        768,
        119547,
        512,
        177.85,
        86.04,
    ),
    EncoderSpec(
        "google/rembert",
        "global-multilingual",
        "rembert",
        32,
        1152,
        250300,
        512,
        575.86,
        511.78,
    ),
)

REGISTRY_BY_ID: Final[dict[str, EncoderSpec]] = {s.model_id: s for s in ENCODER_REGISTRY}


class MultiLayerBackbone(Protocol):
    """Protocol for any pretrained encoder evaluated by KabEncBench."""

    @property
    def model_id(self) -> str: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def num_layers(self) -> int: ...

    def encode_words(
        self, words: Sequence[str], *, max_pieces: int = 510
    ) -> tuple[Tensor, list[tuple[int, int]]]:
        """Return `(input_ids, [(start_subword, end_subword), ...])` per surface word."""
        ...

    def forward_layers(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """Return hidden states `[h^(0), h^(1), ..., h^(L)]`, each `[batch, seq, hidden]`."""
        ...

    def parameters(self) -> Sequence[nn.Parameter]: ...


class ScalarLayerMix(nn.Module):
    """Scale-normalized learned scalar mixture across all encoder layers (0..L).

    Normalizes the weighted layer sum with a parameter-free LayerNorm before projecting
    to the probe head so hidden-state RMS differences across Post-LN, Pre-LN, and AttnRes
    architectures cannot bias step sizes or weight decay.
    """

    def __init__(self, num_layers: int, hidden_size: int, out_dim: int = PROBE_DIM) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers + 1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, layers: Sequence[Tensor] | Tensor) -> Tensor:
        probs = torch.softmax(self.weights, dim=0)
        stacked = layers if isinstance(layers, Tensor) else torch.stack(list(layers), dim=0)
        view_shape = (probs.shape[0],) + (1,) * (stacked.ndim - 1)
        mixed = (stacked * probs.view(view_shape)).sum(dim=0)
        normalized = self.norm(mixed) * self.gamma
        projected: Tensor = self.proj(normalized)
        return projected


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """Result of evaluating one model on one task across selected LR and seeds."""

    task: str
    track: Track
    selected_lr: float
    primary_metric: float
    primary_std: float
    secondary_metrics: dict[str, float]
    seed_item_scores: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "track": self.track,
            "selected_lr": self.selected_lr,
            "primary_metric": round(self.primary_metric, 6),
            "primary_std": round(self.primary_std, 6),
            "secondary_metrics": {k: round(v, 6) for k, v in self.secondary_metrics.items()},
            "num_seeds": len(self.seed_item_scores),
        }


def read_task_split(task: str, split: str, root: Path = DEFAULT_OUT_DIR) -> list[dict[str, object]]:
    """Read a compiled JSONL split from `data/tasks/encbench/{task}/{split}.jsonl`."""
    path = root / task / f"{split}.jsonl"
    if not path.is_file():
        msg = f"missing compiled KabEncBench split: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_f1_from_pairs(
    gold: Sequence[int], pred: Sequence[int], num_classes: int, *, exclude: set[int] | None = None
) -> float:
    """Compute macro-F1 over active classes."""
    excluded = exclude or set()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for g, p in zip(gold, pred, strict=True):
        if g < 0 or g >= num_classes:
            continue
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            if 0 <= p < num_classes:
                fp[p] += 1
    f1s: list[float] = []
    for c in range(num_classes):
        if c in excluded:
            continue
        if tp[c] + fn[c] == 0 and tp[c] + fp[c] == 0:
            continue
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append((2.0 * tp[c] / denom) if denom > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def extract_iob_spans(labels: Sequence[str]) -> set[tuple[int, int, str]]:
    """Extract `(start, end, category)` spans from an IOB2 tag sequence."""
    spans: set[tuple[int, int, str]] = set()
    start = -1
    current_tag = ""
    for idx, label in enumerate(labels):
        if label.startswith("B-"):
            if current_tag:
                spans.add((start, idx, current_tag))
            start = idx
            current_tag = label[2:]
        elif label.startswith("I-"):
            tag = label[2:]
            if current_tag != tag:
                if current_tag:
                    spans.add((start, idx, current_tag))
                start = idx
                current_tag = tag
        elif current_tag:
            spans.add((start, idx, current_tag))
            current_tag = ""
            start = -1
    if current_tag:
        spans.add((start, len(labels), current_tag))
    return spans


def strict_span_f1(
    gold_sequences: Sequence[Sequence[str]], pred_sequences: Sequence[Sequence[str]]
) -> tuple[float, dict[str, float], list[float]]:
    """Compute strict `seqeval`-equivalent span-level micro-F1, per-class F1, and sentence F1s."""
    tp_total = 0
    fp_total = 0
    fn_total = 0
    by_cat_tp: Counter[str] = Counter()
    by_cat_fp: Counter[str] = Counter()
    by_cat_fn: Counter[str] = Counter()
    item_f1s: list[float] = []

    for g_seq, p_seq in zip(gold_sequences, pred_sequences, strict=True):
        g_spans = extract_iob_spans(g_seq)
        p_spans = extract_iob_spans(p_seq)
        hits = g_spans & p_spans
        tp = len(hits)
        fp = len(p_spans - g_spans)
        fn = len(g_spans - p_spans)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        denom = 2 * tp + fp + fn
        item_f1s.append((2.0 * tp / denom) if denom > 0 else (1.0 if not g_spans else 0.0))

        for _, _, c in hits:
            by_cat_tp[c] += 1
        for _, _, c in p_spans - g_spans:
            by_cat_fp[c] += 1
        for _, _, c in g_spans - p_spans:
            by_cat_fn[c] += 1

    total_denom = 2 * tp_total + fp_total + fn_total
    micro_f1 = (2.0 * tp_total / total_denom) if total_denom > 0 else 0.0
    per_cat: dict[str, float] = {}
    for c in sorted(set(by_cat_tp) | set(by_cat_fp) | set(by_cat_fn)):
        d = 2 * by_cat_tp[c] + by_cat_fp[c] + by_cat_fn[c]
        per_cat[c] = (2.0 * by_cat_tp[c] / d) if d > 0 else 0.0
    return micro_f1, per_cat, item_f1s


MRR_CUTOFF: Final[int] = 10


def evaluate_paraphrase_retrieval(
    embeddings: Sequence[Sequence[float]], cluster_ids: Sequence[int]
) -> dict[str, float]:
    """Compute zero-tune `mAP@R`, `MRR@10`, and isotropic collapse metrics on `Diag-Para`."""
    n = len(embeddings)
    if n == 0 or len(cluster_ids) != n:
        return {"map_at_r": 0.0, "mrr_at_10": 0.0, "isotropic_mean_cos": 0.0}

    t_emb = torch.tensor(embeddings, dtype=torch.float32)
    centered = t_emb - t_emb.mean(dim=0, keepdim=True)
    norms = centered.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = centered / norms
    sim_matrix = torch.matmul(normed, normed.T)
    sim_matrix.fill_diagonal_(-2.0)

    cluster_counts = Counter(cluster_ids)
    ap_scores: list[float] = []
    rr_scores: list[float] = []

    for i in range(n):
        c_i = cluster_ids[i]
        r_total = cluster_counts[c_i] - 1
        if r_total <= 0:
            continue
        ranked = torch.argsort(sim_matrix[i], descending=True).tolist()
        hits = 0
        sum_prec = 0.0
        first_rank = 0
        for rank_pos, j in enumerate(ranked[: max(r_total, MRR_CUTOFF)], start=1):
            if cluster_ids[j] == c_i:
                if first_rank == 0 and rank_pos <= MRR_CUTOFF:
                    first_rank = rank_pos
                if rank_pos <= r_total:
                    hits += 1
                    sum_prec += hits / rank_pos
        ap_scores.append(sum_prec / r_total)
        rr_scores.append((1.0 / first_rank) if first_rank > 0 else 0.0)

    iso = check_isotropic_collapse(embeddings, pairs_sample=2000, seed=42)
    return {
        "map_at_r": sum(ap_scores) / len(ap_scores) if ap_scores else 0.0,
        "mrr_at_10": sum(rr_scores) / len(rr_scores) if rr_scores else 0.0,
        "isotropic_mean_cos": iso.mean_cosine,
        "isotropic_std_cos": iso.std_cosine,
        "collapsed": 1.0 if iso.collapsed else 0.0,
    }


def build_track_leaderboard(
    track: Track,
    results_by_model: Mapping[str, Mapping[str, TaskRunResult]],
    *,
    anchor_model: str = "FacebookAI/xlm-roberta-base",
    num_bootstraps: int = 2000,
) -> Leaderboard:
    """Aggregate `TaskRunResult` objects across all models into a Bradley-Terry Elo Leaderboard."""
    model_task_items: dict[str, dict[str, Sequence[Sequence[float]]]] = {}
    metadata: dict[str, tuple[float, float]] = {}

    for model_id, task_map in results_by_model.items():
        model_task_items[model_id] = {
            t: res.seed_item_scores for t, res in task_map.items() if t in CORE_TASKS
        }
        if model_id in REGISTRY_BY_ID:
            spec = REGISTRY_BY_ID[model_id]
            metadata[model_id] = (spec.total_params_m, spec.trunk_params_m)

    return compute_leaderboard(
        track,
        model_task_items,
        anchor_model=anchor_model,
        num_bootstraps=num_bootstraps,
        model_metadata=metadata,
    )
