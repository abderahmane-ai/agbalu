"""Order-independent Global MLE Bradley-Terry Elo and paired cluster-bootstrap ranking.

Sequential online Elo depends on match order and overstates differences that sit inside
seed and item-sampling variance. This module fits a global Maximum-Likelihood
Bradley-Terry model over paired cluster-bootstrap replicates (sampling identical test
item indices across every model in each replicate), anchored to a reference encoder at
1000.0, and computes 95% bootstrap confidence intervals, pairwise superiority
probabilities P(m > n), and statistical rank-sets (tiers).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

DEFAULT_ANCHOR_ELO: Final[float] = 1000.0
ELO_SCALE: Final[float] = 400.0 / math.log(10.0)
DEFAULT_DEADZONE: Final[float] = 0.001
DEFAULT_REGULARIZATION: Final[float] = 1e-4
DEFAULT_BOOTSTRAPS: Final[int] = 2000
TIER_CONFIDENCE: Final[float] = 0.95
HESSIAN_EPS: Final[float] = 1e-12
DEFAULT_TOL: Final[float] = 1e-8
DEFAULT_MAX_ITER: Final[int] = 400


class EloError(Exception):
    """Raised when a Bradley-Terry leaderboard cannot be estimated."""


@dataclass(frozen=True, slots=True)
class ModelRating:
    """Global Bradley-Terry Elo rating, bootstrap interval, and statistical rank-set."""

    model: str
    elo_median: float
    elo_mean: float
    ci_low: float
    ci_high: float
    rank: int
    tier: int
    mean_task_score: float
    task_scores: dict[str, float]
    total_params_m: float | None = None
    trunk_params_m: float | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rank": self.rank,
            "tier": self.tier,
            "model": self.model,
            "elo_median": round(self.elo_median, 2),
            "elo_mean": round(self.elo_mean, 2),
            "ci_low": round(self.ci_low, 2),
            "ci_high": round(self.ci_high, 2),
            "mean_task_score": round(self.mean_task_score, 6),
            "task_scores": {k: round(v, 6) for k, v in self.task_scores.items()},
        }
        if self.total_params_m is not None:
            payload["total_params_m"] = round(self.total_params_m, 2)
        if self.trunk_params_m is not None:
            payload["trunk_params_m"] = round(self.trunk_params_m, 2)
        return payload


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """Complete Bradley-Terry Elo leaderboard for one evaluation track."""

    track: str
    anchor_model: str
    anchor_elo: float
    num_bootstraps: int
    tie_deadzone: float
    tasks: tuple[str, ...]
    ratings: tuple[ModelRating, ...]
    pairwise_win_prob: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "anchor_model": self.anchor_model,
            "anchor_elo": self.anchor_elo,
            "num_bootstraps": self.num_bootstraps,
            "tie_deadzone": self.tie_deadzone,
            "tasks": list(self.tasks),
            "ratings": [r.as_dict() for r in self.ratings],
            "pairwise_win_prob": {
                m: {n: round(p, 4) for n, p in row.items()}
                for m, row in self.pairwise_win_prob.items()
            },
        }


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_bradley_terry(
    win_matrix: Sequence[Sequence[float]],
    *,
    anchor_index: int = 0,
    anchor_elo: float = DEFAULT_ANCHOR_ELO,
    reg: float = DEFAULT_REGULARIZATION,
    tol: float = DEFAULT_TOL,
) -> list[float]:
    """Fit global MLE Bradley-Terry log-strengths with L2 regularization and project to Elo."""
    n = len(win_matrix)
    if n == 0:
        msg = "empty win matrix"
        raise EloError(msg)
    if not (0 <= anchor_index < n):
        msg = f"anchor_index {anchor_index} out of bounds for {n} models"
        raise EloError(msg)
    if n == 1:
        return [anchor_elo]

    theta = [0.0] * n
    for _ in range(DEFAULT_MAX_ITER):
        max_step = 0.0
        for i in range(n):
            grad = -2.0 * reg * theta[i]
            hess = -2.0 * reg
            for j in range(n):
                if i == j:
                    continue
                w_ij = win_matrix[i][j]
                total_ij = w_ij + win_matrix[j][i]
                if total_ij <= 0.0:
                    continue
                p_ij = _sigmoid(theta[i] - theta[j])
                grad += w_ij - total_ij * p_ij
                hess -= total_ij * p_ij * (1.0 - p_ij)
            if abs(hess) > HESSIAN_EPS:
                step = max(-2.0, min(2.0, -grad / hess))
                theta[i] += step
                max_step = max(max_step, abs(step))
        mean_theta = sum(theta) / n
        theta = [t - mean_theta for t in theta]
        if max_step < tol:
            break

    base = theta[anchor_index]
    return [anchor_elo + ELO_SCALE * (t - base) for t in theta]


def win_matrix_from_scores(
    task_scores_by_model: Sequence[Mapping[str, float]],
    tasks: Sequence[str],
    *,
    deadzone: float = DEFAULT_DEADZONE,
) -> list[list[float]]:
    """Compute pairwise win matrix W[m][n] across tasks with a noise deadzone."""
    m_count = len(task_scores_by_model)
    matrix = [[0.0] * m_count for _ in range(m_count)]
    for t in tasks:
        for i in range(m_count):
            s_i = task_scores_by_model[i].get(t)
            if s_i is None:
                continue
            for j in range(i + 1, m_count):
                s_j = task_scores_by_model[j].get(t)
                if s_j is None:
                    continue
                diff = s_i - s_j
                if diff > deadzone:
                    matrix[i][j] += 1.0
                elif diff < -deadzone:
                    matrix[j][i] += 1.0
                else:
                    matrix[i][j] += 0.5
                    matrix[j][i] += 0.5
    return matrix


def _point_scores_and_lengths(
    models: Sequence[str],
    tasks: Sequence[str],
    model_task_items: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
) -> tuple[list[dict[str, float]], dict[str, int]]:
    point_scores: list[dict[str, float]] = []
    for m in models:
        m_scores: dict[str, float] = {}
        for t in tasks:
            seed_runs = model_task_items[m].get(t, ())
            means = [sum(run) / len(run) for run in seed_runs if len(run) > 0]
            if means:
                m_scores[t] = sum(means) / len(means)
        point_scores.append(m_scores)

    lengths: dict[str, int] = {}
    for t in tasks:
        lens = {len(r) for m in models for r in model_task_items[m].get(t, ()) if len(r) > 0}
        lengths[t] = min(lens) if lens else 0
    return point_scores, lengths


def _sample_bootstrap_replicate(
    models: Sequence[str],
    tasks: Sequence[str],
    model_task_items: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    task_lengths: Mapping[str, int],
    rng: random.Random,
) -> list[dict[str, float]]:
    sampled_indices = {
        t: ([rng.randrange(n) for _ in range(n)] if (n := task_lengths[t]) > 1 else [0])
        for t in tasks
    }
    rep_scores: list[dict[str, float]] = []
    for m in models:
        m_rep: dict[str, float] = {}
        for t in tasks:
            runs = model_task_items[m].get(t, ())
            if not runs:
                continue
            chosen = runs[rng.randrange(len(runs))]
            if not chosen:
                continue
            if len(chosen) > 1 and task_lengths[t] > 1:
                idxs = sampled_indices[t]
                m_rep[t] = sum(chosen[idx] for idx in idxs) / len(idxs)
            else:
                m_rep[t] = float(chosen[0])
        rep_scores.append(m_rep)
    return rep_scores


def _pairwise_probs(
    models: Sequence[str], elo_samples: Sequence[Sequence[float]]
) -> dict[str, dict[str, float]]:
    pairwise: dict[str, dict[str, float]] = {m: {} for m in models}
    b_total = len(elo_samples[0])
    for i, m_i in enumerate(models):
        for j, m_j in enumerate(models):
            if i == j:
                pairwise[m_i][m_j] = 0.5
                continue
            wins = sum(
                1.0
                if elo_samples[i][b] > elo_samples[j][b]
                else (0.5 if elo_samples[i][b] == elo_samples[j][b] else 0.0)
                for b in range(b_total)
            )
            pairwise[m_i][m_j] = wins / b_total
    return pairwise


def compute_leaderboard(
    track: str,
    model_task_items: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    *,
    anchor_model: str | None = None,
    num_bootstraps: int = DEFAULT_BOOTSTRAPS,
    model_metadata: Mapping[str, tuple[float, float]] | None = None,
) -> Leaderboard:
    """Compute Paired Cluster-Bootstrap Global Bradley-Terry Elo Leaderboard."""
    models = sorted(model_task_items.keys())
    if not models:
        msg = "no models provided to compute_leaderboard"
        raise EloError(msg)

    tasks = tuple(sorted({t for m in models for t in model_task_items[m]}))
    if not tasks:
        msg = "no tasks provided to compute_leaderboard"
        raise EloError(msg)

    resolved_anchor = anchor_model if anchor_model in models else models[0]
    anchor_idx = models.index(resolved_anchor)
    point_scores, task_lengths = _point_scores_and_lengths(models, tasks, model_task_items)

    rng = random.Random(42)  # noqa: S311
    elo_samples: list[list[float]] = [[] for _ in models]
    for _ in range(max(1, num_bootstraps)):
        rep = _sample_bootstrap_replicate(models, tasks, model_task_items, task_lengths, rng)
        win_mat = win_matrix_from_scores(rep, tasks, deadzone=DEFAULT_DEADZONE)
        rep_elos = fit_bradley_terry(
            win_mat, anchor_index=anchor_idx, anchor_elo=DEFAULT_ANCHOR_ELO
        )
        for i, val in enumerate(rep_elos):
            elo_samples[i].append(val)

    pairwise = _pairwise_probs(models, elo_samples)
    raw_stats: list[tuple[int, float, float, float, float]] = []
    for i in range(len(models)):
        s_sorted = sorted(elo_samples[i])
        n_s = len(s_sorted)
        mid = n_s // 2
        median = s_sorted[mid] if n_s % 2 == 1 else 0.5 * (s_sorted[mid - 1] + s_sorted[mid])
        mean = sum(s_sorted) / n_s
        low = s_sorted[max(0, int(0.025 * (n_s - 1)))]
        high = s_sorted[min(n_s - 1, int(0.975 * (n_s - 1)))]
        raw_stats.append((i, median, mean, low, high))

    raw_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ratings: list[ModelRating] = []
    current_tier = 1
    tier_leader = models[raw_stats[0][0]]
    for rank_idx, (m_idx, median, mean, low, high) in enumerate(raw_stats, start=1):
        m_name = models[m_idx]
        if rank_idx > 1 and pairwise[tier_leader][m_name] >= TIER_CONFIDENCE:
            current_tier += 1
            tier_leader = m_name
        t_scores = point_scores[m_idx]
        mean_score = sum(t_scores.values()) / len(t_scores) if t_scores else 0.0
        meta = model_metadata.get(m_name) if model_metadata else None
        ratings.append(
            ModelRating(
                model=m_name,
                elo_median=median,
                elo_mean=mean,
                ci_low=low,
                ci_high=high,
                rank=rank_idx,
                tier=current_tier,
                mean_task_score=mean_score,
                task_scores=t_scores,
                total_params_m=meta[0] if meta else None,
                trunk_params_m=meta[1] if meta else None,
            )
        )

    return Leaderboard(
        track=track,
        anchor_model=resolved_anchor,
        anchor_elo=DEFAULT_ANCHOR_ELO,
        num_bootstraps=max(1, num_bootstraps),
        tie_deadzone=DEFAULT_DEADZONE,
        tasks=tasks,
        ratings=tuple(ratings),
        pairwise_win_prob=pairwise,
    )
