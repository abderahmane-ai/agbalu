r"""Semantic similarity, scored beside the control that makes the score readable.

Pearson and Spearman against the gold scores the caller supplies, repeated at each
Matryoshka slice so a truncated vector is measured rather than assumed.

**The control is the reason this module exists.** Mean cosine similarity over aligned pairs
is what the published Kabyle sentence transformer reports, and an encoder that maps every
sentence into one narrow cone maximises it while telling two sentences apart not at all.
`check_isotropic_collapse` samples unaligned pairs: a space that scores them as similar as
the aligned ones is refused, whatever its correlation says.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

COLLAPSE_MEAN_THRESHOLD: Final[float] = 0.50
"""Uncorrelated random pairs averaging above this cosine similarity indicate space collapse."""

COLLAPSE_STD_THRESHOLD: Final[float] = 0.08
"""Embedding standard deviation below this indicates severe representation narrowing."""

MIN_SAMPLES: Final[int] = 2
MAX_ATTEMPTS_MULTIPLIER: Final[int] = 10


@dataclass(frozen=True, slots=True)
class STSPair:
    """One evaluation pair with a ground-truth similarity score."""

    sentence1: str
    sentence2: str
    score: float
    genre: str = "general"


@dataclass(frozen=True, slots=True)
class STSMetrics:
    """Correlation and ranking metrics on the STS benchmark."""

    pearson: float
    spearman: float
    pairs_evaluated: int
    matryoshka: dict[int, dict[str, float]]


@dataclass(frozen=True, slots=True)
class IsotropicCheck:
    """Diagnostic metrics proving whether the embedding geometry is isotropic or collapsed."""

    mean_cosine: float
    std_cosine: float
    min_cosine: float
    max_cosine: float
    collapsed: bool
    pairs_sampled: int


@dataclass(frozen=True, slots=True)
class STSReport:
    """Complete evaluation report combining STS performance and collapse validation."""

    model_name: str
    metrics: STSMetrics
    isotropic: IsotropicCheck

    @property
    def passed(self) -> bool:
        return not self.isotropic.collapsed and self.metrics.spearman > 0.0


def rank_data(values: Sequence[float]) -> list[float]:
    """Assign fractional ranks to data, handling ties with standard average ranking."""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Linear correlation. Returns 0.0 under `MIN_SAMPLES`, which reads as no relationship
    rather than as too little data to say — check the sample count before quoting it."""
    n = len(x)
    if n != len(y) or n < MIN_SAMPLES:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    if var_x == 0.0 or var_y == 0.0:
        return 0.0

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    return float(cov / math.sqrt(var_x * var_y))


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, which is the same computation over ranked inputs and carries the
    same 0.0 floor."""
    if len(x) != len(y) or len(x) < MIN_SAMPLES:
        return 0.0
    return pearson_correlation(rank_data(x), rank_data(y))


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity between two real vectors."""
    dot = sum(ui * vi for ui, vi in zip(u, v, strict=True))
    norm_u = math.sqrt(sum(ui * ui for ui in u))
    norm_v = math.sqrt(sum(vi * vi for vi in v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(dot / (norm_u * norm_v))


def check_isotropic_collapse(
    embeddings: Sequence[Sequence[float]],
    *,
    pairs_sample: int = 5000,
    seed: int = 42,
) -> IsotropicCheck:
    """Check embedding space isotropy over unaligned random pairs."""
    n = len(embeddings)
    if n < MIN_SAMPLES:
        return IsotropicCheck(0.0, 0.0, 0.0, 0.0, collapsed=True, pairs_sampled=0)

    rng = random.Random(seed)  # noqa: S311
    sampled = min(pairs_sample, n * (n - 1) // 2)

    cosines: list[float] = []
    seen: set[tuple[int, int]] = set()

    attempts = 0
    max_attempts = sampled * MAX_ATTEMPTS_MULTIPLIER

    while len(cosines) < sampled and attempts < max_attempts:
        attempts += 1
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        cosines.append(cosine_similarity(embeddings[i], embeddings[j]))

    if not cosines:
        return IsotropicCheck(0.0, 0.0, 0.0, 0.0, collapsed=True, pairs_sampled=0)

    mean_cos = sum(cosines) / len(cosines)
    variance = sum((c - mean_cos) ** 2 for c in cosines) / len(cosines)
    std_cos = math.sqrt(variance)

    collapsed = mean_cos > COLLAPSE_MEAN_THRESHOLD or std_cos < COLLAPSE_STD_THRESHOLD

    return IsotropicCheck(
        mean_cosine=round(mean_cos, 4),
        std_cosine=round(std_cos, 4),
        min_cosine=round(min(cosines), 4),
        max_cosine=round(max(cosines), 4),
        collapsed=collapsed,
        pairs_sampled=len(cosines),
    )


def evaluate_sts(
    pairs: Sequence[STSPair],
    encode_fn: Callable[[list[str]], list[list[float]]],
    *,
    matryoshka_dims: Sequence[int] = (768, 512, 256, 128, 64),
) -> STSMetrics:
    """Evaluate STS benchmark performance across full and truncated Matryoshka dimensions."""
    if not pairs:
        return STSMetrics(0.0, 0.0, 0, {})

    s1_list = [p.sentence1 for p in pairs]
    s2_list = [p.sentence2 for p in pairs]
    gold_scores = [p.score for p in pairs]

    emb1 = encode_fn(s1_list)
    emb2 = encode_fn(s2_list)

    full_dim = len(emb1[0]) if emb1 else 0
    valid_dims = [d for d in matryoshka_dims if d <= full_dim]
    if full_dim not in valid_dims and full_dim > 0:
        valid_dims.insert(0, full_dim)

    matryoshka_results: dict[int, dict[str, float]] = {}

    for d in valid_dims:
        pred_cosines = [
            cosine_similarity(e1[:d], e2[:d]) for e1, e2 in zip(emb1, emb2, strict=True)
        ]
        r = pearson_correlation(gold_scores, pred_cosines)
        rho = spearman_correlation(gold_scores, pred_cosines)
        matryoshka_results[d] = {
            "pearson": round(r, 4),
            "spearman": round(rho, 4),
        }

    full_res = matryoshka_results.get(full_dim, {"pearson": 0.0, "spearman": 0.0})

    return STSMetrics(
        pearson=full_res["pearson"],
        spearman=full_res["spearman"],
        pairs_evaluated=len(pairs),
        matryoshka=matryoshka_results,
    )
