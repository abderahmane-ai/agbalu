"""The correlations, the Matryoshka slices, and the collapse control beside them."""

from __future__ import annotations

import pytest

from agbalu.bench.sts import (
    STSPair,
    check_isotropic_collapse,
    cosine_similarity,
    evaluate_sts,
    pearson_correlation,
    rank_data,
    spearman_correlation,
)


class TestCorrelationMetrics:
    def test_rank_data_with_ties(self) -> None:
        values = [10.0, 20.0, 20.0, 30.0]
        # Rank of 10 is 1.0, ranks of (20, 20) are 2.5, rank of 30 is 4.0
        assert rank_data(values) == [1.0, 2.5, 2.5, 4.0]

    def test_pearson_perfect_linear(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        assert pytest.approx(pearson_correlation(x, y), abs=1e-5) == 1.0

    def test_pearson_inverse_linear(self) -> None:
        x = [1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0]
        assert pytest.approx(pearson_correlation(x, y), abs=1e-5) == -1.0

    def test_pearson_constant_returns_zero(self) -> None:
        x = [1.0, 1.0, 1.0]
        y = [2.0, 3.0, 4.0]
        assert pearson_correlation(x, y) == 0.0

    def test_spearman_monotonic_nonlinear(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 10.0, 100.0, 1000.0]
        # Spearman should be 1.0 even though Pearson is non-linear
        assert pytest.approx(spearman_correlation(x, y), abs=1e-5) == 1.0


class TestCosineSimilarity:
    def test_orthogonal_vectors(self) -> None:
        u = [1.0, 0.0]
        v = [0.0, 1.0]
        assert pytest.approx(cosine_similarity(u, v), abs=1e-5) == 0.0

    def test_identical_vectors(self) -> None:
        u = [3.0, 4.0]
        assert pytest.approx(cosine_similarity(u, u), abs=1e-5) == 1.0


class TestIsotropicCollapseCheck:
    def test_detects_collapsed_embedding_space(self) -> None:
        # All vectors clustered within a 0.01 cone
        collapsed_space = [[1.0, 0.01 * i, 0.01 * (i % 2)] for i in range(50)]
        check = check_isotropic_collapse(collapsed_space, pairs_sample=200, seed=42)
        assert check.collapsed is True
        assert check.mean_cosine > 0.95

    def test_passes_healthy_isotropic_space(self) -> None:
        # Standard basis and alternating orthogonal vectors
        healthy_space = [[1.0 if j == (i % 8) else 0.0 for j in range(8)] for i in range(50)]
        check = check_isotropic_collapse(healthy_space, pairs_sample=200, seed=42)
        assert check.collapsed is False
        assert check.mean_cosine < 0.30
        assert check.std_cosine > 0.10


class TestEvaluateSTS:
    def test_evaluates_matryoshka_slices(self) -> None:
        pairs = [
            STSPair("Axxam amellal", "La maison blanche", 5.0),
            STSPair("Aman semmḍit", "L'eau est froide", 4.0),
            STSPair("Iṭij yeččur", "Le ciel est bleu", 1.0),
        ]

        def fake_encoder(sentences: list[str]) -> list[list[float]]:
            # Vector where first 4 dims encode length/lexical features, rest is noise
            vectors: list[list[float]] = []
            for s in sentences:
                val = float(len(s))
                vec = [val, val * 0.5, val * 0.2, val * 0.1] + [0.01] * 60
                vectors.append(vec)
            return vectors

        metrics = evaluate_sts(pairs, fake_encoder, matryoshka_dims=(64, 4))
        assert metrics.pairs_evaluated == 3
        assert 64 in metrics.matryoshka
        assert 4 in metrics.matryoshka
        assert "spearman" in metrics.matryoshka[64]
