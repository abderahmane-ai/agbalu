"""Unit tests for KabEncBench: Bradley-Terry Elo, scale-normalized probes, and metrics."""

from __future__ import annotations

from pathlib import Path

import torch

from agbalu.bench import cli
from agbalu.bench.elo import (
    DEFAULT_ANCHOR_ELO,
    compute_leaderboard,
    fit_bradley_terry,
    win_matrix_from_scores,
)
from agbalu.bench.encbench import (
    ENCODER_REGISTRY,
    REGISTRY_BY_ID,
    ScalarLayerMix,
    TaskRunResult,
    build_track_leaderboard,
    evaluate_paraphrase_retrieval,
    extract_iob_spans,
    macro_f1_from_pairs,
    strict_span_f1,
)
from agbalu.bench.encbench_data import _extract_punct_case_sequence
from agbalu.bench.encbench_runner import (
    CachedClassificationData,
    DualPunctCaseHead,
    FactoredMorphHead,
    FitSchedule,
    SyntaxJointHead,
    evaluate_cached_classification,
)


def test_encoder_registry_has_13_verified_checkpoints() -> None:
    assert len(ENCODER_REGISTRY) == 13
    assert "agbalu/Masinissa-31M" in REGISTRY_BY_ID
    assert "agbalu/Masinissa-73M" in REGISTRY_BY_ID
    assert "FacebookAI/xlm-roberta-base" in REGISTRY_BY_ID
    for spec in ENCODER_REGISTRY:
        assert spec.total_params_m >= spec.trunk_params_m > 0.0
        assert spec.layers > 0
        assert spec.hidden_size > 0


def test_bradley_terry_anchor_and_monotonicity() -> None:
    scores = [
        {"syntax": 0.90, "morph": 0.88, "ner": 0.85, "sent": 0.86, "topic": 0.80, "punct": 0.84},
        {"syntax": 0.80, "morph": 0.78, "ner": 0.75, "sent": 0.76, "topic": 0.70, "punct": 0.74},
        {"syntax": 0.70, "morph": 0.68, "ner": 0.65, "sent": 0.66, "topic": 0.60, "punct": 0.64},
    ]
    tasks = ("syntax", "morph", "ner", "sent", "topic", "punct")
    win_mat = win_matrix_from_scores(scores, tasks, deadzone=0.001)
    elos = fit_bradley_terry(win_mat, anchor_index=1, anchor_elo=DEFAULT_ANCHOR_ELO)
    assert abs(elos[1] - DEFAULT_ANCHOR_ELO) < 1e-6
    assert elos[0] > elos[1] > elos[2]


def test_compute_leaderboard_order_independent_and_rank_sets() -> None:
    model_items = {
        "model-strong": {
            "syntax": [[1.0, 1.0, 1.0, 0.9, 1.0], [1.0, 0.9, 1.0, 1.0, 1.0]],
            "morph": [[0.9, 0.9, 1.0, 0.9, 0.9]],
        },
        "model-anchor": {
            "syntax": [[0.7, 0.6, 0.7, 0.6, 0.7], [0.6, 0.7, 0.6, 0.7, 0.6]],
            "morph": [[0.6, 0.6, 0.5, 0.6, 0.6]],
        },
        "model-tie-with-anchor": {
            "syntax": [[0.7, 0.6, 0.7, 0.6, 0.7], [0.6, 0.7, 0.6, 0.7, 0.6]],
            "morph": [[0.6, 0.6, 0.5, 0.6, 0.6]],
        },
    }
    lb = compute_leaderboard(
        "frozen",
        model_items,
        anchor_model="model-anchor",
        num_bootstraps=200,
    )
    assert lb.ratings[0].model == "model-strong"
    assert lb.ratings[0].rank == 1
    assert lb.ratings[0].tier == 1
    # Anchor and its identical twin must share the same statistical tier (tier 2)
    assert lb.ratings[1].tier == 2
    assert lb.ratings[2].tier == 2
    anchor_rating = next(r for r in lb.ratings if r.model == "model-anchor")
    assert abs(anchor_rating.elo_median - 1000.0) < 1e-5


def test_scalar_layer_mix_normalizes_scale() -> None:
    mix = ScalarLayerMix(num_layers=4, hidden_size=32, out_dim=16)
    layers = [torch.randn(2, 5, 32) * (10.0**i) for i in range(5)]
    out = mix(layers)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_strict_span_f1_and_iob_extraction() -> None:
    gold = [["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "I-ORG"]]
    pred = [["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O"]]
    spans = extract_iob_spans(gold[0])
    assert spans == {(0, 2, "PER"), (3, 4, "LOC"), (5, 7, "ORG")}
    micro_f1, per_cat, item_f1s = strict_span_f1(gold, pred)
    assert abs(micro_f1 - (2 * 2 / (2 * 2 + 1 + 1))) < 1e-6
    assert per_cat["PER"] == 1.0
    assert per_cat["LOC"] == 1.0
    assert per_cat["ORG"] == 0.0
    assert len(item_f1s) == 1


def test_paraphrase_retrieval_and_isotropy() -> None:
    embeddings = [
        [1.0, 0.0, 0.0, 0.1],
        [0.98, 0.05, 0.0, 0.1],
        [0.0, 1.0, 0.0, -0.1],
        [0.05, 0.97, 0.0, -0.1],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.04, 0.99, 0.2],
    ]
    clusters = [1, 1, 2, 2, 3, 3]
    res = evaluate_paraphrase_retrieval(embeddings, clusters)
    assert res["map_at_r"] > 0.95
    assert res["mrr_at_10"] > 0.95
    assert res["collapsed"] == 0.0


def test_extract_punct_case_sequence() -> None:
    parsed = _extract_punct_case_sequence("Yella Mulud deg Tizi, neɣ di Bgayet?")
    assert parsed is not None
    words, punct, case = parsed
    assert words == ["yella", "mulud", "deg", "tizi", "neɣ", "di", "bgayet"]
    assert case == ["CAP", "CAP", "LOWER", "CAP", "LOWER", "LOWER", "CAP"]
    assert punct == ["NONE", "NONE", "NONE", "COMMA", "NONE", "NONE", "QUESTION"]


def test_macro_f1_from_pairs_excludes_none() -> None:
    gold = [0, 0, 1, 2, 1]
    pred = [0, 0, 1, 1, 1]
    f1_all = macro_f1_from_pairs(gold, pred, 3)
    f1_active = macro_f1_from_pairs(gold, pred, 3, exclude={0})
    assert f1_all != f1_active
    assert 0.0 < f1_active < 1.0


def test_build_track_leaderboard_attaches_trunk_params() -> None:
    results = {
        "agbalu/Masinissa-73M": {
            "syntax": TaskRunResult(
                task="syntax",
                track="frozen",
                selected_lr=1e-3,
                primary_metric=0.92,
                primary_std=0.01,
                secondary_metrics={},
                seed_item_scores=((0.92, 0.93, 0.91),),
            )
        },
        "FacebookAI/xlm-roberta-base": {
            "syntax": TaskRunResult(
                task="syntax",
                track="frozen",
                selected_lr=1e-3,
                primary_metric=0.84,
                primary_std=0.01,
                secondary_metrics={},
                seed_item_scores=((0.84, 0.85, 0.83),),
            )
        },
    }
    lb = build_track_leaderboard("frozen", results, num_bootstraps=100)
    top = lb.ratings[0]
    assert top.model == "agbalu/Masinissa-73M"
    assert top.total_params_m == 73.32
    assert top.trunk_params_m == 65.13


def test_task_heads_and_cached_classification_runner(tmp_path: Path) -> None:
    syntax_head = SyntaxJointHead(in_dim=32, num_upos=17, num_feats=24, num_rels=35)
    x_seq = torch.randn(2, 6, 32)
    upos_logits, feats_logits, arc_scores, rel_dep = syntax_head(x_seq)
    assert upos_logits.shape == (2, 6, 17)
    assert feats_logits.shape == (2, 6, 24)
    assert arc_scores.shape == (2, 7, 7)
    assert rel_dep.shape == (2, 7, 64)

    morph_head = FactoredMorphHead(in_dim=32, attr_sizes={"Aspect": 4, "Person": 4})
    morph_out = morph_head(torch.randn(3, 32))
    assert morph_out["Aspect"].shape == (3, 4)
    assert morph_out["Person"].shape == (3, 4)

    punct_head = DualPunctCaseHead(in_dim=32)
    p_logits, c_logits = punct_head(x_seq)
    assert p_logits.shape == (2, 6, 5)
    assert c_logits.shape == (2, 6, 3)

    # Cached classification runner with LR selection and seed evaluation
    torch.manual_seed(42)
    train_x = [torch.randn(12, 16) for _ in range(3)]
    train_y = torch.tensor([i % 3 for i in range(12)], dtype=torch.long)
    dev_x = [torch.randn(6, 16) for _ in range(3)]
    dev_y = torch.tensor([i % 3 for i in range(6)], dtype=torch.long)
    test_x = [torch.randn(6, 16) for _ in range(3)]
    test_y = torch.tensor([i % 3 for i in range(6)], dtype=torch.long)
    run_res = evaluate_cached_classification(
        CachedClassificationData(
            task="sent",
            num_classes=3,
            train_layers=train_x,
            train_labels=train_y,
            dev_layers=dev_x,
            dev_labels=dev_y,
            test_layers=test_x,
            test_labels=test_y,
        ),
        lr_grid=(1e-3, 3e-3),
        seeds=(42, 43),
        schedule=FitSchedule(epochs=2, batch_size=6),
        device=torch.device("cpu"),
    )
    assert run_res.task == "sent"
    assert run_res.selected_lr in (1e-3, 3e-3)
    assert len(run_res.seed_item_scores) == 2

    rc = cli.main(["encbench", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "encbench-report.json").is_file()
