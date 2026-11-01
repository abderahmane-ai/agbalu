"""Measure classification support and calibrate bounded conformal scores."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import TypedDict

import torch
from torch.nn import functional

from agbalu.dihya.config import ModelConfig
from agbalu.dihya.dataset import Example, TextDataset, collate_examples
from agbalu.dihya.model import Classifier


class Evaluation(TypedDict):
    accuracy: float
    macro_f1: float
    selection_score: float
    support: int
    worst_group_recall: float | None
    worst_group: str | None
    worst_class_recall: float
    worst_class: str
    raw_worst_group_recall: float
    raw_worst_group: str
    min_group_support: int
    group_support: dict[str, int]
    group_recalls: dict[str, float]
    excluded_groups: list[str]
    class_support: dict[str, int]
    class_recalls: dict[str, float | None]
    class_precision: dict[str, float | None]
    class_f1: dict[str, float | None]
    confusion: list[list[int]]


def partition_development(
    dataset: TextDataset, seed: int
) -> tuple[TextDataset, TextDataset, TextDataset]:
    """Reserve calibration fit and coverage rows before checkpoint selection."""
    rng = random.Random(seed)
    cells: dict[str, list[Example]] = {}
    for example in dataset.examples:
        cells.setdefault(example.group_key, []).append(example)
    selection: list[Example] = []
    fit: list[Example] = []
    coverage: list[Example] = []
    for key in sorted(cells):
        rows = sorted(cells[key], key=lambda example: example.text)
        rng.shuffle(rows)
        quarter = len(rows) // 4
        if quarter:
            fit.extend(rows[:quarter])
            coverage.extend(rows[quarter : 2 * quarter])
        selection.extend(rows[2 * quarter :])
    return TextDataset(selection), TextDataset(fit), TextDataset(coverage)


def classification_metrics(
    predictions: Sequence[int],
    targets: Sequence[int],
    groups: Sequence[str],
    classes: Sequence[str],
    min_group_support: int = 20,
) -> Evaluation:
    """Report all cells while selecting on cells with declared minimum support."""
    if not targets or min_group_support < 1:
        msg = "Metrics require nonempty targets and positive min_group_support"
        raise ValueError(msg)
    matrix = [[0 for _ in classes] for _ in classes]
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for prediction, target, group in zip(predictions, targets, groups, strict=True):
        if not 0 <= prediction < len(classes) or not 0 <= target < len(classes):
            msg = f"Invalid prediction {prediction} or target {target}"
            raise ValueError(msg)
        matrix[target][prediction] += 1
        totals[group] += 1
        correct[group] += int(prediction == target)
    support = {label: sum(matrix[index]) for index, label in enumerate(classes)}
    recalls: dict[str, float | None] = {}
    precisions: dict[str, float | None] = {}
    f1: dict[str, float | None] = {}
    for index, label in enumerate(classes):
        tp = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        recalls[label] = tp / support[label] if support[label] else None
        precisions[label] = tp / predicted if predicted else None
        f1[label] = 2 * tp / (support[label] + predicted) if support[label] else None
    measured_f1 = [value for value in f1.values() if value is not None]
    macro = sum(measured_f1) / len(measured_f1)
    measured_recalls = {label: value for label, value in recalls.items() if value is not None}
    worst_class = min(measured_recalls, key=lambda label: measured_recalls[label])
    group_recalls = {group: correct[group] / total for group, total in sorted(totals.items())}
    eligible = {
        group: recall
        for group, recall in group_recalls.items()
        if totals[group] >= min_group_support
    }
    worst = min(eligible, key=lambda group: eligible[group]) if eligible else None
    worst_recall = eligible[worst] if worst is not None else None
    raw_worst = min(group_recalls, key=lambda group: group_recalls[group])
    return Evaluation(
        accuracy=sum(matrix[i][i] for i in range(len(classes))) / len(targets),
        macro_f1=macro,
        selection_score=(macro + worst_recall) / 2 if worst_recall is not None else macro,
        support=len(targets),
        worst_group_recall=worst_recall,
        worst_group=worst,
        worst_class_recall=measured_recalls[worst_class],
        worst_class=worst_class,
        raw_worst_group_recall=group_recalls[raw_worst],
        raw_worst_group=raw_worst,
        min_group_support=min_group_support,
        group_support=dict(totals),
        group_recalls=group_recalls,
        excluded_groups=sorted(set(totals) - set(eligible)),
        class_support=support,
        class_recalls=recalls,
        class_precision=precisions,
        class_f1=f1,
        confusion=matrix,
    )


def evaluate_dataset(
    model: Classifier,
    dataset: TextDataset,
    config: ModelConfig,
    device: torch.device,
    batch_size: int = 128,
    *,
    min_group_support: int = 20,
) -> Evaluation:
    """Evaluate fixed rows in length order without retaining unused probabilities."""
    if batch_size < 1:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    was_training = model.training
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    groups: list[str] = []
    ordered = sorted(dataset.examples, key=lambda ex: len(ex.text.encode("utf-8")))
    try:
        with torch.inference_mode():
            for start in range(0, len(ordered), batch_size):
                examples = ordered[start : start + batch_size]
                batch = collate_examples(examples, config, {}, {})
                output = model(batch.input_ids.to(device), pad_mask=batch.pad_mask.to(device))
                predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
                targets.extend(batch.labels.tolist())
                groups.extend(ex.group_key for ex in examples)
    finally:
        model.train(was_training)
    return classification_metrics(predictions, targets, groups, config.classes, min_group_support)


def conformal_quantile(scores: Sequence[float], target_coverage: float) -> float:
    """Use the finite-sample rank, including the bounded-score infinity case."""
    if not scores or not 0 < target_coverage < 1:
        msg = "Calibration needs scores and 0 < target_coverage < 1"
        raise ValueError(msg)
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        msg = f"Invalid nonconformity scores {scores!r}"
        raise ValueError(msg)
    rank = math.ceil((len(scores) + 1) * target_coverage)
    return sorted(scores)[rank - 1] if rank <= len(scores) else 1.0


def calibration_scores(
    model: Classifier,
    dataset: TextDataset,
    config: ModelConfig,
    device: torch.device,
) -> list[float]:
    """Measure true-class nonconformity without changing the model's mode."""
    was_training = model.training
    model.eval()
    scores: list[float] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(dataset), 64):
                batch = collate_examples(dataset.examples[start : start + 64], config, {}, {})
                output = model(batch.input_ids.to(device), pad_mask=batch.pad_mask.to(device))
                probs = functional.softmax(output.logits.float(), dim=-1)
                selected = probs.gather(1, batch.labels.to(device).unsqueeze(1)).squeeze(1)
                scores.extend((1 - selected).cpu().tolist())
    finally:
        model.train(was_training)
    return scores


def calibrate_conformal(
    model: Classifier,
    calibration_dataset: TextDataset,
    config: ModelConfig,
    device: torch.device,
    target_coverage: float = 0.99,
) -> float:
    """Fit a quantile on calibration rows only."""
    return conformal_quantile(
        calibration_scores(model, calibration_dataset, config, device), target_coverage
    )
