"""Fit output intercepts on reserved development documents."""

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.nn import functional

from agbalu.restoration.annotations import IGNORE
from agbalu.restoration.dataset import Windows, collate
from agbalu.restoration.evaluation import confusion, fold_indices, macro
from agbalu.restoration.model import Restorer, Scores

FALSE_INSERTION_BUDGET = 37.0


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    presence_bias: float = 0.0
    casing_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    type_bias: tuple[float, float, float, float, float, float, float, float] = (0.0,) * 8

    def apply(self, scores: Scores) -> Scores:
        return Scores(
            scores.presence.float() + self.presence_bias,
            scores.punctuation.float()
            + torch.tensor(self.type_bias, device=scores.punctuation.device, dtype=torch.float32),
            scores.casing.float()
            + torch.tensor(self.casing_bias, device=scores.casing.device, dtype=torch.float32),
            scores.structure,
        )


def fit_class_bias(logits: Tensor, labels: Tensor) -> tuple[float, ...]:
    classes = logits.size(-1)
    biases = torch.zeros(classes, dtype=torch.float64)
    frequencies = torch.bincount(labels, minlength=classes).double() / len(labels)
    regularization = 1e-4
    for _ in range(32):
        probabilities = (logits.double() + biases).softmax(-1)[:, 1:]
        gradient = probabilities.mean(0) - frequencies[1:] + regularization * biases[1:]
        hessian = (
            probabilities.mean(0).diag()
            - probabilities.T @ probabilities / len(labels)
            + regularization * torch.eye(classes - 1, dtype=torch.float64)
        )
        biases[1:] -= torch.linalg.solve(hessian, gradient).clamp(-2, 2)
    return tuple(float(value) for value in biases)


def fit_intercepts(
    presence: Tensor, marks: Tensor, cases: Tensor, labels: Tensor, types: Tensor | None = None
) -> Calibration:
    if not marks.numel() or not labels.numel():
        message = (
            f"calibration requires punctuation and casing support: {marks.numel()}/{labels.numel()}"
        )
        raise ValueError(message)
    presence = presence.double()
    parameter = torch.zeros((), dtype=torch.float64)
    regularization = 1e-3
    for _ in range(32):
        probability = (presence + parameter).sigmoid()
        gradient = (probability - (marks > 0).double()).mean() + regularization * parameter
        curvature = (probability * (1 - probability)).mean() + regularization
        parameter -= (gradient / curvature).clamp(-2, 2)
    positive = marks > 0
    if types is not None and not bool(positive.any()):
        message = "calibration fold has no positive punctuation targets"
        raise ValueError(message)
    payload = {
        "presence_bias": float(parameter),
        "casing_bias": fit_class_bias(cases, labels),
        "type_bias": fit_class_bias(types[positive], marks[positive] - 1)
        if types is not None
        else (0.0,) * 8,
    }
    return Calibration.model_validate(payload)


def fit(
    model: Restorer, dataset: Windows, device: torch.device
) -> tuple[Calibration, dict[str, object]]:
    order = fold_indices(dataset, "calibration", 128)
    presence, types, marks, cases, labels = [], [], [], [], []
    words = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in order:
            batch = collate([dataset[int(index)]], model.geometry.context_tokens).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = model.forward(batch)
            valid = batch.marks != IGNORE
            known = batch.casing != IGNORE
            presence.append(scores.presence[valid].float().cpu())
            types.append(scores.punctuation[valid].float().cpu())
            marks.append(batch.marks[valid].cpu())
            cases.append(scores.casing[known].float().cpu())
            labels.append(batch.casing[known].cpu())
            words += int(batch.owned_words.sum())
    model.train(was_training)
    p, t, gold, c, case_gold = map(torch.cat, (presence, types, marks, cases, labels))
    fitted = fit_intercepts(p, gold, c, case_gold, t)
    best_score = -1.0
    chosen = -20.0
    trials: list[dict[str, object]] = []
    log_types = (t + torch.tensor(fitted.type_bias)).log_softmax(-1)
    for adjustment in (-20.0, *(index / 4 for index in range(-16, 9))):
        bias = fitted.presence_bias + adjustment
        logits = p + bias
        joint = torch.cat(
            (
                functional.logsigmoid(-logits)[:, None],
                functional.logsigmoid(logits)[:, None] + log_types,
            ),
            -1,
        )
        matrix = confusion(gold, joint.argmax(-1), 9)
        score = macro(matrix, start=1)
        false = 1000 * int(matrix[0, 1:].sum()) / words
        trials.append({"bias": bias, "macro_f1": score, "false_insertions_per_1000_words": false})
        if score is not None and false <= FALSE_INSERTION_BUDGET and score > best_score:
            chosen, best_score = bias, score
    calibration = fitted.model_copy(update={"presence_bias": chosen})
    return calibration, {
        "calibration": calibration.model_dump(),
        "windows": len(order),
        "document_ids": sorted({dataset.windows[int(index)][0] for index in order}),
        "punctuation_support": len(gold),
        "casing_support": len(case_gold),
        "words": words,
        "false_insertion_budget_per_1000_words": FALSE_INSERTION_BUDGET,
        "threshold_trials": trials,
    }
