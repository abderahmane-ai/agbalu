"""Measure restoration predictions on fixed word and boundary populations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from agbalu.restoration.annotations import CASES, IGNORE, MARKS, TERMINAL_LABELS
from agbalu.restoration.dataset import Batch, Windows, collate
from agbalu.restoration.model import Restorer, Scores, joint_probabilities


def confusion(gold: Tensor, predicted: Tensor, classes: int) -> Tensor:
    valid = gold != IGNORE
    return torch.bincount(
        gold[valid] * classes + predicted[valid], minlength=classes * classes
    ).reshape(classes, classes)


def report(matrix: Tensor, names: tuple[str, ...]) -> dict[str, object]:
    values = matrix.cpu().numpy()
    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        support = int(values[index].sum())
        predicted = int(values[:, index].sum())
        hits = int(values[index, index])
        precision = hits / predicted if predicted else None
        recall = hits / support if support else None
        f1 = 2 * hits / (support + predicted) if support else None
        rows.append(
            {
                "label": name,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {"classes": rows, "scored": int(values.sum()), "confusion": values.tolist()}


def macro(matrix: Tensor, *, start: int = 0) -> float | None:
    values = matrix.float()
    support = values.sum(1)
    predicted = values.sum(0)
    present = support[start:] > 0
    if not bool(present.any()):
        return None
    f1 = 2 * values.diag() / (support + predicted).clamp(min=1)
    return float(f1[start:][present].mean())


def fold_indices(
    dataset: Windows,
    fold: Literal["diagnostic", "calibration", "selection"],
    limit: int | None,
) -> np.typing.NDArray[np.int64]:
    documents = np.asarray([document for document, _ in dataset.windows], np.int64)
    order = np.random.default_rng(1618).permutation(len(dataset))
    if fold != "diagnostic":
        unique = np.random.default_rng(2718).permutation(np.unique(documents))
        reserved = unique[: max(1, len(unique) // 5)]
        mask = np.isin(documents[order], reserved)
        order = order[mask if fold == "calibration" else ~mask]
    if not len(order):
        message = f"{fold}: no development windows after document separation"
        raise ValueError(message)
    return order if limit is None else order[:limit]


@dataclass
class Metrics:
    punctuation: Tensor
    casing: Tensor
    structure: Tensor
    noninitial: Tensor
    words: int = 0
    structure_hits: int = 0
    structure_support: int = 0
    word_case_hits: int = 0
    word_case_support: int = 0

    @classmethod
    def create(cls, device: torch.device, templates: int) -> "Metrics":
        matrices = [
            torch.zeros((classes, classes), device=device, dtype=torch.long)
            for classes in (len(MARKS), len(CASES), templates, len(CASES))
        ]
        return cls(matrices[0], matrices[1], matrices[2], matrices[3])

    def add(self, scores: Scores, batch: Batch) -> None:
        self.punctuation += confusion(
            batch.marks, joint_probabilities(scores).argmax(-1), len(MARKS)
        )
        casing = scores.casing.argmax(-1)
        self.casing += confusion(batch.casing, casing, len(CASES))
        real = batch.case_spans[..., 1] > batch.case_spans[..., 0]
        known = batch.casing != IGNORE
        shape = batch.words.shape
        supported = torch.zeros(shape, device=casing.device, dtype=torch.long).scatter_add(
            1, batch.case_words, known.long()
        )
        unknown = torch.zeros_like(supported).scatter_add(
            1, batch.case_words, (real & ~known).long()
        )
        wrong = torch.zeros_like(supported).scatter_add(
            1, batch.case_words, ((casing != batch.casing) & known).long()
        )
        whole_words = (supported > 0) & (unknown == 0) & batch.words
        self.word_case_support += int(whole_words.sum())
        self.word_case_hits += int((whole_words & (wrong == 0)).sum())
        noninitial = batch.casing.clone()
        first_component = batch.case_features[..., 0] == 0
        noninitial[(batch.case_words == 0) & first_component] = IGNORE
        terminal = torch.zeros_like(batch.casing, dtype=torch.bool)
        for label in TERMINAL_LABELS:
            terminal |= batch.marks.gather(1, batch.case_words) == label
        terminal &= first_component
        noninitial[terminal] = IGNORE
        self.noninitial += confusion(noninitial, casing, len(CASES))
        self.structure += confusion(
            batch.structure, scores.structure.argmax(-1), self.structure.size(0)
        )
        self.words += int(batch.owned_words.sum())

    def finish(self, templates: tuple[str, ...]) -> dict[str, object]:
        marks = macro(self.punctuation, start=1)
        casing = macro(self.casing)
        false = int(self.punctuation[0, 1:].sum())
        return {
            "punctuation": report(self.punctuation, MARKS),
            "casing": report(self.casing, CASES),
            "noninitial_casing": report(self.noninitial, CASES),
            "structure": report(self.structure, templates),
            "punctuation_macro_f1": marks,
            "casing_macro_f1": casing,
            "false_insertions_per_1000_words": 1000 * false / self.words if self.words else None,
            "score": 0.7 * marks + 0.3 * casing
            if marks is not None and casing is not None
            else None,
            "words": self.words,
            "exact_word_casing_accuracy": self.word_case_hits / self.word_case_support
            if self.word_case_support
            else None,
            "exact_word_casing_support": self.word_case_support,
        }


@torch.inference_mode()
def evaluate(
    model: Restorer,
    dataset: Windows,
    device: torch.device,
    limit: int | None = None,
    *,
    baseline: bool = False,
    fold: Literal["diagnostic", "calibration", "selection"] = "diagnostic",
    calibration: Callable[[Scores], Scores] | None = None,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    metrics = Metrics.create(device, len(model.templates))
    order = fold_indices(dataset, fold, limit)
    for start in range(0, len(order), model.geometry.context_tokens // 256):
        indices = order[start : start + model.geometry.context_tokens // 256]
        batch = collate(
            [dataset[int(index)] for index in indices], model.geometry.context_tokens
        ).to(device)
        if baseline:
            batch_size, words = batch.words.shape
            casing = torch.zeros((*batch.casing.shape, len(CASES)), device=device)
            casing[..., 0] = 1
            structures = torch.zeros((batch_size, words + 1, len(model.templates)), device=device)
            structures[..., 0] = 1
            scores = Scores(
                torch.full_like(batch.marks, -100, dtype=torch.float),
                torch.zeros((batch_size, words + 1, len(MARKS) - 1), device=device),
                casing,
                structures,
            )
        else:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = model.forward(batch)
            if calibration is not None:
                scores = calibration(scores)
        metrics.add(scores, batch)
    model.train(was_training)
    return {
        **metrics.finish(model.templates),
        "windows": len(order),
        "seed": 1618,
        "fold": fold,
        "document_ids": sorted({dataset.windows[int(index)][0] for index in order}),
        "reference_quality": "writer-derived silver; pretrained-seen",
        "predictor": "NONE/LOWER/@" if baseline else "restoration",
        "case_unit": "hyphen/apostrophe component",
        "structure_enabled": True,
    }
