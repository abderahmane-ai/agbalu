"""Evaluation metrics and benchmark harness for Kabyle document OCR (Feraoun).

Calculates Character Error Rate (CER), Word Error Rate (WER), and exact diacritic precision
over Kabyle extended characters and sub-dots.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import torch
from torch.utils.data import DataLoader

from agbalu.ocr.models import VisionEncoderDecoder
from agbalu.ocr.vocabulary import KABYLE_EXTENDED, KABYLE_SUBDOTS

KABYLE_SPECIAL_GLYPHS: Final[frozenset[str]] = frozenset(KABYLE_EXTENDED + KABYLE_SUBDOTS)


class EvaluationError(ValueError):
    """A population that cannot carry the metric asked of it."""


def levenshtein_distance(s1: Sequence[str] | str, s2: Sequence[str] | str) -> int:
    """Compute standard Levenshtein edit distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def align(reference: str, hypothesis: str) -> list[tuple[str | None, str | None]]:
    """Character alignment behind the edit distance, as (reference, hypothesis) pairs.

    `None` on the reference side is an insertion and `None` on the hypothesis side a
    deletion. This is what separates *placing* a diacritic from *counting* one: a line with
    the right number of `ḍ` in the wrong positions scores as errors, not as matches.
    """
    rows, columns = len(reference), len(hypothesis)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        costs[i][0] = i
    for j in range(columns + 1):
        costs[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            substitution = costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            costs[i][j] = min(costs[i - 1][j] + 1, costs[i][j - 1] + 1, substitution)

    pairs: list[tuple[str | None, str | None]] = []
    i, j = rows, columns
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution = costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            if costs[i][j] == substitution:
                pairs.append((reference[i - 1], hypothesis[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
            continue
        pairs.append((None, hypothesis[j - 1]))
        j -= 1
    pairs.reverse()
    return pairs


def compute_cer(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Total character edits over total reference characters."""
    total_edits = 0
    total_chars = 0

    for hyp, ref in zip(hypotheses, references, strict=True):
        total_edits += levenshtein_distance(hyp, ref)
        total_chars += len(ref)

    if total_chars == 0:
        message = "no reference characters to divide by"
        raise EvaluationError(message)
    return total_edits / total_chars


def compute_wer(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Total word edits over total reference words."""
    total_edits = 0
    total_words = 0

    for hyp, ref in zip(hypotheses, references, strict=True):
        ref_words = ref.strip().split()
        total_edits += levenshtein_distance(hyp.strip().split(), ref_words)
        total_words += len(ref_words)

    if total_words == 0:
        message = "no reference words to divide by"
        raise EvaluationError(message)
    return total_edits / total_words


def compute_diacritic_f1(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Positional precision, recall and F1 over Kabyle extended and sub-dot glyphs.

    Scored on the alignment, not on per-line counts: a bag-of-characters comparison scores a
    hypothesis holding the right glyphs in the wrong places as perfect.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for hyp, ref in zip(hypotheses, references, strict=True):
        for ref_char, hyp_char in align(ref, hyp):
            wanted = ref_char in KABYLE_SPECIAL_GLYPHS if ref_char is not None else False
            produced = hyp_char in KABYLE_SPECIAL_GLYPHS if hyp_char is not None else False
            if wanted and ref_char == hyp_char:
                true_positives += 1
                continue
            if wanted:
                false_negatives += 1
            if produced:
                false_positives += 1

    predicted = true_positives + false_positives
    gold = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_gold_special_glyphs": gold,
    }


def compute_exact_match(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Compute line Exact Match (EM) accuracy."""
    if not references:
        return 0.0
    matches = sum(1 for h, r in zip(hypotheses, references, strict=True) if h.strip() == r.strip())
    return matches / len(references)


@torch.no_grad()
def evaluate_ocr_model(
    model: VisionEncoderDecoder,
    dataloader: DataLoader[object],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a Vision-Encoder-Decoder model and return comprehensive benchmark metrics."""
    model.eval()
    model.to(device)

    all_hypotheses: list[str] = []
    all_references: list[str] = []
    total_val_loss = 0.0
    loss_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not isinstance(batch, dict):
            continue
        pixel_values = batch["pixel_values"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
            out = model(pixel_values=pixel_values, labels=labels)
            if out.loss is not None:
                total_val_loss += out.loss.item()
                loss_batches += 1

        texts = batch["texts"]
        preds = model.generate(pixel_values=pixel_values)
        all_hypotheses.extend(preds)
        all_references.extend(texts)

    cer = compute_cer(all_hypotheses, all_references)
    wer = compute_wer(all_hypotheses, all_references)
    em = compute_exact_match(all_hypotheses, all_references)
    diacritic_metrics = compute_diacritic_f1(all_hypotheses, all_references)
    val_loss = total_val_loss / max(loss_batches, 1)

    samples = list(zip(all_references[:3], all_hypotheses[:3], strict=False))

    return {
        "cer": cer,
        "wer": wer,
        "exact_match": em,
        "val_loss": val_loss,
        "diacritic_precision": diacritic_metrics["precision"],
        "diacritic_recall": diacritic_metrics["recall"],
        "diacritic_f1": diacritic_metrics["f1"],
        # The support behind the F1. A rate over a handful of glyphs is a lottery.
        "diacritic_support": diacritic_metrics["total_gold_special_glyphs"],
        "num_evaluated_lines": len(all_references),
        "samples": samples,
    }
