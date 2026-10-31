"""Compute learning and masking rates from consumed tokens."""

import math

from agbalu.encoder.configuration import Recipe


def learning_rate(tokens: int, recipe: Recipe) -> float:
    progress = min(max(tokens / recipe.total_tokens, 0), 1)
    warmup = recipe.warmup_fraction
    if progress < warmup:
        return recipe.peak_lr * max(progress, 1 / recipe.total_tokens) / warmup
    cosine = (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup))) / 2
    return recipe.peak_lr * (recipe.final_lr_fraction + (1 - recipe.final_lr_fraction) * cosine)


def mask_probability(tokens: int, total_tokens: int) -> float:
    return 0.3 - 0.1 * min(max(tokens / total_tokens / 0.7, 0), 1)


def context_length(tokens: int, recipe: Recipe) -> int:
    return 512 if tokens < recipe.total_tokens * recipe.long_context_fraction else 1024
