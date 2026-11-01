"""Verify optimizer coverage and exact accumulated updates."""

import copy

import torch

from agbalu.restoration.configuration import Settings
from agbalu.restoration.dataset import Example, collate
from agbalu.restoration.model import Restorer
from agbalu.restoration.training import optimizer_for, rate_multiplier, update


def test_optimizer_covers_all_parameters_once(model: Restorer, settings: Settings) -> None:
    optimizer = optimizer_for(model, settings, torch.device("cpu"))
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in model.parameters()})


def test_accumulation_matches_full_batch(
    model: Restorer, settings: Settings, example: Example
) -> None:
    other = copy.deepcopy(model)
    left = optimizer_for(model, settings, torch.device("cpu"))
    right = optimizer_for(other, settings, torch.device("cpu"))
    update(
        model, left, [collate([example], 64), collate([example], 64)], settings, torch.device("cpu")
    )
    update(other, right, [collate([example, example], 64)], settings, torch.device("cpu"))
    for expected, observed in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(expected, observed, atol=1e-6, rtol=1e-5)


def test_schedule_starts_nonzero_and_finishes_at_floor(settings: Settings) -> None:
    assert rate_multiplier(0, 1000, settings) > 0
    assert rate_multiplier(999, 1000, settings) == settings.final_lr_fraction
