"""Adapt only the selected decoder projections with rank-32 LoRA."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.codec_tts.backbone import Generator

PROJECTIONS: Final = frozenset(
    ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
)
RANK: Final = 32
SCALE: Final = 2.0


class AdaptedLinear(nn.Module):
    def __init__(self, original: nn.Linear) -> None:
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        self.factor_a = nn.Parameter(
            torch.empty(
                RANK, original.in_features, dtype=torch.float32, device=original.weight.device
            )
        )
        self.factor_b = nn.Parameter(
            torch.zeros(
                original.out_features, RANK, dtype=torch.float32, device=original.weight.device
            )
        )
        nn.init.kaiming_uniform_(self.factor_a, a=math.sqrt(5))
        self.is_enabled = True

    def forward(self, inputs: Tensor) -> Tensor:
        output: Tensor = self.original(inputs)
        if not self.is_enabled:
            return output
        adapted = functional.linear(
            functional.linear(inputs.to(self.factor_a.dtype), self.factor_a), self.factor_b
        )
        return output + (adapted * SCALE).to(output.dtype)

    def merged_weight(self) -> Tensor:
        delta = self.factor_b.float() @ self.factor_a.float() * SCALE
        return (self.original.weight.float() + delta).to(self.original.weight.dtype)


def inject_adapter(model: Generator) -> tuple[str, ...]:
    model.requires_grad_(False)
    selected: list[str] = []
    for name, layer in tuple(model.named_modules()):
        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in PROJECTIONS:
            parent_name, attribute = name.rsplit(".", 1)
            model.get_submodule(parent_name).set_submodule(attribute, AdaptedLinear(layer))
            selected.append(name)
    expected = model.geometry.layers * len(PROJECTIONS)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if len(selected) != expected or trainable != model.geometry.adapter_parameters:
        message = f"adapter contract failed: projections={len(selected)}/{expected}"
        message += f", parameters={trainable}/{model.geometry.adapter_parameters}"
        raise ValueError(message)
    return tuple(selected)


def adapter_state(model: Generator) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_adapter(model: Generator, state: Mapping[str, Tensor]) -> None:
    expected = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != set(expected):
        message = f"adapter keys mismatch: missing={set(expected) - set(state)}"
        message += f", extra={set(state) - set(expected)}"
        raise ValueError(message)
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all().item()):
                message = f"invalid adapter tensor {name}: shape={tensor.shape}"
                message += f", expected={parameter.shape}"
                raise ValueError(message)
            parameter.copy_(tensor)


def merge_adapter(model: Generator) -> None:
    for name, layer in tuple(model.named_modules()):
        if isinstance(layer, AdaptedLinear):
            with torch.no_grad():
                delta = layer.factor_b.float() @ layer.factor_a.float() * SCALE
                layer.original.weight.add_(delta.to(layer.original.weight.dtype))
            parent, attribute = name.rsplit(".", 1)
            model.get_submodule(parent).set_submodule(attribute, layer.original)


@contextmanager
def disabled_adapter(model: Generator) -> Iterator[None]:
    layers = [
        (layer, layer.is_enabled) for layer in model.modules() if isinstance(layer, AdaptedLinear)
    ]
    try:
        for layer, _ in layers:
            layer.is_enabled = False
        yield
    finally:
        for layer, previous in layers:
            layer.is_enabled = previous
