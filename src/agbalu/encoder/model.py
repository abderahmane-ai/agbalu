"""Compose the encoder with block attention residuals and a tied MLM head."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.encoder.attention import Attention, Layout, make_layout
from agbalu.encoder.configuration import MATRIX_AXES, Geometry
from agbalu.encoder.routing import DepthAttention, RMSNorm


class Layer(nn.Module):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        width = geometry.hidden_size
        self.attention = Attention(geometry)
        self.attention_route = DepthAttention(width, geometry.epsilon)
        self.feedforward_route = DepthAttention(width, geometry.epsilon)
        self.attention_norm = RMSNorm(width, geometry.epsilon)
        self.feedforward_norm = RMSNorm(width, geometry.epsilon)
        self.gate_up = nn.Linear(width, 2 * geometry.intermediate_size, bias=False)
        self.down = nn.Linear(geometry.intermediate_size, width, bias=False)
        self.dropout = nn.Dropout(geometry.dropout)

    def forward(
        self,
        blocks: list[Tensor],
        partial: Tensor | None,
        relative: tuple[Tensor, Tensor],
        layout: Layout,
    ) -> Tensor:
        mixed = self.attention_route.forward(blocks, partial)
        delta = self.dropout(
            self.attention.forward(self.attention_norm.forward(mixed), relative, layout)
        )
        partial = delta if partial is None else partial + delta
        mixed = self.feedforward_route.forward(blocks, partial)
        projected: Tensor = self.gate_up(self.feedforward_norm.forward(mixed))
        gate, value = projected.chunk(2, -1)
        change: Tensor = self.down(functional.silu(gate) * value)
        result: Tensor = partial + self.dropout(change)
        return result


class Encoder(nn.Module):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        self.config = geometry
        self.embedding = nn.Embedding(geometry.vocab_size, geometry.hidden_size)
        self.relative = nn.Parameter(torch.empty(2 * geometry.buckets - 1, geometry.hidden_size))
        self.position_q = nn.Linear(geometry.hidden_size, geometry.hidden_size, bias=False)
        self.position_k = nn.Linear(geometry.hidden_size, geometry.hidden_size, bias=False)
        self.layers = nn.ModuleList(Layer(geometry) for _ in range(geometry.layers))
        self.final_route = DepthAttention(geometry.hidden_size, geometry.epsilon)
        self.final_norm = RMSNorm(geometry.hidden_size, geometry.epsilon)
        self.head = nn.Linear(geometry.hidden_size, geometry.hidden_size, bias=False)
        self.head_norm = RMSNorm(geometry.hidden_size, geometry.epsilon)
        self.vocabulary_bias = nn.Parameter(torch.zeros(geometry.vocab_size))
        self.dropout = nn.Dropout(geometry.dropout)
        self.compiled_features: Callable[[Tensor, Layout], Tensor] | None = None
        standard_deviation = math.sqrt(2 / (5 * geometry.hidden_size))
        for parameter in self.parameters():
            if parameter.ndim >= MATRIX_AXES:
                nn.init.trunc_normal_(
                    parameter,
                    std=standard_deviation,
                    a=-2 * standard_deviation,
                    b=2 * standard_deviation,
                )
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != geometry.parameters:
            message = f"instantiated parameters {actual} != ledger {geometry.parameters}"
            raise ValueError(message)

    def contextualise_packed(
        self,
        ids: Tensor,
        layout: Layout,
        observer: Callable[[str, Tensor], None] | None = None,
    ) -> Tensor:
        embedded: Tensor = self.embedding(ids)
        blocks = [self.dropout(embedded)]
        partial: Tensor | None = None
        head_shape = (2 * self.config.buckets - 1, self.config.heads, self.config.head_size)
        pq: Tensor = self.position_q(self.relative)
        pk: Tensor = self.position_k(self.relative)
        relative = (pq.view(head_shape).transpose(0, 1), pk.view(head_shape).transpose(0, 1))
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                message = f"encoder layer {index} has unexpected type {type(layer).__name__}"
                raise TypeError(message)
            if observer is not None:
                values = torch.stack([*blocks, *([] if partial is None else [partial])])
                weights = layer.attention_route.weights(values)
                entropy = -(weights * weights.clamp(min=1e-12).log()).sum(0).mean()
                observer("routing_entropy", entropy)
            partial = layer.forward(blocks, partial, relative, layout)
            if observer is not None:
                observer("activation_rms", partial.float().square().mean().sqrt())
            if (index + 1) % self.config.block_layers == 0:
                blocks.append(partial)
                partial = None
        return self.final_norm.forward(self.final_route.forward(blocks, partial))

    def contextualise(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        batch, length = input_ids.shape
        documents = torch.arange(batch, device=input_ids.device).repeat_interleave(length)
        positions = torch.arange(length, device=input_ids.device).repeat(batch)
        layout = make_layout(
            documents,
            positions,
            attention_mask.flatten().bool(),
            flex=self.config.backend == "flex",
        )
        return self.contextualise_packed(input_ids.flatten(), layout).reshape(batch, length, -1)

    def logits(self, hidden: Tensor) -> Tensor:
        projected: Tensor = self.head(hidden)
        transformed = self.head_norm.forward(functional.gelu(projected))
        return functional.linear(transformed, self.embedding.weight, self.vocabulary_bias)

    def forward(self, ids: Tensor, layout: Layout, labels: Tensor) -> tuple[Tensor, Tensor]:
        hidden = (
            self.contextualise_packed(ids, layout)
            if self.compiled_features is None
            else self.compiled_features(ids, layout)
        )
        selected = labels >= 0
        logits = self.logits(hidden[selected])
        loss = functional.cross_entropy(logits.float(), labels[selected], reduction="sum")
        correct = (logits.argmax(-1) == labels[selected]).sum()
        return loss, correct
