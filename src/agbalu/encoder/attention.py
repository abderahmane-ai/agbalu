"""Compute document-isolated disentangled attention."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from agbalu.encoder.configuration import Geometry
from agbalu.encoder.positions import bucket_indices
from agbalu.encoder.routing import RMSNorm

Score = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Layout:
    documents: Tensor
    positions: Tensor
    valid: Tensor
    block_mask: BlockMask | None = None


def make_layout(documents: Tensor, positions: Tensor, valid: Tensor, *, flex: bool) -> Layout:
    def allowed(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del batch, head
        length = documents.numel()
        q, k = query.clamp(max=length - 1), key.clamp(max=length - 1)
        real = valid[q] & valid[k] & (documents[q] == documents[k])
        safe_pad = (~valid[q]) & (query == key)
        return (query < length) & (key < length) & (real | safe_pad)

    mask = (
        create_block_mask(
            allowed,
            None,
            None,
            documents.numel(),
            documents.numel(),
            device=str(documents.device),
            BLOCK_SIZE=128,
            _compile=True,
        )
        if flex
        else None
    )
    return Layout(documents, positions, valid, mask)


def reference_attention(query: Tensor, key: Tensor, value: Tensor, score_mod: Score) -> Tensor:
    length = query.shape[-2]
    q = torch.arange(length, device=query.device)[:, None]
    k = torch.arange(length, device=query.device)[None, :]
    heads = torch.arange(query.shape[1], device=query.device)[:, None, None]
    # Autocast otherwise rounds the reference score matrix before the relative terms.
    with torch.autocast(query.device.type, enabled=False):
        score = query.float() @ key.float().transpose(-1, -2)
        score = score_mod(
            score, torch.zeros((), device=query.device, dtype=torch.long), heads, q, k
        )
        context = score.softmax(-1) @ value.float()
    return context.to(value.dtype)


class Attention(nn.Module):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        self.geometry = geometry
        width = geometry.hidden_size
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(geometry.head_size, geometry.epsilon)
        self.k_norm = RMSNorm(geometry.head_size, geometry.epsilon)
        self.fast = (
            torch.compile(flex_attention, dynamic=False)
            if geometry.backend == "flex"
            else flex_attention
        )

    def forward(self, hidden: Tensor, relative: tuple[Tensor, Tensor], layout: Layout) -> Tensor:
        config = self.geometry
        qkv: Tensor = self.qkv(hidden)
        q, k, v = qkv.view(-1, 3, config.heads, config.head_size).unbind(1)
        q = self.q_norm.forward(q).transpose(0, 1).unsqueeze(0)
        k = self.k_norm.forward(k).transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        position_q, position_k = relative
        # Preserve positional-score precision before reducing gradients to content precision.
        with torch.autocast(q.device.type, enabled=False):
            c2p = torch.matmul(q[0].float(), position_k.float().transpose(-1, -2))
            p2c = torch.matmul(k[0].float(), position_q.float().transpose(-1, -2))
        scale = 1 / math.sqrt(3 * config.head_size)
        documents, positions, valid = layout.documents, layout.positions, layout.valid

        def modify(score: Tensor, batch: Tensor, head: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
            del batch
            # FlexAttention evaluates padded tiles beyond the physical query/key length.
            length = documents.numel()
            q_index, k_index = qi.clamp(max=length - 1), ki.clamp(max=length - 1)
            distance = positions[q_index] - positions[k_index]
            forward = bucket_indices(distance, config.buckets, config.context_tokens)
            reverse = bucket_indices(-distance, config.buckets, config.context_tokens)
            combined = (score + c2p[head, q_index, forward] + p2c[head, k_index, reverse]) * scale
            allowed = valid[q_index] & valid[k_index] & (documents[q_index] == documents[k_index])
            allowed = allowed | ((~valid[q_index]) & (qi == ki))
            return torch.where(allowed, combined, float("-inf"))

        if config.backend == "flex":
            context = self.fast(
                q,
                k,
                v,
                score_mod=modify,
                block_mask=layout.block_mask,
                scale=1.0,
                kernel_options={
                    "BLOCK_M": 32,
                    "BLOCK_N": 32,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 32,
                    "BLOCK_M2": 32,
                    "BLOCK_N2": 32,
                    "num_stages": 1,
                },
            )
            if not isinstance(context, Tensor):
                message = "FlexAttention returned auxiliary output instead of a context tensor"
                raise TypeError(message)
        else:
            context = reference_attention(q, k, v, modify)
        merged = context[0].transpose(0, 1).reshape(-1, config.hidden_size)
        projected: Tensor = self.output(merged)
        return projected * valid.unsqueeze(-1)
