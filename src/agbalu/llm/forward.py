"""A packed row through the model: device transfer, then the hidden states.

The trainer and the evaluator call the same forward, so a number scored here is a number
measured on the path the model trained on. The classifier is applied by `agbalu.llm.loss`,
never here, so no full logit surface is materialised.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from agbalu.llm.packing import PackedRow

if TYPE_CHECKING:
    import torch


class HiddenOutput(Protocol):
    last_hidden_state: torch.Tensor


class Backbone(Protocol):
    """The decoder stack without the head: `model.model` of a causal LM."""

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        max_length_q: int,
        max_length_k: int,
        use_cache: bool,
    ) -> HiddenOutput: ...


class Head(Protocol):
    @property
    def weight(self) -> torch.Tensor: ...


class LanguageModel(Protocol):
    """The slice of a `transformers` causal LM that training and evaluation drive."""

    @property
    def model(self) -> Backbone: ...

    @property
    def lm_head(self) -> Head: ...

    @property
    def config(self) -> object: ...

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...

    def train(self, mode: bool = True) -> object: ...

    def eval(self) -> object: ...

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int,
        stop_strings: list[str],
        tokenizer: object,
    ) -> object: ...

    def save_pretrained(self, save_directory: str | Path) -> object: ...

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, object] | None = None
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class DeviceRow:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    cu_seqlens: torch.Tensor
    max_length: int
    targets: int


def to_device(row: PackedRow, device: torch.device, *, pinned: bool = False) -> DeviceRow:
    """Copy a host row to `device`; with `pinned`, the copies do not block the host."""
    import torch

    def move(array: object, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=dtype)
        if pinned:
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=pinned)

    return DeviceRow(
        input_ids=move(row.input_ids, torch.long),
        position_ids=move(row.position_ids, torch.long),
        labels=move(row.labels, torch.long),
        cu_seqlens=move(row.cu_seqlens, torch.int32),
        max_length=row.max_length,
        targets=row.targets,
    )


def hidden_states(backbone: Backbone, row: DeviceRow) -> torch.Tensor:
    """`[1, tokens, hidden]` for one packed row, every document attending only to itself.

    No cache: an eval-mode Qwen3.5 otherwise keeps a recurrent state per packed document in
    every Gated DeltaNet layer, about 2 MB each, which ran a full held-out row out of the
    47 GB slice on 2026-10-14.
    """
    return backbone(
        input_ids=row.input_ids,
        position_ids=row.position_ids,
        cu_seq_lens_q=row.cu_seqlens,
        cu_seq_lens_k=row.cu_seqlens,
        max_length_q=row.max_length,
        max_length_k=row.max_length,
        use_cache=False,
    ).last_hidden_state
