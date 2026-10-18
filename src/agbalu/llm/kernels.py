"""The Qwen3.5 fast path on Hopper, and the one patch packing needs.

Three facts measured on the HPC's H100 MIG slice on 2026-10-14 decide this module:

- transformers' `causal_conv1d_fn` for Qwen3.5 ignores `cu_seq_lens_q`, so the four-tap
  convolution before every Gated DeltaNet layer mixes the last tokens of one packed document
  into the first three of the next. The recurrent state then carries the leak through the
  whole document: +0.02 to +0.06 nats a token on average, up to 2.4 on single tokens.
  `fla`'s Triton convolution takes `cu_seqlens`; with it, packed and unpacked losses agree
  to 0.0000 nats.
- Triton 3.4 to 3.7.0 computes the gated delta rule's backward wrongly on Hopper (fla issue
  640). torch 2.11 pins Triton 3.6, so fla must route that kernel to TileLang, which needs
  an `nvcc` and a host C++ compiler at runtime.
- `sdpa` ignores the varlen kwargs, so the full-attention layers attend across documents.
  FlashAttention-3 from the Hub reads them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import torch

    from agbalu.llm.forward import LanguageModel

ATTENTION: Final = "kernels-community/vllm-flash-attn3"


class KernelError(Exception):
    """The fast path this run depends on is not installed."""


def packed_causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """transformers' `causal_conv1d_fn` contract, with document boundaries respected.

    Takes and returns `[batch, channels, time]`; `fla` works in `[batch, time, channels]`.
    """
    from fla.modules.convolution import causal_conv1d

    boundaries = kwargs.get("cu_seq_lens_q")
    out = causal_conv1d(
        hidden_states.transpose(1, 2),
        weight=weight,
        bias=bias,
        activation=activation,
        cu_seqlens=boundaries,
    )
    result = out[0] if isinstance(out, tuple) else out
    transposed: torch.Tensor = result.transpose(1, 2)
    return transposed


def install_packed_convolution() -> None:
    """Route every Qwen3.5 Gated DeltaNet layer's convolution through `fla`.

    The module-level name is replaced, which is what `Qwen3_5GatedDeltaNet.forward` looks up
    at call time.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5

    modeling_qwen3_5.causal_conv1d_fn = packed_causal_conv1d


def require_fast_path() -> None:
    """Refuse to train on the slow or wrong path rather than discover it in the loss."""
    missing = [
        name for name in ("fla", "tilelang", "kernels") if importlib.util.find_spec(name) is None
    ]
    if missing:
        message = f"{missing} not installed; build the environment from requirements/hpc.lock"
        raise KernelError(message)
    from fla.utils._compat import has_usable_nvcc

    if not has_usable_nvcc():
        message = "no usable nvcc for TileLang; install nvidia-cuda-nvcc 13.0 from the lock"
        raise KernelError(message)


def load(path: str | Path, *, dtype: torch.dtype, device: torch.device) -> LanguageModel:
    """A causal LM on the packed fast path, refusing to load onto any other."""
    from transformers import AutoModelForCausalLM

    require_fast_path()
    install_packed_convolution()
    model: LanguageModel = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=dtype, attn_implementation=ATTENTION, device_map=str(device)
    )
    return model
