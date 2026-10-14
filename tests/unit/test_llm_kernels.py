"""The fast path refuses to start without its kernels; the HPC is where they exist."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from agbalu.llm import kernels


@pytest.mark.skipif(
    importlib.util.find_spec("fla") is not None, reason="fla is installed on this machine"
)
def test_a_missing_kernel_package_is_named() -> None:
    with pytest.raises(kernels.KernelError, match="fla"):
        kernels.require_fast_path()


@pytest.mark.skipif(
    importlib.util.find_spec("fla") is not None, reason="fla is installed on this machine"
)
def test_load_refuses_before_downloading_anything() -> None:
    with pytest.raises(kernels.KernelError):
        kernels.load("Qwen/does-not-exist", dtype=torch.bfloat16, device=torch.device("cpu"))


def test_the_attention_is_the_varlen_flash_kernel() -> None:
    assert kernels.ATTENTION == "kernels-community/vllm-flash-attn3"
