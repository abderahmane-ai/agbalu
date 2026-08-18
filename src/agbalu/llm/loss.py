"""Cross-entropy over a large vocabulary, without materialising the logits (task 11.6).

The base's vocabulary is 248,320, so one token's fp32 logit row is 0.95 MiB and a
micro-batch of 4 × 1024 costs 3.9 GB to upcast — twice, because the causal shift forces a
contiguous copy. That transient, not the weights, is what caps the micro-batch, and the
micro-batch is what holds the A10 at 16.9% of its peak.

Cut Cross-Entropy (`arXiv 2411.09009`, ICLR 2025) computes the loss from the hidden states
and the classifier matrix directly, evaluating the log-sum-exp in flash memory: measured
24 GB → 1 MB at a comparable vocabulary, at no cost to speed or convergence. This module is
the one place that decision is expressed, so the trainer asks for a loss and never for
logits.

Two properties are not negotiable, and `tests/` pins both:

- **A softcap, where the config declares one, travels with the loss.** A fused kernel that
  skipped it would return a plausible number for a different model.
- **A CUDA path never silently falls back.** Materialising the logits on the GPU is the
  defect being removed; doing it quietly under a name that promises otherwise would return
  the OOM this module exists to prevent, with nothing in the log to say why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import torch

IGNORE_INDEX: Final = -100
"""What the loss skips. Shared with `baseline`, which scores with the same convention."""


class LossError(Exception):
    """The loss cannot be computed as configured."""


def available() -> bool:
    """Whether the fused kernel can be imported.

    Ampere or newer plus Triton 3.0, so it is present in the container and absent on the
    laptop. `tests/unit/test_modal_image.py` records it as a runtime dependency, because an
    AST closure over our own code cannot see that the import is optional.
    """
    try:
        import cut_cross_entropy  # noqa: F401
    except ImportError:
        return False
    return True


def reference_loss(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """The definition, materialising the logits. The fused path is checked against this.

    `hidden[t]` predicts `targets[t + 1]`, so the last position has nothing to predict and
    the first target has nothing to predict it — the same shift `transformers` applies, and
    the same one `shift=True` asks the fused kernel for.

    Memory is quadratic in the vocabulary here by construction. It exists to define the
    answer, not to train with.
    """
    import torch

    if hidden.shape[:-1] != targets.shape:
        message = f"hidden {tuple(hidden.shape[:-1])} does not match targets {tuple(targets.shape)}"
        raise LossError(message)
    if hidden.shape[-1] != classifier.shape[-1]:
        message = (
            f"hidden width {hidden.shape[-1]} does not match "
            f"classifier width {classifier.shape[-1]}"
        )
        raise LossError(message)

    logits = (hidden[..., :-1, :] @ classifier.T).float()
    if softcap is not None:
        if softcap <= 0:
            message = f"softcap must be positive, got {softcap}"
            raise LossError(message)
        logits = softcap * torch.tanh(logits / softcap)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, -2),
        targets[..., 1:].flatten(),
        ignore_index=ignore_index,
        reduction="mean",
    )


def sequence_loss(
    hidden: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    *,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mean next-token cross-entropy, fused when the kernel is available.

    On CUDA the fused kernel is required rather than preferred: falling back would rebuild
    the multi-gigabyte fp32 surface this module exists to avoid, and would do it invisibly.
    Off CUDA — the laptop, the test suite — the reference is the only path and its memory
    cost is bounded by the test's own shapes.
    """
    if not available():
        if hidden.is_cuda:
            message = (
                "cut-cross-entropy is not importable, and the reference loss materialises "
                f"a {classifier.shape[0]:,}-wide fp32 logit surface that will not fit. "
                "Install `cut-cross-entropy` in the training image."
            )
            raise LossError(message)
        return reference_loss(
            hidden, classifier, targets, softcap=softcap, ignore_index=ignore_index
        )

    from cut_cross_entropy import linear_cross_entropy

    return require_tensor(
        linear_cross_entropy(
            hidden,
            classifier,
            targets,
            ignore_index=ignore_index,
            softcap=softcap,
            reduction="mean",
            shift=True,
        )
    )


def require_tensor(value: object) -> torch.Tensor:
    """Narrow the fused kernel's return, which its own annotation leaves open.

    `cce_linear_cross_entropy` returns `tuple[Tensor, Tensor | None]` and the public wrapper
    returns the loss alone unless `return_lse` is set. Asserting that here turns a change in
    that contract into a named error at the call site, rather than an optimizer stepping on
    something that is not a loss.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        message = (
            f"the fused loss returned {type(value).__name__}, not a tensor; "
            f"if it now returns a tuple, `return_lse` is the flag that decides it"
        )
        raise LossError(message)
    return value


def logit_softcap(config: object) -> float | None:
    """The model's own softcap, or `None` where it has none.

    Read off the config rather than hardcoded. Qwen3.5 declares none; a constant here would
    be a second copy of a number the checkpoint already carries.
    """
    value = getattr(config, "final_logit_softcapping", None)
    if value is None:
        return None
    return float(value)
