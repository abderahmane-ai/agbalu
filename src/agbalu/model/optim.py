"""LAMB, following `ltgoslo/gpt-bert` `pretraining/lamb.py`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from agbalu.model.config import ModelError


class Lamb(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            msg = f"learning rate must be positive, got {lr}"
            raise ModelError(msg)
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            msg = f"betas must lie in [0, 1), got {betas}"
            raise ModelError(msg)
        if eps <= 0.0:
            msg = f"eps must be positive, got {eps}"
            raise ModelError(msg)
        super().__init__(
            params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    msg = "Lamb does not support sparse gradients"
                    raise ModelError(msg)

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                corrected_1 = 1 - beta1 ** state["step"]
                corrected_2 = 1 - beta2 ** state["step"]
                update = (exp_avg / corrected_1) / (
                    (exp_avg_sq / corrected_2).sqrt() + group["eps"]
                )

                ratio = 1.0
                if group["weight_decay"] > 0:
                    update.add_(parameter, alpha=group["weight_decay"])
                    weight_norm = torch.norm(parameter.flatten())
                    update_norm = torch.norm(update.flatten())
                    if weight_norm > 0.0 and update_norm > 0.0:
                        ratio = float(weight_norm / update_norm)

                parameter.add_(update, alpha=-group["lr"] * ratio)

        return loss
