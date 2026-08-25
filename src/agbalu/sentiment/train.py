"""Fine-tune Masinissa for sentiment, over several seeds, and calibrate what is selected.

Three decisions, in the order they are taken. **Several seeds**, because a 12,000-row
fine-tune of a 31M encoder moves by more between seeds than most of the differences anyone
would report from it. **A uniform weight average of the runs** —
[Wortsman et al., ICML 2022](https://arxiv.org/abs/2203.05482) — taken only if it beats
every individual run on dev, so the average is a candidate and not an assumption. Then
**temperature and a conformal quantile**, fitted on dev and scored once on test.

Selection is on dev and test is read once. Choosing the epoch, the seed, or the soup on
test would report the maximum of thirty draws as though it were one.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import torch
from torch import Tensor, nn

from agbalu.bench.sentiment import (
    NUM_CLASSES,
    Item,
    Tokenizer,
    batches,
    confusion_of,
    encode,
    macro_f1_of,
)
from agbalu.sentiment.calibrate import (
    DEFAULT_ALPHA,
    CalibrationMetrics,
    ConformalPredictor,
    TemperatureScaler,
    evaluate_calibration,
)
from agbalu.sentiment.model import SequenceClassifier

log: Final = logging.getLogger("agbalu.sentiment")

DEFAULT_SEEDS: Final[tuple[int, ...]] = (42, 1337, 2026, 7, 999)


class TrainingError(Exception):
    """A fine-tune that cannot produce a checkpoint or a number."""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """How one seed is fitted, and how many are.

    Two learning rates, because the encoder arrives pretrained and the head does not: one
    rate that moves the head usefully would undo the representation.
    """

    epochs: int = 6
    batch_size: int = 64
    encoder_lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.10
    max_gradient: float = 1.0
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    alpha: float = DEFAULT_ALPHA


@dataclass(frozen=True, slots=True)
class Split:
    """One split, tokenised once and reused by every seed."""

    encoded: list[list[int]]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.encoded)


@dataclass(frozen=True, slots=True)
class SweepSummary:
    """What the seeds scored on dev, and which candidate the soup lost or won against."""

    seed_scores: dict[str, float]
    mean_dev_macro_f1: float
    stdev_dev_macro_f1: float
    soup_dev_macro_f1: float
    selected: str

    def as_dict(self) -> dict[str, object]:
        return {
            "seed_scores": self.seed_scores,
            "mean_dev_macro_f1": round(self.mean_dev_macro_f1, 6),
            "stdev_dev_macro_f1": round(self.stdev_dev_macro_f1, 6),
            "soup_dev_macro_f1": round(self.soup_dev_macro_f1, 6),
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The held-out result, and everything a card would have to disclose about it."""

    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    selected_dev_macro_f1: float
    calibration: CalibrationMetrics
    sweep: SweepSummary | None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "selected_dev_macro_f1": round(self.selected_dev_macro_f1, 6),
            "per_class": self.per_class,
            "confusion": self.confusion,
            "calibration": self.calibration.as_dict(),
        }
        if self.sweep is not None:
            payload["sweep"] = self.sweep.as_dict()
        return payload


class ParameterGroup(TypedDict):
    """One AdamW group. Typed, so a caller can read the rate a group was given."""

    params: list[nn.Parameter]
    lr: float
    weight_decay: float


def parameter_groups(model: SequenceClassifier, config: TrainConfig) -> list[ParameterGroup]:
    """Four groups: encoder and head, each split into decayed and undecayed.

    Norm gains, biases and the relative-position table are excluded from weight decay.
    Decaying a LayerNorm gain pulls it towards zero, which scales the activation it
    normalises towards zero with it.
    """
    undecayed = ("bias", "norm", "relative_embedding")
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("head", True): [],
        ("head", False): [],
        ("encoder", True): [],
        ("encoder", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = "head" if name.startswith("head.") else "encoder"
        groups[part, any(token in name for token in undecayed)].append(parameter)

    rates = {"head": config.head_lr, "encoder": config.encoder_lr}
    return [
        ParameterGroup(
            params=parameters,
            lr=rates[part],
            weight_decay=0.0 if no_decay else config.weight_decay,
        )
        for (part, no_decay), parameters in groups.items()
        if parameters
    ]


def average_states(states: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """The uniform weight average of several runs of the same architecture.

    Averaged in float and cast back, because a state dict holds integer buffers whose mean
    is not an integer and whose average is meaningless — identical across runs, so the cast
    returns them unchanged.
    """
    if not states:
        message = "there is nothing to average"
        raise TrainingError(message)
    count = len(states)
    return {
        key: (torch.stack([state[key].float() for state in states]).sum(dim=0) / count).to(
            states[0][key].dtype
        )
        for key in states[0]
    }


@dataclass(frozen=True, slots=True)
class Fitting:
    """One model, one pair of splits, one recipe — everything a seed varies against."""

    model: SequenceClassifier
    train: Split
    dev: Split
    config: TrainConfig
    device: torch.device

    def logits(self, split: Split) -> tuple[Tensor, Tensor]:
        """Raw, unscaled logits and gold labels for a whole split, in its own order."""
        self.model.eval()
        collected: list[Tensor] = []
        gold: list[Tensor] = []
        with torch.no_grad():
            for input_ids, mask, labels in batches(
                split.encoded,
                split.labels,
                batch_size=self.config.batch_size,
                order=list(range(len(split))),
                device=self.device,
            ):
                predicted, _ = self.model(input_ids, mask)
                collected.append(predicted.detach().float().cpu())
                gold.append(labels.detach().cpu())
        return torch.cat(collected), torch.cat(gold)

    def dev_macro_f1(self) -> float:
        predicted, gold = self.logits(self.dev)
        score, _ = macro_f1_of(confusion_of(gold.tolist(), predicted.argmax(dim=-1).tolist()))
        return score

    def epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        shuffler: random.Random,
    ) -> float:
        """One pass over the training split, returning the mean loss per example."""
        self.model.train()
        order = list(range(len(self.train)))
        shuffler.shuffle(order)
        total = 0.0
        for input_ids, mask, gold in batches(
            self.train.encoded,
            self.train.labels,
            batch_size=self.config.batch_size,
            order=order,
            device=self.device,
        ):
            optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(input_ids, mask, labels=gold)
            if loss is None:
                message = "the head returned no loss for a batch that carried labels"
                raise TrainingError(message)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * gold.numel()
        return total / len(self.train)

    def run(self, seed: int, initial: dict[str, Tensor]) -> tuple[float, dict[str, Tensor]]:
        """Fit from the same initialisation, returning the best dev score and its weights."""
        torch.manual_seed(seed)
        self.model.load_state_dict(initial)
        self.model.to(self.device)

        groups = [dict(group) for group in parameter_groups(self.model, self.config)]
        optimizer = torch.optim.AdamW(groups)
        steps = math.ceil(len(self.train) / self.config.batch_size) * self.config.epochs
        warmup = max(1, int(steps * self.config.warmup_fraction))

        def factor(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        shuffler = random.Random(seed)  # noqa: S311 — a seeded batch order, not a secret
        best_score = -1.0
        best_state: dict[str, Tensor] = {}

        for number in range(1, self.config.epochs + 1):
            loss = self.epoch(optimizer, scheduler, shuffler)
            score = self.dev_macro_f1()
            improved = score > best_score
            log.info(
                "seed=%d epoch=%d/%d train_loss=%.4f dev_macro_f1=%.4f%s",
                seed,
                number,
                self.config.epochs,
                loss,
                score,
                " best" if improved else "",
            )
            if improved:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if not best_state:
            message = f"seed {seed} completed no epoch, so nothing was selected"
            raise TrainingError(message)
        return best_score, best_state


def select(
    fitting: Fitting, scores: dict[int, float], states: dict[int, dict[str, Tensor]]
) -> tuple[dict[str, Tensor], float, SweepSummary | None]:
    """The soup if it beats the best single run on dev, and that run otherwise."""
    best_seed = max(scores, key=lambda seed: scores[seed])
    if len(states) == 1:
        return states[best_seed], scores[best_seed], None

    soup = average_states(list(states.values()))
    fitting.model.load_state_dict(soup)
    fitting.model.to(fitting.device)
    soup_score = fitting.dev_macro_f1()

    values = list(scores.values())
    log.info(
        "sweep mean=%.4f stdev=%.4f best=seed_%d(%.4f) soup=%.4f",
        statistics.fmean(values),
        statistics.pstdev(values),
        best_seed,
        scores[best_seed],
        soup_score,
    )

    wins = soup_score >= scores[best_seed]
    summary = SweepSummary(
        seed_scores={str(seed): round(score, 6) for seed, score in scores.items()},
        mean_dev_macro_f1=statistics.fmean(values),
        stdev_dev_macro_f1=statistics.pstdev(values),
        soup_dev_macro_f1=soup_score,
        selected="model_soup" if wins else f"seed_{best_seed}",
    )
    if wins:
        return soup, soup_score, summary
    return states[best_seed], scores[best_seed], summary


def train_classifier(
    model: SequenceClassifier,
    tokenizer: Tokenizer,
    splits: dict[str, list[Item]],
    *,
    device: torch.device,
    config: TrainConfig | None = None,
) -> tuple[EvaluationReport, TemperatureScaler, ConformalPredictor]:
    """Fit every seed, select on dev, calibrate on dev, and score test once."""
    recipe = config or TrainConfig()
    missing = [name for name in ("train", "dev", "test") if not splits.get(name)]
    if missing:
        message = f"no rows for {missing}; the splits have to be built first"
        raise TrainingError(message)
    if not recipe.seeds:
        message = "at least one seed is needed"
        raise TrainingError(message)

    prepared = {
        name: Split(encode(items, tokenizer), [item.label for item in items])
        for name, items in splits.items()
    }
    fitting = Fitting(model, prepared["train"], prepared["dev"], recipe, device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    scores: dict[int, float] = {}
    states: dict[int, dict[str, Tensor]] = {}
    for seed in recipe.seeds:
        scores[seed], states[seed] = fitting.run(seed, initial)

    selected, dev_score, sweep = select(fitting, scores, states)
    model.load_state_dict(selected)
    model.to(device)

    dev_logits, dev_gold = fitting.logits(prepared["dev"])
    scaler = TemperatureScaler()
    scaler.fit(dev_logits, dev_gold)
    conformal = ConformalPredictor()
    conformal.fit(scaler.probabilities(dev_logits), dev_gold, alpha=recipe.alpha)

    test_logits, test_gold = fitting.logits(prepared["test"])
    matrix = confusion_of(test_gold.tolist(), test_logits.argmax(dim=-1).tolist())
    macro_f1, per_class = macro_f1_of(matrix)
    report = EvaluationReport(
        accuracy=sum(matrix[index][index] for index in range(NUM_CLASSES)) / len(test_gold),
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=matrix,
        selected_dev_macro_f1=dev_score,
        calibration=evaluate_calibration(test_logits, test_gold, scaler, conformal),
        sweep=sweep,
    )
    log.info(
        "test accuracy=%.4f macro_f1=%.4f temperature=%.4f q_hat=%.4f coverage=%.4f",
        report.accuracy,
        report.macro_f1,
        report.calibration.temperature,
        report.calibration.q_hat,
        report.calibration.coverage,
    )
    return report, scaler, conformal


def save_checkpoint(directory: Path, model: SequenceClassifier, report: EvaluationReport) -> None:
    """The weights and the report, in the two shapes the release path reads them from.

    `model_state_dict` is `tools.export_checkpoint`'s first spelling, and `evaluation.json`
    is where `agbalu.sentiment.release` reads the calibration the published config carries.
    Nothing else is written: a second copy of a number is one of them going stale.
    """
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, directory / "model.pt")
    (directory / "evaluation.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("wrote the checkpoint and its report to %s", directory)
