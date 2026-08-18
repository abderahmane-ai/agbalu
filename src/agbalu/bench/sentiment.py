"""Sentence classification over the frozen and the fine-tuned encoder.

Two settings, because they answer two different questions and only the pair is
informative. The **probe** is one `[hidden -> 3]` layer over a frozen encoder: it
measures what pretraining already put in the representation, and nothing else can be
credited for it. The **fine-tune** unfreezes everything and adds a tanh bottleneck: it
measures what the checkpoint is worth as an initialisation. A gap between the two is the
task's non-linearity; no gap would mean the head was doing the work.

Model selection is on dev, and test is scored once from the selected weights. That is not
a formality on a 1,500-row split: choosing the epoch on test would report the maximum of
fifteen draws as if it were one.

The labels are projected from the English side of a Tatoeba pair by a cross-lingual
classifier, so this measures agreement with that projection, not human judgement — the
same distinction that put the previously published Kabyle POS tagger at 63.69% on gold
against its own reported 94.8%. `docs/cards/kabsentiment.md` carries it.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import torch
from torch import Tensor, nn

from agbalu.model.config import ModelConfig
from agbalu.tokenizer.spec import CLS_ID, SEP_ID

log: Final = logging.getLogger("agbalu.bench")

SPLIT_DIR: Final = Path("data/tasks/sentiment")
LABEL_NAMES: Final[tuple[str, str, str]] = ("negative", "neutral", "positive")
NUM_CLASSES: Final = len(LABEL_NAMES)
MAX_PIECES: Final = 126
"""Leaves room for `[CLS]` and `[SEP]` inside a 128-piece window. The corpus is
sentence-level and gated at 25 English words, so nothing real reaches it."""

Setting = Literal["probe", "finetune"]


class SentimentError(Exception):
    """A benchmark run that cannot produce a number."""


class Tokenizer(Protocol):
    """The slice of the SentencePiece API this module drives, as in `bench.probe`."""

    def encode(self, text: str, out_type: type[int]) -> list[int]: ...


class Encoder(Protocol):
    config: ModelConfig

    def contextualise(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...


@dataclass(frozen=True, slots=True)
class Item:
    """One labelled sentence."""

    id: str
    text: str
    label: int


@dataclass(frozen=True, slots=True)
class System:
    """An encoder and the tokenizer it was pretrained with.

    Paired in a type because they are not independently choosable: piece ids are
    positions in this checkpoint's embedding table, and a mismatched pair produces a
    number rather than an error.
    """

    encoder: Encoder
    tokenizer: Tokenizer


@dataclass(frozen=True, slots=True)
class RunConfig:
    """How a setting is fitted.

    The defaults differ by setting because the two are not the same optimisation: a head
    over frozen features wants a large learning rate and many epochs, and an encoder that
    already has the representation wants neither.
    """

    epochs: int
    learning_rate: float
    batch_size: int = 64
    weight_decay: float = 0.01
    warmup_fraction: float = 0.10
    max_gradient: float = 1.0
    seed: int = 42
    """Fixes the head's initialisation *and* the batch order. Seeding only the second
    moved the POS probe by up to 0.5 points between refits of one checkpoint."""


PROBE: Final = RunConfig(epochs=15, learning_rate=2e-3)
FINETUNE: Final = RunConfig(epochs=6, learning_rate=3e-5)
SETTINGS: Final[dict[Setting, RunConfig]] = {"probe": PROBE, "finetune": FINETUNE}


@dataclass(frozen=True, slots=True)
class Report:
    """What one setting scored, and what it was selected on."""

    setting: Setting
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    selected_dev_macro_f1: float
    epochs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "setting": self.setting,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "per_class": self.per_class,
            "confusion": self.confusion,
            "selected_dev_macro_f1": round(self.selected_dev_macro_f1, 6),
            "epochs": self.epochs,
        }


def read_split(split: str, directory: Path = SPLIT_DIR) -> list[Item]:
    """A built split of `agbalu/KabSentiment`."""
    path = directory / f"{split}.jsonl"
    if not path.is_file():
        message = f"no {split} split at {path}"
        raise SentimentError(message)
    items: list[Item] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(Item(id=row["id"], text=row["text_kab"], label=int(row["label"])))
    return items


class Head(nn.Module):
    """`[hidden -> 3]`, or a tanh bottleneck before it when the encoder is trainable.

    One class rather than two, because the only thing that distinguishes the probe from
    the fine-tune is whether the encoder receives gradient — and a reader comparing two
    near-identical modules cannot see that.
    """

    def __init__(self, hidden_size: int, *, bottleneck: bool, dropout: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size) if bottleneck else None
        self.dropout = nn.Dropout(dropout) if bottleneck else None
        self.out_proj = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, pooled: Tensor) -> Tensor:
        hidden = pooled
        if self.dense is not None and self.dropout is not None:
            hidden = self.dropout(torch.tanh(self.dense(self.dropout(hidden))))
        projected: Tensor = self.out_proj(hidden)
        return projected


class Classifier(nn.Module):
    """Encoder plus head, mean-pooled over the real positions.

    Mean pooling rather than `[CLS]`: this encoder is masked-language-model pretrained
    with no next-sentence objective, so `[CLS]` carries no sentence-level signal that was
    ever trained for.
    """

    def __init__(self, encoder: Encoder, *, trainable: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.trainable = trainable
        for parameter in encoder.parameters():
            parameter.requires_grad = trainable
        self.head = Head(encoder.config.hidden_size, bottleneck=trainable)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            hidden = self.encoder.contextualise(input_ids, attention_mask)
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        logits: Tensor = self.head(pooled)
        return logits


def encode(items: Sequence[Item], tokenizer: Tokenizer) -> list[list[int]]:
    """Piece ids per item, bracketed and truncated to the encoder's window."""
    return [
        [CLS_ID, *tokenizer.encode(item.text, out_type=int)[:MAX_PIECES], SEP_ID] for item in items
    ]


def batches(
    encoded: Sequence[Sequence[int]],
    labels: Sequence[int],
    *,
    batch_size: int,
    order: Sequence[int],
    device: torch.device,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Padded batches in a caller-supplied order.

    The order is passed in rather than drawn here so that shuffling is the caller's
    seeded decision and an evaluation pass can be given the identity permutation.
    """
    for start in range(0, len(order), batch_size):
        window = [order[index] for index in range(start, min(start + batch_size, len(order)))]
        width = max(len(encoded[index]) for index in window)
        input_ids = torch.zeros((len(window), width), dtype=torch.long)
        mask = torch.zeros((len(window), width), dtype=torch.bool)
        for row, index in enumerate(window):
            pieces = encoded[index]
            input_ids[row, : len(pieces)] = torch.tensor(pieces, dtype=torch.long)
            mask[row, : len(pieces)] = True
        gold = torch.tensor([labels[index] for index in window], dtype=torch.long)
        yield input_ids.to(device), mask.to(device), gold.to(device)


def confusion_of(gold: Sequence[int], predicted: Sequence[int]) -> list[list[int]]:
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for actual, guess in zip(gold, predicted, strict=True):
        matrix[actual][guess] += 1
    return matrix


def macro_f1_of(matrix: Sequence[Sequence[int]]) -> tuple[float, dict[str, dict[str, float]]]:
    """Macro F1 and the per-class table it averages.

    Macro rather than micro because the split is balanced by construction, so micro F1
    would equal accuracy and report the same number twice.
    """
    per_class: dict[str, dict[str, float]] = {}
    scores: list[float] = []
    for index, name in enumerate(LABEL_NAMES):
        true_positives = matrix[index][index]
        predicted = sum(matrix[row][index] for row in range(NUM_CLASSES))
        actual = sum(matrix[index])
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / actual if actual else 0.0
        total = precision + recall
        f1 = 2 * precision * recall / total if total else 0.0
        per_class[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": actual,
        }
        scores.append(f1)
    return sum(scores) / len(scores), per_class


@torch.no_grad()
def predict(
    model: Classifier,
    encoded: Sequence[Sequence[int]],
    labels: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    gold: list[int] = []
    guessed: list[int] = []
    order = range(len(encoded))
    for input_ids, mask, batch_labels in batches(
        encoded, labels, batch_size=batch_size, order=list(order), device=device
    ):
        logits = model(input_ids, mask)
        guessed.extend(logits.argmax(dim=-1).tolist())
        gold.extend(batch_labels.tolist())
    return gold, guessed


def _schedule(config: RunConfig, total_steps: int) -> Callable[[int], float]:
    """Linear warmup then cosine decay, as a fraction of the schedule rather than a count.

    A fixed warmup makes a short run meaningless — 500 steps of warmup on a 20-step run
    trains at 4% of the peak rate — and these runs are short.
    """
    warmup = max(1, int(total_steps * config.warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return factor


def run(
    system: System,
    splits: dict[str, list[Item]],
    setting: Setting,
    *,
    device: torch.device,
    config: RunConfig | None = None,
) -> Report:
    """Fit one setting, select on dev, and score test once."""
    settings = config or SETTINGS[setting]
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)

    encoded = {name: encode(items, system.tokenizer) for name, items in splits.items()}
    labels = {name: [item.label for item in items] for name, items in splits.items()}

    model = Classifier(system.encoder, trainable=setting == "finetune").to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    steps_per_epoch = math.ceil(len(encoded["train"]) / settings.batch_size)
    factor = _schedule(settings, steps_per_epoch * settings.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    criterion = nn.CrossEntropyLoss()

    shuffler = random.Random(settings.seed)  # noqa: S311 — a fixed-seed shuffle, not a secret
    best_dev = -1.0
    best_state: dict[str, Tensor] = {}

    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = list(range(len(encoded["train"])))
        shuffler.shuffle(order)
        total = 0.0
        for input_ids, mask, gold in batches(
            encoded["train"],
            labels["train"],
            batch_size=settings.batch_size,
            order=order,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(input_ids, mask), gold)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, settings.max_gradient)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * gold.numel()

        dev_gold, dev_guess = predict(
            model,
            encoded["dev"],
            labels["dev"],
            batch_size=settings.batch_size,
            device=device,
        )
        dev_f1, _ = macro_f1_of(confusion_of(dev_gold, dev_guess))
        log.info(
            "setting=%s epoch=%d/%d train_loss=%.4f dev_macro_f1=%.4f%s",
            setting,
            epoch,
            settings.epochs,
            total / len(encoded["train"]),
            dev_f1,
            " best" if dev_f1 > best_dev else "",
        )
        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not best_state:
        message = f"setting {setting!r} completed no epoch, so nothing was selected"
        raise SentimentError(message)
    model.load_state_dict(best_state)
    test_gold, test_guess = predict(
        model, encoded["test"], labels["test"], batch_size=settings.batch_size, device=device
    )
    matrix = confusion_of(test_gold, test_guess)
    macro_f1, per_class = macro_f1_of(matrix)
    accuracy = sum(matrix[index][index] for index in range(NUM_CLASSES)) / len(test_gold)
    return Report(
        setting=setting,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=matrix,
        selected_dev_macro_f1=best_dev,
        epochs=settings.epochs,
    )
