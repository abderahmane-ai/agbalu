"""Fine-tuning configuration and the dataset that feeds it.

The training loop itself is `transformers.Seq2SeqTrainer`: it already carries
resume-from-checkpoint, gradient accumulation and checkpoint retention, and a second
bespoke trainer beside `agbalu.model.trainer` would be two implementations of the same
failure modes.

What is not standard, and lives here:

- both directions of a pair are one example set, so the language tag is per example;
- the encoder's ids are the *trimmed* model's, via `vocab.Remap`;
- embeddings are freezable, because they are 42.7% of the 600M checkpoint and the
  vocabulary is already correct for Kabyle (UNK 0.011%), so there is little for them to
  learn and a great deal for them to overfit on 1,085,458 examples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, Protocol

from agbalu.mt.data import Direction

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import torch

    from agbalu.mt.data import Example
    from agbalu.mt.vocab import Remap

NLLB_CODE: Final[dict[str, str]] = {"kab": "kab_Latn", "eng": "eng_Latn", "fra": "fra_Latn"}

BASE_MODEL: Final = "facebook/nllb-200-distilled-1.3B"
"""Measured on the corrected FLORES+ devtest, chrF++, zero-shot:

    600M   kab-eng 39.70  eng-kab 27.51  kab-fra 37.30  fra-kab 26.35
    1.3B   kab-eng 44.60  eng-kab 31.53  kab-fra 42.00  fra-kab 30.21

Four to five points everywhere, and +5.0 BLEU on kab-eng. That is why the fine-tune starts
from the 1.3B despite it being the harder fit: the base model is worth more than anything
the recipe can recover. `SMALL_MODEL` is kept for the ablation that says how much."""

SMALL_MODEL: Final = "facebook/nllb-200-distilled-600M"

DEFAULT_RUN_NAME: Final = "nllb-kab-v2"
"""Names the run directory on the checkpoint volume, as `model.config.DEFAULT_RUN_NAME`
does for the encoder. The two must differ: both write under `/checkpoints/<run_name>`."""


@dataclass(frozen=True, slots=True)
class Config:
    model: str = BASE_MODEL
    max_source_length: int = 128
    """Applied to both sides. FLORES+ rows are single sentences and almost every pair
    fits; the tail is truncated rather than dropped so the pair count stays the reported
    one."""

    learning_rate: float = 2e-4
    """arXiv 2602.04442, single-language full fine-tuning of NLLB-200 on Turkic languages.
    The rate belongs to the effective batch below and does not survive being read off it:
    at a batch two orders of magnitude smaller, 2e-4 diverges."""

    warmup_ratio: float = 0.1
    """The one number here not taken from the source, whose warmup is quoted per epoch and
    is ambiguous against its step count."""
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    optimizer: Literal["adamw_torch", "adafactor"] = "adafactor"
    gradient_checkpointing: bool = True
    """Both exist to fit the 1.3B on a 24 GiB A10G, and neither is free.

    Trimmed to 52,209 tokens the model is 1,162M parameters. Adam would want 4 bytes of
    parameter, 4 of gradient and 8 of moments — 18.7 GiB before a single activation, which
    does not leave room. Adafactor factors the second moment into row and column vectors,
    taking the optimiser's share to roughly nothing and the total to ~9.4 GiB; gradient
    checkpointing then trades about 30% of the step time for most of the activation
    memory. On the 600M ablation both can be turned off."""

    batch_size: int = 32
    gradient_accumulation: int = 64
    """Effective batch 2,048, from arXiv 2602.04442's single-language recipe. The split is
    the only part a run can overturn: 32 x 128 tokens is the worst-case bucket against
    24 GiB, and halving `batch_size` while doubling `gradient_accumulation` preserves the
    effective batch exactly."""
    epochs: float = 2.0
    """The source's epoch count on a corpus of comparable size. At 2,048 this is ~1,060
    steps over 1.09M examples; early stopping on dev loss is what actually ends it."""

    eval_every: int = 50
    save_every: int = 100
    log_every: int = 10
    patience: int = 3
    eval_sentences: int = 500
    """Enough dev examples to rank checkpoints, few enough not to dominate the run.

    The cadence is set against the ~1,060 steps this recipe runs: `patience` counts
    evaluations, so `eval_every` must divide the run many times over or early stopping
    cannot fire at all. `save_every` stays a multiple of `eval_every`, which
    `load_best_model_at_end` requires.
    """

    label_smoothing_factor: float = 0.1
    seed: int = 0

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def small(self) -> Config:
        """The 600M ablation: it fits an A10G outright, so it drops both memory
        workarounds and takes the larger micro-batch back at the same effective batch."""
        return replace(
            self,
            model=SMALL_MODEL,
            optimizer="adamw_torch",
            gradient_checkpointing=False,
            batch_size=64,
            gradient_accumulation=32,
        )


class Tokenizer(Protocol):
    """The slice of the tokenizer API the dataset drives.

    `src_lang` and `tgt_lang` are assigned, not passed: NLLB's tokenizer takes the pair
    from its own attributes, which is why examples are grouped by direction at all.
    """

    src_lang: str
    tgt_lang: str

    def __call__(
        self, texts: list[str], *, text_target: list[str], max_length: int, truncation: bool
    ) -> Mapping[str, list[list[int]]]: ...


class Embeddings(Protocol):
    weight: torch.Tensor


class Adaptable(Protocol):
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def get_input_embeddings(self) -> Embeddings: ...


def language_of(direction: Direction, side: Literal["source", "target"]) -> str:
    source, target = direction.split("-")
    return source if side == "source" else target


def nllb_code(direction: Direction, side: Literal["source", "target"]) -> str:
    return NLLB_CODE[language_of(direction, side)]


class Dataset:
    """Tokenised examples, one direction each, with the language tags NLLB expects.

    `src_lang` is an attribute of the tokenizer rather than an argument to it, so examples
    are grouped by direction and the attribute is set once per group — setting it per
    example would be correct and roughly 40x slower.
    """

    def __init__(
        self,
        examples: list[Example],
        tokenizer: Tokenizer,
        config: Config,
        remap: Remap | None = None,
    ) -> None:
        self.features = _encode(examples, tokenizer, config, remap)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


def _encode(
    examples: list[Example],
    tokenizer: Tokenizer,
    config: Config,
    remap: Remap | None,
) -> list[dict[str, list[int]]]:
    grouped: dict[Direction, list[int]] = {}
    for position, example in enumerate(examples):
        grouped.setdefault(example.direction, []).append(position)

    features: list[dict[str, list[int]]] = [{} for _ in examples]
    for direction, positions in grouped.items():
        tokenizer.src_lang = nllb_code(direction, "source")
        tokenizer.tgt_lang = nllb_code(direction, "target")
        encoded = tokenizer(
            [examples[p].source for p in positions],
            text_target=[examples[p].target for p in positions],
            max_length=config.max_source_length,
            truncation=True,
        )
        for offset, position in enumerate(positions):
            feature = {
                "input_ids": encoded["input_ids"][offset],
                "attention_mask": encoded["attention_mask"][offset],
                "labels": encoded["labels"][offset],
            }
            if remap is not None:
                feature["input_ids"] = remap.to_new(feature["input_ids"])
                feature["labels"] = remap.to_new(feature["labels"])
            features[position] = feature

    return features


def freeze_embeddings(model: Adaptable) -> int:
    """Freeze the shared embedding table; return how many parameters that covers.

    NLLB ties input embeddings to the output projection, so freezing the table also
    freezes the softmax weights — which is the point: neither has anything to learn from
    Kabyle that its 52,209 reachable tokens do not already encode.
    """
    embedding = model.get_input_embeddings()
    embedding.weight.requires_grad = False
    return int(embedding.weight.numel())


def trainable_parameters(model: Adaptable) -> tuple[int, int]:
    """(trainable, total), for the line the run log opens with."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
