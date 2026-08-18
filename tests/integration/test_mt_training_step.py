"""Two real training steps through the real `Seq2SeqTrainer`.

`tests/unit/test_modal_mt.py` pins what the collator emits against a fake. This pins the
thing that actually broke: the *stack* stopped supplying `decoder_input_ids` — transformers
5 dropped `prepare_decoder_input_ids_from_labels` from the M2M100 classes and
`label_smoothing_factor` pops `labels` before the model can shift them itself — and no
fake would have modelled that. It cost a container and reported the opposite defect.

The model is a two-layer M2M100 built from a config, so nothing is downloaded but the
tokenizer the collator pads with.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from modal_app.mt import Collator, normalise_accumulated_loss
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    M2M100Config,
    M2M100ForConditionalGeneration,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from agbalu.mt.finetune import SMALL_MODEL

pytestmark = pytest.mark.integration

ROWS = 64
"""Enough for two optimizer steps at four accumulated micro-batches of four: the loss
scales with the micro-batches a step actually accumulates, not with the argument."""


class Rows(torch.utils.data.Dataset[dict[str, list[int]]]):
    def __len__(self) -> int:
        return ROWS

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"input_ids": [5, 6, 7, 2], "attention_mask": [1, 1, 1, 1], "labels": [8, 9, 2]}


def _tiny() -> M2M100ForConditionalGeneration:
    config = M2M100Config(
        vocab_size=64,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=64,
    )
    return M2M100ForConditionalGeneration(config)


Collate = Callable[[list[dict[str, list[int]]]], dict[str, torch.Tensor]]


def _build(
    output: Path,
    collate: Callable[[M2M100ForConditionalGeneration], Collate],
    accumulation: int = 1,
) -> Seq2SeqTrainer:
    model = _tiny()
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=accumulation,
        max_steps=2,
        label_smoothing_factor=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
    )
    return Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=Rows(),
        data_collator=collate(model),
    )


def _train(output: Path, collate: Callable[[M2M100ForConditionalGeneration], Collate]) -> int:
    trainer = _build(output, collate)
    normalise_accumulated_loss(trainer)
    return int(trainer.train().global_step)


def _reported_loss(trainer: Seq2SeqTrainer) -> float:
    trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    return float(losses[-1])


def _collators(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[
    Callable[[M2M100ForConditionalGeneration], Collate],
    Callable[[M2M100ForConditionalGeneration], Collate],
]:
    def base(model: M2M100ForConditionalGeneration) -> Collate:
        collator: Collate = DataCollatorForSeq2Seq(tokenizer, model=model)
        return collator

    def ours(model: M2M100ForConditionalGeneration) -> Collate:
        pad, start = model.config.pad_token_id, model.config.decoder_start_token_id
        assert pad is not None
        assert start is not None
        return Collator(DataCollatorForSeq2Seq(tokenizer, model=model), pad, start)

    return base, ours


def test_the_collator_carries_a_step_that_the_base_one_cannot(tmp_path: Path) -> None:
    base, ours = _collators(AutoTokenizer.from_pretrained(SMALL_MODEL))

    with pytest.raises(ValueError, match="decoder_inputs_embeds"):
        _train(tmp_path / "base", base)

    assert _train(tmp_path / "ours", ours) == 2


def test_the_reported_loss_does_not_scale_with_accumulation(tmp_path: Path) -> None:
    """It scaled exactly 4x at four accumulation steps, and so did the gradient, which
    `max_grad_norm=1.0` then clipped into normalised gradient descent."""
    _, ours = _collators(AutoTokenizer.from_pretrained(SMALL_MODEL))

    single = _reported_loss(_build(tmp_path / "single", ours, accumulation=1))

    unfixed = _build(tmp_path / "unfixed", ours, accumulation=4)
    assert _reported_loss(unfixed) == pytest.approx(4 * single, rel=0.05)

    fixed = _build(tmp_path / "fixed", ours, accumulation=4)
    normalise_accumulated_loss(fixed)
    assert _reported_loss(fixed) == pytest.approx(single, rel=0.05)
