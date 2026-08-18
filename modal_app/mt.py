"""Fine-tune NLLB for Kabyle on a Modal GPU (task 4.4).

Three functions, in the order they are meant to run: `trim` builds the reduced-vocabulary
checkpoint once, `finetune` trains against it, and the resulting model is scored by the
existing task 7.7 harness in `modal_app.bench` so a fine-tune and a baseline are never
compared under two different protocols.

Checkpoints go to the checkpoint volume and `Seq2SeqTrainer` resumes from the newest one,
so a preempted container costs the steps since the last save and nothing else.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from agbalu.mt.finetune import DEFAULT_RUN_NAME as MT_RUN_NAME
from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    MODELS_PATH,
    MT_TIMEOUT,
    MT_VOLUMES,
    RETRIES,
    app,
    checkpoint_volume,
    data_volume,
    models_volume,
    mt_image,
)

if TYPE_CHECKING:
    import torch

GPU: Final = "A10"
"""24 GiB, and the only card available. The trimmed 1.3B fits it on Adafactor with
gradient checkpointing — see `agbalu.mt.finetune.Config` for the arithmetic."""

LOCAL_DATA: Final = Path("data/processed/mt")

REMOTE_DATA: Final = "mt"
REMOTE_TRIMMED: Final = "trimmed"

log: Final = logging.getLogger("agbalu.mt")

LABEL_PAD: Final = -100
"""`DataCollatorForSeq2Seq.label_pad_token_id`, which is also the loss's ignore index."""


def _slug(repo: str) -> str:
    """A directory name per base model, so the 1.3B's trim and the 600M ablation's do not
    overwrite each other on the volume."""
    return repo.replace("/", "__")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )


class Collate(Protocol):
    """`transformers.DataCollatorForSeq2Seq`, structurally, so `Collator` is testable."""

    def __call__(self, features: list[dict[str, list[int]]], /) -> dict[str, torch.Tensor]: ...


def shift_right(labels: torch.Tensor, pad_token_id: int, start_token_id: int) -> torch.Tensor:
    """Decoder inputs from labels: everything one place right, `start_token_id` first, and
    the collator's label padding back to `pad_token_id` so the embedding can index it."""
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = start_token_id
    return shifted.masked_fill(shifted == LABEL_PAD, pad_token_id)


class LossScaled(Protocol):
    model_accepts_loss_kwargs: bool


def normalise_accumulated_loss(trainer: LossScaled) -> None:
    """Divide the accumulated loss by the accumulation count again.

    `Trainer` skips that division whenever the model accepts loss kwargs, on the
    assumption that `compute_loss` already normalised by `num_items_in_batch` — which it
    does not do on the label-smoothing path, because it never passes that count to the
    smoother. Both the reported loss and the gradient then come out `gradient_accumulation`
    times too large, measured at exactly 1x, 4x and 16x for those accumulation counts, and
    `max_grad_norm` turns every step into normalised gradient descent. This is the opt-out
    `Trainer.compute_loss` documents. The cost is that the accumulated batch becomes a mean
    of micro-batch means rather than token-weighted.
    """
    trainer.model_accepts_loss_kwargs = False


class Collator:
    """The base collator plus the `decoder_input_ids` nothing else supplies.

    transformers 5 dropped `prepare_decoder_input_ids_from_labels` from the M2M100
    classes, so `DataCollatorForSeq2Seq` no longer builds them; and a non-zero
    `label_smoothing_factor` makes `Trainer.compute_loss` pop `labels` before the model
    could shift them itself. The decoder then receives neither, which M2M100's guard
    reports as "You cannot specify both decoder_input_ids and decoder_inputs_embeds".
    """

    def __init__(self, inner: Collate, pad_token_id: int, start_token_id: int) -> None:
        self._inner = inner
        self._pad = pad_token_id
        self._start = start_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = dict(self._inner(features))
        if "decoder_input_ids" not in batch:
            batch["decoder_input_ids"] = shift_right(batch["labels"], self._pad, self._start)
        return batch


@app.function(image=mt_image, volumes=MT_VOLUMES, timeout=MT_TIMEOUT, retries=RETRIES)
def trim(small: bool = False, force: bool = False) -> dict[str, object]:
    """Build the reduced-vocabulary checkpoint and the id map beside it.

    CPU-only on purpose: loading the checkpoint, tokenising the corpus for coverage and
    gathering embedding rows are all host work, and none of it touches a GPU.

    Written once per base model and reused, because the trim depends only on the corpus
    and that model. `force` rebuilds it after either changes.
    """
    _configure_logging()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from agbalu.mt.data import read
    from agbalu.mt.finetune import Config
    from agbalu.mt.vocab import coverage, read_keep, trim_model, write_keep

    repo = (Config().small() if small else Config()).model
    target = Path(MODELS_PATH) / REMOTE_TRIMMED / _slug(repo)
    if not force and (target / "keep.json").is_file():
        kept = read_keep(target / "keep.json")
        log.info("reusing the trim at %s | %s tokens", target, f"{len(kept):,}")
        return {"repo": repo, "kept": len(kept), "path": str(target), "rebuilt": False}

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)

    texts: list[str] = []
    for split in ("train", "dev"):
        for example in read(Path(DATA_PATH) / REMOTE_DATA / f"{split}.jsonl"):
            texts.append(example.source)
            texts.append(example.target)

    found = coverage(texts, tokenizer)
    log.info(
        "coverage %s of %s tokens (%.1f%%) over %s strings",
        f"{found.kept:,}",
        f"{found.original_size:,}",
        100 * found.share,
        f"{found.scanned:,}",
    )

    before = sum(p.numel() for p in model.parameters())
    trim_model(model, found.keep)
    after = sum(p.numel() for p in model.parameters())
    log.info("parameters %s -> %s", f"{before:,}", f"{after:,}")

    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    write_keep(found, target / "keep.json")
    models_volume.commit()

    return {
        "repo": repo,
        "kept": found.kept,
        "original_size": found.original_size,
        "parameters_before": before,
        "parameters_after": after,
        "path": str(target),
        "rebuilt": True,
    }


@app.function(image=mt_image, gpu=GPU, volumes=MT_VOLUMES, timeout=MT_TIMEOUT, retries=RETRIES)
def finetune(
    run_name: str = MT_RUN_NAME,
    freeze_embeddings: bool = False,
    trimmed: bool = True,
    max_steps: int = 0,
    small: bool = False,
) -> dict[str, object]:
    """Adapt NLLB on the four Kabyle directions. Resumes from the newest checkpoint.

    Stopping is on `eval_loss`, not chrF++: the dev set mixes all four directions and
    generation needs one `forced_bos_token_id` per direction, so a single generated
    evaluation over it would decode three quarters of the sentences into the wrong
    language. chrF++ is what `modal_app.bench` reports afterwards, on FLORES+, under the
    task 7.7 protocol.
    """
    _configure_logging()
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from agbalu.mt.data import read
    from agbalu.mt.finetune import (
        Config,
        Dataset,
        trainable_parameters,
    )
    from agbalu.mt.finetune import (
        freeze_embeddings as freeze,
    )
    from agbalu.mt.vocab import Remap, read_keep

    config = Config().small() if small else Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    source = Path(MODELS_PATH) / REMOTE_TRIMMED / _slug(config.model) if trimmed else config.model
    if trimmed and not Path(source).is_dir():
        message = f"{source} missing; run the trim step first"
        raise RuntimeError(message)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(source))
    remap = Remap(read_keep(Path(source) / "keep.json")) if trimmed else None

    if freeze_embeddings:
        frozen = freeze(model)
        log.info("froze %s embedding parameters", f"{frozen:,}")
    trainable, total = trainable_parameters(model)
    log.info(
        "run %s | %s | %s trainable of %s parameters | vocab %s",
        run_name,
        config.model,
        f"{trainable:,}",
        f"{total:,}",
        f"{model.config.vocab_size:,}",
    )

    class Rows(torch.utils.data.Dataset[dict[str, list[int]]]):
        """`Seq2SeqTrainer` types its datasets as `torch.utils.data.Dataset`. Keeping the
        subclass here rather than in `agbalu.mt.finetune` is what lets that module — and
        its tests — stay importable without torch."""

        def __init__(self, features: list[dict[str, list[int]]]) -> None:
            self.features = features

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.features[index]

    train = Rows(
        Dataset(
            read(Path(DATA_PATH) / REMOTE_DATA / "train.jsonl"), tokenizer, config, remap
        ).features
    )
    dev_examples = read(Path(DATA_PATH) / REMOTE_DATA / "dev.jsonl")[: config.eval_sentences]
    dev = Rows(Dataset(dev_examples, tokenizer, config, remap).features)
    log.info("data %s train examples | %s dev", f"{len(train):,}", f"{len(dev):,}")

    output = Path(CHECKPOINT_PATH) / "mt" / run_name
    output.mkdir(parents=True, exist_ok=True)

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing_factor,
        optim=config.optimizer,
        gradient_checkpointing=config.gradient_checkpointing,
        num_train_epochs=config.epochs,
        max_steps=max_steps or -1,
        # 64% of the padded tokens in a random batch are padding: the corpus median pair
        # is 12 tokens against a 128 cap (`tools/mt_lengths.py`). v5 renamed the flag from
        # `group_by_length`.
        train_sampling_strategy="group_by_length",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        eval_steps=config.eval_every,
        save_strategy="steps",
        save_steps=config.save_every,
        save_total_limit=3,
        logging_steps=config.log_every,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        seed=config.seed,
        report_to=[],
        disable_tqdm=True,
    )

    # The trimmed checkpoint's config still names its specials in the tokenizer's id
    # space. NLLB's are its four lowest ids and `required_ids` always keeps them, so the
    # remap is the identity here — going through it anyway is what makes a tokenizer where
    # that does not hold fail loudly instead of training on the wrong start token.
    pad_id, start_id = model.config.pad_token_id, model.config.decoder_start_token_id
    if remap is not None:
        pad_id, start_id = remap.to_new([pad_id, start_id])

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=Collator(DataCollatorForSeq2Seq(tokenizer, model=model), pad_id, start_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    normalise_accumulated_loss(trainer)

    resume = any(output.glob("checkpoint-*"))
    log.info("training from %s", "the newest checkpoint" if resume else "the base weights")
    result = trainer.train(resume_from_checkpoint=resume)

    final = output / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    if remap is not None:
        (final / "keep.json").write_text(
            json.dumps({"keep": list(remap.keep)}) + "\n", encoding="utf-8"
        )
    checkpoint_volume.commit()

    summary: dict[str, object] = {
        "run_name": run_name,
        "model": config.model,
        "trimmed": trimmed,
        "freeze_embeddings": freeze_embeddings,
        "steps": result.global_step,
        "train_loss": round(result.training_loss, 4),
        "trainable": trainable,
        "total": total,
        "path": str(final),
    }
    log.info("done %s", json.dumps(summary))
    return summary


@app.local_entrypoint()
def upload_mt() -> None:
    """Push the built fine-tuning corpus to the data volume."""
    train = LOCAL_DATA / "train.jsonl"
    dev = LOCAL_DATA / "dev.jsonl"
    for path in (train, dev):
        if not path.exists():
            message = f"{path} missing; run `make mt-data` first"
            raise SystemExit(message)

    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(train, f"/{REMOTE_DATA}/train.jsonl")
        batch.put_file(dev, f"/{REMOTE_DATA}/dev.jsonl")
    size = (train.stat().st_size + dev.stat().st_size) / 1e6
    print(f"uploaded the MT corpus, {size:.1f} MB")


@app.local_entrypoint()
def prepare(small: bool = False, force: bool = False) -> None:
    """Build the trimmed checkpoint. Short and CPU-only, so a blocking call is fine here.

    The fine-tune itself is *not* launched this way: it runs for hours, and a local
    entrypoint blocking on `.remote()` gives Modal a client whose disconnect cancels the
    call. `modal_app.launch --function finetune` spawns it against the deployed app.
    """
    print(json.dumps(trim.remote(small=small, force=force), indent=2))
