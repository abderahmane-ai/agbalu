"""Modal app for training and deploying Boulifa-48M and generating the KabStandard dataset."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import torch
from torch.utils.data import DataLoader, Dataset

from modal_app.common import (
    CHECKPOINT_PATH,
    DATA_PATH,
    RETRIES,
    VOLUMES,
    app,
    checkpoint_volume,
    data_volume,
    image,
)

log: Final = logging.getLogger("agbalu.boulifa")

GPU: Final = "A10G"
DEFAULT_RUN: Final = "boulifa-48m-v1"
REMOTE_DATA_ROOT: Final = Path(DATA_PATH) / "standardise"
REMOTE_CHECKPOINT_ROOT: Final = Path(CHECKPOINT_PATH) / "boulifa"

TIMEOUT: Final = 3 * 60 * 60

TATOEBA_COLUMNS: Final = 3
"""id, language, text. A shorter row is a truncated line, not a Kabyle sentence."""


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, pairs: list[dict[str, str]], tokenizer: object) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        src_ids = self.tokenizer.encode(  # type: ignore[attr-defined]
            pair["source"], add_bos=True, add_eos=True, max_length=256
        )
        tgt_ids = self.tokenizer.encode(  # type: ignore[attr-defined]
            pair["target"], add_bos=True, add_eos=True, max_length=256
        )
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    srcs, tgts = zip(*batch, strict=True)
    src_padded = torch.nn.utils.rnn.pad_sequence(list(srcs), batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(list(tgts), batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded


@app.function(
    image=image,
    volumes=VOLUMES,
    timeout=TIMEOUT,
)
def boulifa_prepare(
    *,
    limit: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    import urllib.request

    import pyarrow.parquet as pq

    from agbalu.standardise.corpus import generate_pairs, save_jsonl

    _configure_logging()
    data_volume.reload()

    sentences: list[str] = []

    # 1. Look for existing parquets on volume or fetch from Hub
    parquet_dir = Path(DATA_PATH) / "tasks" / "tifinagh" / "script_conversion"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    hub_url = "https://huggingface.co/datasets/agbalu/KabTifinagh/resolve/main/script_conversion"

    for split in ("train", "dev", "test"):
        pq_path = parquet_dir / f"{split}.parquet"
        if not pq_path.is_file():
            url = f"{hub_url}/{split}.parquet"
            log.info("Fetching %s from %s...", split, url)
            staging = pq_path.with_suffix(".partial")
            try:
                urllib.request.urlretrieve(url, staging)  # noqa: S310
                staging.replace(pq_path)
                data_volume.commit()
            except Exception as exc:
                log.warning("Could not fetch %s from Hub: %s", url, exc)

        if pq_path.is_file():
            try:
                table = pq.read_table(pq_path)
                col_name = "text_latn" if "text_latn" in table.column_names else "latin"
                if col_name in table.column_names:
                    col_data = table[col_name].to_pylist()
                    sentences.extend(col_data)
                    log.info(
                        "Loaded %d sentences from %s (%s)",
                        len(col_data),
                        pq_path.name,
                        col_name,
                    )
            except Exception as exc:
                log.warning("Failed to read %s: %s", pq_path, exc)

    # 2. Fallback to volume text files if parquets are missing
    if not sentences:
        tatoeba_path = Path(DATA_PATH) / "raw" / "tatoeba" / "tatoeba_kab_mono_2026-08-05.tsv"
        if tatoeba_path.is_file():
            with tatoeba_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    parts = raw_line.strip().split("\t")
                    if len(parts) >= TATOEBA_COLUMNS and parts[1] == "kab":
                        sentences.append(parts[2])

    if limit > 0:
        sentences = sentences[:limit]

    log.info("Generating parallel pairs for %d canonical sentences...", len(sentences))
    pairs = list(generate_pairs(sentences, seed=seed))

    # Split: 90% train, 5% dev, 5% test
    n = len(pairs)
    n_train = int(n * 0.90)
    n_dev = int(n * 0.05)

    train_pairs = pairs[:n_train]
    dev_pairs = pairs[n_train : n_train + n_dev]
    test_pairs = pairs[n_train + n_dev :]

    out_dir = REMOTE_DATA_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_pairs, out_dir / "train.jsonl")
    save_jsonl(dev_pairs, out_dir / "dev.jsonl")
    save_jsonl(test_pairs, out_dir / "test.jsonl")

    data_volume.commit()
    log.info(
        "KabStandard dataset written: train=%d, dev=%d, test=%d",
        len(train_pairs),
        len(dev_pairs),
        len(test_pairs),
    )

    return {
        "status": "success",
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "test_pairs": len(test_pairs),
        "root": str(out_dir),
    }


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    timeout=TIMEOUT,
    retries=RETRIES,
)
def boulifa_train(
    *,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    limit: int = 0,
) -> dict[str, object]:
    """Train Boulifa-48M on GPU."""
    from dataclasses import asdict

    from agbalu.standardise.config import ModelConfig
    from agbalu.standardise.corpus import load_jsonl
    from agbalu.standardise.model import CharTransformer
    from agbalu.standardise.tokenizer import Tokenizer

    _configure_logging()
    data_volume.reload()
    checkpoint_volume.reload()

    train_path = REMOTE_DATA_ROOT / "train.jsonl"
    if not train_path.is_file():
        # Auto-prepare dataset
        log.info("KabStandard train.jsonl not found; generating dataset first...")
        boulifa_prepare.local(limit=limit)
        data_volume.reload()

    train_pairs = load_jsonl(train_path)
    if limit > 0:
        train_pairs = train_pairs[:limit]

    dev_pairs = load_jsonl(REMOTE_DATA_ROOT / "dev.jsonl")
    if limit > 0:
        dev_pairs = dev_pairs[: max(10, limit // 10)]

    tokenizer = Tokenizer.build()
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CharTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    train_dataset = TextDataset([p.as_dict() for p in train_pairs], tokenizer)
    dev_dataset = TextDataset([p.as_dict() for p in dev_pairs], tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.pad_id),
    )

    log.info(
        "Starting Boulifa-48M training: %d train pairs, %d dev pairs",
        len(train_pairs),
        len(dev_pairs),
    )
    started = time.monotonic()
    best_dev_acc = 0.0

    out_dir = REMOTE_CHECKPOINT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_dev_acc = 0.0
    latest_ckpt = out_dir / "boulifa_latest.pt"

    if latest_ckpt.is_file():
        log.info("Found existing checkpoint at %s; loading state...", latest_ckpt)
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model"])
        if "optimizer" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer"])
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_dev_acc = ckpt_data.get("best_dev_acc", 0.0)
        log.info(
            "Resuming training from Epoch %d (Best Dev Acc: %.4f)",
            start_epoch,
            best_dev_acc,
        )

    # Fixed qualitative probe sentences to monitor progress in real-time
    eval_probes = [
        "achimi ur d-thekhedmedh ara tamazight g l'ecole?",
        "a7bib-iw l3ali, khedmegh g taddarth",
        "djedjiga thetchour d zith n wzemmour",
        "dyeffegh fellawen tmeslaytnnegh",
        "Azul fell-awen, amek i telliḍ taṣebḥit-a?",
    ]

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        epoch_start = time.monotonic()

        for step, (src_in, tgt_in) in enumerate(train_loader, start=1):
            src_dev = src_in.to(device)
            tgt_dev = tgt_in.to(device)
            optimizer.zero_grad()

            loss_out = model(src_dev, tgt_dev, pad_id=tokenizer.pad_id)
            loss_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_out.loss.item()
            total_correct += int(loss_out.correct.item())
            total_tokens += int(loss_out.tokens.item())

            if step % 100 == 0:
                acc = total_correct / max(1, total_tokens)
                elapsed_sec = max(0.1, time.monotonic() - epoch_start)
                tok_per_sec = total_tokens / elapsed_sec
                log.info(
                    "Epoch [%d/%d] Step [%d/%d] Loss: %.4f Acc: %.4f (%.0f tokens/s)",
                    epoch,
                    epochs,
                    step,
                    len(train_loader),
                    loss_out.loss.item(),
                    acc,
                    tok_per_sec,
                )

        # Dev evaluation
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_tokens = 0

        with torch.no_grad():
            for src_in, tgt_in in dev_loader:
                src_dev = src_in.to(device)
                tgt_dev = tgt_in.to(device)
                loss_out = model(src_dev, tgt_dev, pad_id=tokenizer.pad_id)
                dev_loss += loss_out.loss.item()
                dev_correct += int(loss_out.correct.item())
                dev_tokens += int(loss_out.tokens.item())

        dev_acc = dev_correct / max(1, dev_tokens)
        dev_mean_loss = dev_loss / max(1, len(dev_loader))
        log.info(
            "epoch %d complete: dev loss %.4f, dev char accuracy %.2f%%",
            epoch,
            dev_mean_loss,
            dev_acc * 100.0,
        )

        with torch.no_grad():
            for idx, probe in enumerate(eval_probes, start=1):
                p_ids = tokenizer.encode(probe, add_bos=True, add_eos=True)
                p_in = torch.tensor([p_ids], dtype=torch.long, device=device)
                memory = model.encode(p_in)
                gen_ids = [tokenizer.bos_id]
                for _ in range(min(256, len(p_ids) * 2)):
                    d_in = torch.tensor([gen_ids], dtype=torch.long, device=device)
                    decoded = model.decode(d_in, memory)
                    logits = model.output_projection(decoded[:, -1:, :])
                    nxt = int(logits.argmax(dim=-1).item())
                    if nxt == tokenizer.eos_id:
                        break
                    gen_ids.append(nxt)
                gen_text = tokenizer.decode(gen_ids[1:], skip_special_tokens=True)
                log.info("  probe %d in:  %s", idx, probe)
                log.info("  probe %d out: %s", idx, gen_text)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_dev_acc": best_dev_acc,
                "config": asdict(config),
            },
            out_dir / "boulifa_latest.pt",
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(
                {"model": model.state_dict(), "config": asdict(config)},
                out_dir / "boulifa_best.pt",
            )

        checkpoint_volume.commit()

    torch.save(
        {"model": model.state_dict(), "config": asdict(config)},
        out_dir / "boulifa_final.pt",
    )
    checkpoint_volume.commit()
    elapsed = round(time.monotonic() - started, 1)
    log.info("Training complete in %.1f seconds. Best Dev Acc: %.4f", elapsed, best_dev_acc)

    return {
        "status": "success",
        "epochs": epochs,
        "best_dev_acc": best_dev_acc,
        "seconds": elapsed,
        "checkpoint": str(out_dir / "boulifa_final.pt"),
    }
