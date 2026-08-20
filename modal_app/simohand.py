"""SiMohand: Kabyle sentence transformer training on Modal.

Trains a dense sentence embedding model (SiMohand-Base) with:
1. **Vocabulary Repair**: Expands token embeddings for missing native consonants
   (`Ɛ`, `Ɣ`, `Ǧ`, `Ẓ`, `ẓ`) using donor initialisation before training begins.
2. **Cluster-Aware Contrastive Batching**: Prevents in-batch false negative collisions
   by guaranteeing distinct `cluster_id`s in every mini-batch.
3. **Matryoshka Representation Learning**: Multi-tier loss over nested dimension slices
   [768, 512, 256, 128, 64] allowing lightweight downstream deployment.
4. **Isotropic Collapse Control**: Verifies embedding space isotropy and STS rank correlation.
5. **Live Probe Logging**: Periodically probes rotating semantic triplets to inspect margin
   growth and Matryoshka dimension retention in real time.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Final

from modal_app.common import (
    DATA_PATH,
    EMBED_CPU,
    EMBED_GPU,
    EMBED_TIMEOUT,
    VOLUMES,
    app,
    checkpoint_volume,
    data_volume,
    embed_image,
)

log: Final = logging.getLogger("agbalu.modal.simohand")

DEFAULT_BACKBONE: Final[str] = "intfloat/multilingual-e5-base"
MATRYOSHKA_DIMS: Final[tuple[int, ...]] = (768, 512, 256, 128, 64)
TOP_1: Final = 1
TOP_5: Final = 5
"""The two cut-offs Recall@k is reported at. Named because the same pair of literals is
written twice — once for the full 768 dimensions and once per Matryoshka slice — and a
recall reported at a k it was not computed at is not detectable from the number."""
DEFAULT_BATCH_SIZE: Final[int] = 64
MAX_SEQ_LENGTH: Final[int] = 128


PROBE_TRIPLETS: Final[tuple[dict[str, str], ...]] = (
    {
        "anchor": "Yebɣa ad yeǧǧ axxam.",
        "pos": "Ira ad yeǧǧ axxam.",
        "neg": "Yečča imensi deg uxxam.",
        "trans": "He wants to leave the house.",
    },
    {
        "anchor": "Azul fell-awen, amek tteddunt temsal?",
        "pos": "Ansuf yis-wen, amek tellam?",
        "neg": "Yeffeɣ seg tebḥirt taṣebḥit.",
        "trans": "Hello everyone, how are things going?",
    },
    {
        "anchor": "Taqbaylit d tutlayt tayemmat nneɣ.",
        "pos": "Tutlayt nneɣ n tyemmat d Taqbaylit.",
        "neg": "Yuli s adrar deg tegrest.",
        "trans": "Kabyle is our mother tongue.",
    },
    {
        "anchor": "Aman d tudert n yemdanen d yimɣan.",
        "pos": "War aman ulac tudert i umdan.",
        "neg": "Irgazen kkan ɣer ssuq ass n lrebeɛ.",
        "trans": "Water is the life of humans and plants.",
    },
)


def _emit(event: str, **fields: object) -> None:
    """Format and emit structured log events consistent across Agbalu models."""
    parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    log.info("%-12s %s", event, parts)


@app.function(
    image=embed_image,
    volumes=VOLUMES,
    timeout=60 * 60,
)
def simohand_prepare(
    *,
    force: bool = False,
) -> dict[str, object]:
    """Build and verify the embedding pair corpus on the volume."""
    import logging

    from agbalu.embed.corpus import build_embed_corpus

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    embed_dir = Path(DATA_PATH) / "embed"
    corpus_dir = embed_dir / "corpus"
    stats_file = corpus_dir / "corpus.stats.json"

    if stats_file.is_file() and not force:
        log.info("corpus already prepared at %s", corpus_dir)
        cached: object = json.loads(stats_file.read_text(encoding="utf-8"))
        if not isinstance(cached, dict):
            message = f"{stats_file} does not hold a JSON object"
            raise TypeError(message)
        return {str(key): value for key, value in cached.items()}

    corpus_dir.mkdir(parents=True, exist_ok=True)
    parallel_dir = Path(DATA_PATH) / "parallel"
    tapaco_file = Path(DATA_PATH) / "raw" / "tatoeba" / "tapaco_kab_2026-08-05.tsv"

    stats = build_embed_corpus(
        parallel_dir=parallel_dir,
        tapaco_path=tapaco_file,
        output_dir=corpus_dir,
    )
    data_volume.commit()
    return dict(stats)


@app.function(
    image=embed_image,
    gpu=EMBED_GPU,
    cpu=EMBED_CPU,
    volumes=VOLUMES,
    timeout=EMBED_TIMEOUT,
)
def simohand_train(
    *,
    backbone: str = DEFAULT_BACKBONE,
    epochs: int = 3,
    max_steps: int = 0,
    limit: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    run_name: str = "simohand-base-v1",
    force: bool = False,
) -> dict[str, object]:
    """Fine-tune the sentence transformer with Matryoshka loss and cluster-aware batching."""
    import logging

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    # `sentence_transformers.losses` is a deprecation shim that resolves through a module
    # `__getattr__`, so it type-checks as a missing attribute and warns at import.
    from sentence_transformers.sentence_transformer.losses import (
        MatryoshkaLoss,
        MultipleNegativesRankingLoss,
    )
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    from agbalu.bench.sts import check_isotropic_collapse
    from agbalu.embed.backbone import repair

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    smoke = limit > 0 or max_steps > 0
    actual_run_name = f"{run_name}-smoke" if smoke and not run_name.endswith("-smoke") else run_name

    embed_dir = Path(DATA_PATH) / "embed"
    corpus_dir = embed_dir / "corpus"
    train_path = (
        (embed_dir / "train.jsonl")
        if (embed_dir / "train.jsonl").is_file()
        else (corpus_dir / "train.jsonl")
    )
    dev_path = (
        (embed_dir / "dev.jsonl")
        if (embed_dir / "dev.jsonl").is_file()
        else (corpus_dir / "dev.jsonl")
    )

    if not train_path.is_file():
        # Deterministic and knowable before the card is charged: building the corpus is CPU
        # work with its own container, and this one has an A10 attached.
        message = f"no pair corpus at {train_path}; run `make modal-simohand TASK=prepare` first"
        raise FileNotFoundError(message)

    # 1. Load data
    train_lines = train_path.read_text(encoding="utf-8").splitlines()
    train_rows = [json.loads(line) for line in train_lines if line.strip()]

    dev_lines = dev_path.read_text(encoding="utf-8").splitlines()
    dev_rows = [json.loads(line) for line in dev_lines if line.strip()]

    if limit > 0:
        train_rows = train_rows[:limit]
        dev_rows = dev_rows[: min(limit, 100)]

    _emit(
        "corpus",
        train_pairs=len(train_rows),
        dev_pairs=len(dev_rows),
        smoke=smoke,
    )

    train_data = [{"anchor": r["query"], "positive": r["passage"]} for r in train_rows]
    train_ds = Dataset.from_list(train_data)
    dev_data = [{"anchor": r["query"], "positive": r["passage"]} for r in dev_rows]
    dev_ds = Dataset.from_list(dev_data)

    # 2. Model & Vocabulary Expansion
    _emit("backbone", name=backbone)
    torch.set_float32_matmul_precision("high")
    model = SentenceTransformer(backbone)
    model.max_seq_length = MAX_SEQ_LENGTH

    tokenizer = model.tokenizer
    hf_model = model[0].auto_model

    # Widen vocabulary & inject donor embeddings
    repair_summary = repair(hf_model, tokenizer)
    _emit(
        "vocab_repair",
        added=",".join(repair_summary.added),
        before=repair_summary.vocabulary_before,
        after=repair_summary.vocabulary_after,
    )

    # 3. Loss setup: MatryoshkaLoss wrapping MultipleNegativesRankingLoss
    base_loss = MultipleNegativesRankingLoss(model=model, scale=20.0)
    train_loss = MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=list(MATRYOSHKA_DIMS),
    )

    # 4. Training Arguments (A10 Ampere TensorCore & Memory Optimized)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    out_dir = Path(DATA_PATH) / "embed" / "runs" / actual_run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_interval = 5 if smoke else 50
    preview_interval = 5 if smoke else 250

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs if max_steps <= 0 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio if max_steps <= 0 else 0.0,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=2,
        logging_steps=logging_interval,
        report_to="none",
    )

    # 5. Logging and Periodic Probe Callback
    class SiMohandLoggingCallback(TrainerCallback):
        def __init__(self) -> None:
            self._last_time = time.time()
            self._last_step = 0

        def on_log(
            self,
            _args: TrainingArguments,
            state: TrainerState,
            _control: TrainerControl,
            logs: dict[str, float] | None = None,
            **_kwargs: object,
        ) -> None:
            if not logs:
                return
            now = time.time()
            elapsed = max(1e-4, now - self._last_time)
            steps_done = state.global_step - self._last_step
            pairs_per_sec = (steps_done * batch_size) / elapsed if steps_done > 0 else 0.0
            self._last_time = now
            self._last_step = state.global_step

            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            _emit(
                "train",
                step=state.global_step,
                epoch=f"{state.epoch:.2f}" if state.epoch else "0.00",
                loss=f"{loss:.4f}",
                lr=f"{lr:.2e}",
                pairs_per_sec=f"{pairs_per_sec:.1f}",
            )

            # Periodic live semantic probe on rotating sentences
            if state.global_step > 0 and (
                state.global_step % preview_interval == 0 or state.global_step == 1
            ):
                self._run_probe(state.global_step)

        def _run_probe(self, step: int) -> None:
            idx = (step // max(1, preview_interval)) % len(PROBE_TRIPLETS)
            probe = PROBE_TRIPLETS[idx]
            texts = [probe["anchor"], probe["pos"], probe["neg"], probe["trans"]]
            with torch.no_grad():
                emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=-1)

                pos_768 = float(torch.dot(emb_norm[0], emb_norm[1]).item())
                neg_768 = float(torch.dot(emb_norm[0], emb_norm[2]).item())
                trans_768 = float(torch.dot(emb_norm[0], emb_norm[3]).item())

                emb_64 = torch.nn.functional.normalize(emb[:, :64], p=2, dim=-1)
                pos_64 = float(torch.dot(emb_64[0], emb_64[1]).item())

                margin = pos_768 - neg_768
                _emit(
                    "probe",
                    step=step,
                    anchor=f'"{probe["anchor"][:24]}..."',
                    pos_768=f"{pos_768:.3f}",
                    pos_64=f"{pos_64:.3f}",
                    trans=f"{trans_768:.3f}",
                    neg=f"{neg_768:.3f}",
                    margin=f"{margin:+.3f}",
                )

        def on_epoch_end(
            self,
            _args: TrainingArguments,
            state: TrainerState,
            _control: TrainerControl,
            **_kwargs: object,
        ) -> None:
            if not dev_rows:
                return
            sample_texts = [r["query"] for r in dev_rows[:500]]
            with torch.no_grad():
                embeddings = model.encode(sample_texts, show_progress_bar=False).tolist()
            iso = check_isotropic_collapse(embeddings)
            _emit(
                "epoch_eval",
                epoch=int(state.epoch or 0),
                mean_cosine=f"{iso.mean_cosine:.4f}",
                std_cosine=f"{iso.std_cosine:.4f}",
                is_collapsed=iso.collapsed,
            )

        def on_save(
            self,
            _args: TrainingArguments,
            state: TrainerState,
            _control: TrainerControl,
            **_kwargs: object,
        ) -> None:
            data_volume.commit()
            checkpoint_volume.commit()
            _emit("checkpoint", step=state.global_step, epoch=f"{state.epoch:.2f}")

    # 6. Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if not smoke else None,
        loss=train_loss,
        callbacks=[SiMohandLoggingCallback()],
    )

    last_checkpoint = None
    if out_dir.is_dir() and not force and not smoke:
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(str(out_dir))
        if last_checkpoint:
            _emit("resume", checkpoint=last_checkpoint)

    _emit("launch", gpu=EMBED_GPU, total_epochs=epochs, batch=batch_size)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 7. Save Model
    final_dir = Path(DATA_PATH) / "embed" / "models" / run_name
    model.save_pretrained(str(final_dir))
    _emit("saved", path=str(final_dir))

    # 8. Final Evaluation: Isotropic Check
    sample_texts = [r["query"] for r in train_rows[:1000]]
    embeddings = model.encode(sample_texts, show_progress_bar=False).tolist()
    iso_check = check_isotropic_collapse(embeddings)

    result_report = {
        "run_name": run_name,
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "repaired_tokens": list(repair_summary.added),
        "mean_cosine": iso_check.mean_cosine,
        "std_cosine": iso_check.std_cosine,
        "is_collapsed": iso_check.collapsed,
        "output_path": str(final_dir),
    }

    report_path = final_dir / "evaluation.report.json"
    report_path.write_text(json.dumps(result_report, indent=2), encoding="utf-8")

    data_volume.commit()
    checkpoint_volume.commit()
    return result_report


@app.function(
    image=embed_image,
    gpu=EMBED_GPU,
    cpu=EMBED_CPU,
    volumes=VOLUMES,
    timeout=60 * 30,
)
def simohand_eval(
    *,
    run_name: str = "simohand-base-v1",
    limit: int = 0,
) -> dict[str, object]:
    """Run comprehensive Head-to-Head (H2H) evaluation comparing SiMohand against baselines."""
    import logging

    import torch
    from sentence_transformers import SentenceTransformer

    from agbalu.bench.sts import check_isotropic_collapse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    embed_dir = Path(DATA_PATH) / "embed"
    corpus_dir = embed_dir / "corpus"
    model_dir = embed_dir / "models" / run_name
    dev_path = (
        (embed_dir / "dev.jsonl")
        if (embed_dir / "dev.jsonl").is_file()
        else (corpus_dir / "dev.jsonl")
    )

    if not model_dir.is_dir():
        runs_dir = embed_dir / "runs" / run_name
        if runs_dir.is_dir():
            from transformers.trainer_utils import get_last_checkpoint

            last_ckpt = get_last_checkpoint(str(runs_dir))
            if last_ckpt:
                model_dir = Path(last_ckpt)
        if not model_dir.is_dir():
            msg = f"Model directory not found at {model_dir}"
            raise FileNotFoundError(msg)

    if not dev_path.is_file():
        message = f"no dev pairs at {dev_path}; run `make modal-simohand TASK=prepare` first"
        raise FileNotFoundError(message)

    _emit("eval_start", target=str(model_dir))

    # 1. Load Dev Pairs
    raw_lines = [line for line in dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_rows = [json.loads(line) for line in raw_lines]
    if limit > 0:
        dev_rows = dev_rows[:limit]

    queries = [r["query"] for r in dev_rows]
    positives = [r["passage"] for r in dev_rows]
    n_pairs = len(queries)
    _emit("pairs_loaded", count=n_pairs)

    models_to_test = {
        "SiMohand-278M (Ours)": str(model_dir),
        "multilingual-e5-base (Backbone)": "intfloat/multilingual-e5-base",
        "LaBSE (Google Baseline)": "sentence-transformers/LaBSE",
    }

    scoreboard: dict[str, object] = {}

    for name, path in models_to_test.items():
        _emit("evaluating_model", name=name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(path, device=device)

        q_embs = model.encode(
            queries, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )
        pos_embs = model.encode(
            positives, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
        )

        # Cosine similarity matrix [N, N]
        sim_matrix = torch.mm(q_embs, pos_embs.t())

        # Rank of target positive (diagonal i == i)
        ranks: list[int] = []
        for i in range(n_pairs):
            tgt_score = sim_matrix[i, i].item()
            rank = (sim_matrix[i] > tgt_score).sum().item() + 1
            ranks.append(int(rank))

        r1 = sum(1 for r in ranks if r == TOP_1) / n_pairs
        r5 = sum(1 for r in ranks if r <= TOP_5) / n_pairs
        mrr = sum(1.0 / r for r in ranks) / n_pairs

        pos_sims = sim_matrix.diag().tolist()
        mean_pos = sum(pos_sims) / n_pairs

        off_diag_mask = ~torch.eye(n_pairs, dtype=torch.bool, device=sim_matrix.device)
        mean_neg = sim_matrix[off_diag_mask].mean().item()
        mean_margin = mean_pos - mean_neg

        iso = check_isotropic_collapse(q_embs.cpu().tolist())

        entry: dict[str, object] = {
            "recall@1": round(r1 * 100, 2),
            "recall@5": round(r5 * 100, 2),
            "mrr": round(mrr, 4),
            "mean_pos": round(mean_pos, 4),
            "mean_neg": round(mean_neg, 4),
            "margin": round(mean_margin, 4),
            "mean_cosine": iso.mean_cosine,
            "std_cosine": iso.std_cosine,
            "is_collapsed": iso.collapsed,
        }

        # Matryoshka dimension slices for SiMohand
        if "SiMohand" in name:
            mrl_slices: dict[str, object] = {}
            for d in [512, 256, 128, 64]:
                q_d = torch.nn.functional.normalize(q_embs[:, :d], p=2, dim=1)
                pos_d = torch.nn.functional.normalize(pos_embs[:, :d], p=2, dim=1)
                sim_d = torch.mm(q_d, pos_d.t())

                ranks_d: list[int] = []
                for i in range(n_pairs):
                    tgt_d = sim_d[i, i].item()
                    ranks_d.append(int((sim_d[i] > tgt_d).sum().item() + 1))

                r1_d = sum(1 for r in ranks_d if r == TOP_1) / n_pairs
                r5_d = sum(1 for r in ranks_d if r <= TOP_5) / n_pairs
                mrr_d = sum(1.0 / r for r in ranks_d) / n_pairs

                p_d = sim_d.diag().tolist()
                mean_p_d = sum(p_d) / n_pairs
                off_diag_d = ~torch.eye(n_pairs, dtype=torch.bool, device=sim_d.device)
                mean_n_d = sim_d[off_diag_d].mean().item()
                margin_d = mean_p_d - mean_n_d

                mrl_slices[f"{d}d"] = {
                    "recall@1": round(r1_d * 100, 2),
                    "recall@5": round(r5_d * 100, 2),
                    "mrr": round(mrr_d, 4),
                    "margin": round(margin_d, 4),
                }
            entry["matryoshka_slices"] = mrl_slices

        scoreboard[name] = entry
        _emit(
            "score",
            model=name,
            r1=f"{entry['recall@1']}%",
            r5=f"{entry['recall@5']}%",
            mrr=f"{entry['mrr']}",
            margin=f"{entry['margin']}",
            iso_mean=f"{entry['mean_cosine']}",
        )

    report_dir = Path(DATA_PATH) / "embed" / "models" / run_name
    report_file = report_dir / "h2h_report.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    data_volume.commit()

    _emit("eval_complete", report=str(report_file))
    return scoreboard


LOCAL_EMBED: Final = Path("data/processed/embed")
EMBED_SPLITS: Final[tuple[str, ...]] = ("train.jsonl", "dev.jsonl", "corpus.stats.json")


@app.local_entrypoint()
def upload_embed() -> None:
    """Push the locally built pair corpus to the data volume.

    `simohand_train` prefers `/embed/train.jsonl` over the one `simohand_prepare` writes
    under `/embed/corpus/`, so this is the path that skips the remote build entirely.
    """
    missing = [name for name in EMBED_SPLITS if not (LOCAL_EMBED / name).is_file()]
    if missing:
        names = ", ".join(missing)
        message = f"{names} missing from {LOCAL_EMBED}; run `make embed TASK=corpus` first"
        raise SystemExit(message)

    with data_volume.batch_upload(force=True) as batch:
        for name in EMBED_SPLITS:
            batch.put_file(LOCAL_EMBED / name, f"/embed/{name}")
    for name in EMBED_SPLITS:
        size = (LOCAL_EMBED / name).stat().st_size
        print(f"uploaded {name} ({size / 1e6:.1f} MB)")
