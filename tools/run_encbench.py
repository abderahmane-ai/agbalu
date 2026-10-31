"""KabEncBench: Multi-task evaluation harness and Elo leaderboard runner.

Evaluates pretrained encoders across 6 core supervised tasks and 3 diagnostic tasks:
Core:
  1. syntax (KabSyntax / UD_Kabyle-ADPT): UPOS macro-F1 & dependency tagging
  2. morph (KabMorph): Factored morphological classification
  3. ner (KabNER): 9 IOB2 tags, span micro-F1
  4. sent (KabSentiment): 3 classes (neg, neu, pos)
  5. topic (KabTopic / SIB-200): 7 classes
  6. punct (KabPunct): Joint punctuation (5 classes) and casing
Diagnostics:
  1. diag_clitic: Clitic attachment host classification (10 classes)
  2. diag_para: Zero-tune layer-wise paraphrase retrieval (TaPaCo-kab)
  3. diag_spell: Zero-leakage minimal pairs (1,000 pairs across 4 phenomena)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from torch import Tensor
from tqdm import tqdm  # type: ignore[import-untyped]

from agbalu.bench.elo import compute_leaderboard
from agbalu.bench.encbench import (
    DEFAULT_SEEDS,
    FROZEN_LR_GRID,
    TaskRunResult,
    evaluate_paraphrase_retrieval,
    read_task_split,
)
from agbalu.bench.encbench_runner import (
    PUNCT_TAG_TO_ID,
    PUNCT_TAGS,
    CachedClassificationData,
    FitSchedule,
    evaluate_cached_classification,
)


class BackboneAdapter:
    """Unified interface for extracting multi-layer representations from encoders."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.model_id = model_id
        self.device = device

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    def encode_text(self, text: str) -> tuple[Tensor, Tensor]:
        """Return (input_ids, attention_mask) [1, seq_len]."""
        raise NotImplementedError

    def encode_words(self, words: Sequence[str]) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
        """Return (input_ids, attention_mask, word_spans) for aligned word tokens."""
        raise NotImplementedError

    def forward_layers(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        """Return list of all L+1 hidden representations [batch, seq_len, hidden_size]."""
        raise NotImplementedError

    def score_sentence_loss(self, text: str) -> float:
        """Compute MLM cross-entropy loss or pseudo-loss on sentence for minimal pairs."""
        raise NotImplementedError


class Masinissa73MAdapter(BackboneAdapter):
    """Adapter for Masinissa-73M pretraining checkpoint."""

    def __init__(
        self,
        snapshot_path: Path,
        tokenizer_path: Path,
        device: torch.device,
    ) -> None:
        super().__init__("agbalu/Masinissa-73M", device)
        from agbalu.encoder.attention import make_layout
        from agbalu.encoder.configuration import Geometry
        from agbalu.encoder.model import Encoder

        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        # Use reference backend for inference to avoid FlexAttention dynamic compilation overhead
        self.geo = Geometry(backend="reference")
        self.encoder = Encoder(self.geo)
        ckpt = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(ckpt["weights"])
        self.encoder.to(device).eval()
        self._make_layout = make_layout

    @property
    def hidden_size(self) -> int:
        return self.geo.hidden_size

    @property
    def num_layers(self) -> int:
        return self.geo.layers

    def encode_text(self, text: str) -> tuple[Tensor, Tensor]:
        pieces = self.sp.encode_as_ids(text)
        ids = [2, *pieces[: self.geo.context_tokens - 2], 3]
        t_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        t_mask = torch.ones_like(t_ids)
        return t_ids, t_mask

    def encode_words(self, words: Sequence[str]) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
        pieces = [2]
        spans: list[tuple[int, int]] = []
        cur = 1
        for w in words:
            wp = self.sp.encode_as_ids(w)
            if not wp:
                wp = [1]
            spans.append((cur, cur + len(wp)))
            pieces.extend(wp)
            cur += len(wp)
        pieces.append(3)
        if len(pieces) > self.geo.context_tokens:
            pieces = [*pieces[: self.geo.context_tokens - 1], 3]
            spans = [
                (min(s, self.geo.context_tokens - 1), min(e, self.geo.context_tokens - 1))
                for s, e in spans
            ]
        t_ids = torch.tensor([pieces], dtype=torch.long, device=self.device)
        t_mask = torch.ones_like(t_ids)
        return t_ids, t_mask, spans

    def forward_layers(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        batch, length = input_ids.shape
        documents = torch.arange(batch, device=self.device).repeat_interleave(length)
        positions = torch.arange(length, device=self.device).repeat(batch)
        layout = self._make_layout(
            documents, positions, attention_mask.flatten().bool(), flex=False
        )

        with torch.inference_mode():
            embedded = self.encoder.embedding(input_ids.flatten())
            layers: list[Tensor] = [embedded.reshape(batch, length, -1)]
            blocks = [self.encoder.dropout(embedded)]
            partial = None
            head_shape = (
                2 * self.encoder.config.buckets - 1,
                self.encoder.config.heads,
                self.encoder.config.head_size,
            )
            pq = self.encoder.position_q(self.encoder.relative)
            pk = self.encoder.position_k(self.encoder.relative)
            relative = (
                pq.view(head_shape).transpose(0, 1),
                pk.view(head_shape).transpose(0, 1),
            )

            for index, layer in enumerate(self.encoder.layers):
                partial = layer.forward(blocks, partial, relative, layout)
                layers.append(partial.reshape(batch, length, -1))
                if (index + 1) % self.encoder.config.block_layers == 0:
                    blocks.append(partial)
                    partial = None

            final_out = self.encoder.final_norm.forward(
                self.encoder.final_route.forward(blocks, partial)
            )
            layers[-1] = final_out.reshape(batch, length, -1)
        return layers

    def score_sentence_loss(self, text: str) -> float:
        input_ids, attention_mask = self.encode_text(text)
        with torch.inference_mode():
            layers = self.forward_layers(input_ids, attention_mask)
            logits = self.encoder.logits(layers[-1][0])
            targets = input_ids[0, 1:-1]
            pred_logits = logits[1:-1]
            loss = torch.nn.functional.cross_entropy(pred_logits, targets, reduction="mean")
            return float(loss.item())


class Masinissa31MAdapter(BackboneAdapter):
    """Adapter for Masinissa-31M release."""

    def __init__(self, model_dir_or_hub: str, device: torch.device) -> None:
        super().__init__("agbalu/Masinissa-31M", device)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_hub, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir_or_hub, trust_remote_code=True)
        self.model.to(device).eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def encode_text(self, text: str) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def encode_words(self, words: Sequence[str]) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
        encoded = self.tokenizer(list(words), is_split_into_words=True, return_tensors="pt")
        t_ids = encoded["input_ids"].to(self.device)
        t_mask = encoded["attention_mask"].to(self.device)
        spans: list[tuple[int, int]] = []
        cur_word = 0
        cur_start = -1
        word_ids = encoded.word_ids()
        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id == cur_word:
                if cur_start < 0:
                    cur_start = idx
            else:
                spans.append((cur_start, idx))
                cur_word = w_id
                cur_start = idx
        if cur_start >= 0:
            spans.append((cur_start, len(word_ids) - 1))
        while len(spans) < len(words):
            spans.append((1, 2))
        return t_ids, t_mask, spans[: len(words)]

    def forward_layers(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        with torch.inference_mode():
            words, relative = self.model.embedding(input_ids)
            layers: list[Tensor] = [words]
            x = words.transpose(0, 1)
            mask = ~attention_mask.bool()[:, None, None, :]
            for att, ff in zip(
                self.model.transformer.attention_layers,
                self.model.transformer.feed_forward_layers,
                strict=True,
            ):
                x = x + att(x, mask, relative)
                x = x + ff(x)
                layers.append(x.transpose(0, 1))
        return layers

    def score_sentence_loss(self, text: str) -> float:
        input_ids, attention_mask = self.encode_text(text)
        with torch.inference_mode():
            layers = self.forward_layers(input_ids, attention_mask)
            h = layers[-1][0, 1:-1]
            return float(h.norm(dim=-1).mean().item())


class HFAutoModelAdapter(BackboneAdapter):
    """Adapter for standard Hugging Face AutoModel encoders."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        super().__init__(model_id, device)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(device).eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(getattr(self.model.config, "num_hidden_layers", 12))

    def encode_text(self, text: str) -> tuple[Tensor, Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def encode_words(self, words: Sequence[str]) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
        encoded = self.tokenizer(
            list(words),
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        t_ids = encoded["input_ids"].to(self.device)
        t_mask = encoded["attention_mask"].to(self.device)
        word_ids = encoded.word_ids()
        spans: list[tuple[int, int]] = []
        cur_word = 0
        cur_start = -1
        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id == cur_word:
                if cur_start < 0:
                    cur_start = idx
            else:
                spans.append((cur_start, idx))
                cur_word = w_id
                cur_start = idx
        if cur_start >= 0:
            spans.append((cur_start, len(word_ids) - 1))
        while len(spans) < len(words):
            spans.append((1, 2))
        return t_ids, t_mask, spans[: len(words)]

    def forward_layers(self, input_ids: Tensor, attention_mask: Tensor) -> list[Tensor]:
        with torch.inference_mode():
            outputs = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            return list(outputs.hidden_states)

    def score_sentence_loss(self, text: str) -> float:
        input_ids, attention_mask = self.encode_text(text)
        with torch.inference_mode():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state[0, 1:-1]
            return float(h.norm(dim=-1).mean().item())


# ---------------------------------------------------------------------------
# Task Evaluation Routines
# ---------------------------------------------------------------------------


def extract_pooled_dataset(
    backbone: BackboneAdapter,
    split_records: list[dict[str, Any]],
    desc: str = "Extracting",
) -> tuple[list[Tensor], Tensor]:
    """Extract mean-pooled representations across all L+1 layers for a split."""
    n_layers = backbone.num_layers + 1
    layer_chunks: list[list[Tensor]] = [[] for _ in range(n_layers)]
    labels: list[int] = []

    for item in tqdm(split_records, desc=desc, ncols=80, file=sys.stdout, leave=False):
        labels.append(int(item["label"]))
        ids, mask = backbone.encode_text(str(item["text"]))
        layers = backbone.forward_layers(ids, mask)
        m = mask.float().unsqueeze(-1)
        for l_idx, layer in enumerate(layers):
            pooled = (layer * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
            layer_chunks[l_idx].append(pooled[0].cpu())

    t_layers = [torch.stack(chunk, dim=0) for chunk in layer_chunks]
    t_labels = torch.tensor(labels, dtype=torch.long)
    return t_layers, t_labels


def evaluate_sent_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Task 4: KabSentiment (3 classes: neg, neu, pos)."""
    tr_recs = read_task_split("sent", "train", data_dir)
    dv_recs = read_task_split("sent", "dev", data_dir)
    te_recs = read_task_split("sent", "test", data_dir)

    tr_l, tr_y = extract_pooled_dataset(backbone, tr_recs, desc="  Extract sent (train)")
    dv_l, dv_y = extract_pooled_dataset(backbone, dv_recs, desc="  Extract sent (dev)")
    te_l, te_y = extract_pooled_dataset(backbone, te_recs, desc="  Extract sent (test)")

    cached = CachedClassificationData(
        task="sent",
        num_classes=3,
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=20, batch_size=64),
        device=backbone.device,
    )


def evaluate_topic_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Task 5: KabTopic / SIB-200 (7 classes)."""
    tr_recs = read_task_split("topic", "train", data_dir)
    dv_recs = read_task_split("topic", "dev", data_dir)
    te_recs = read_task_split("topic", "test", data_dir)

    tr_l, tr_y = extract_pooled_dataset(backbone, tr_recs, desc="  Extract topic (train)")
    dv_l, dv_y = extract_pooled_dataset(backbone, dv_recs, desc="  Extract topic (dev)")
    te_l, te_y = extract_pooled_dataset(backbone, te_recs, desc="  Extract topic (test)")

    cached = CachedClassificationData(
        task="topic",
        num_classes=7,
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=25, batch_size=32),
        device=backbone.device,
    )


def evaluate_clitic_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Diagnostic 1: Diag-Clitic host attachment (10 classes)."""
    label_map = {
        "OTHER": 0,
        "DAT_M": 1,
        "DAT_F": 2,
        "ACC_M": 3,
        "ACC_F": 4,
        "ACC_PL": 5,
        "DAT_PL": 6,
        "DIR_D": 7,
        "DIR_N": 8,
        "PASS": 9,
    }
    n_layers = backbone.num_layers + 1

    def _extract_clitic(records: list[dict[str, Any]], desc: str) -> tuple[list[Tensor], Tensor]:
        chunks: list[list[Tensor]] = [[] for _ in range(n_layers)]
        labels: list[int] = []
        for r in tqdm(records, desc=desc, ncols=80, file=sys.stdout, leave=False):
            lbl = label_map.get(str(r["label"]), 0)
            labels.append(lbl)
            tokens = r["tokens"]
            t_idx = int(r["target_index"])
            ids, mask, spans = backbone.encode_words(tokens)
            layers = backbone.forward_layers(ids, mask)
            s, e = spans[min(t_idx, len(spans) - 1)]
            for l_idx, layer in enumerate(layers):
                host_rep = layer[0, s:e].mean(dim=0).cpu()
                chunks[l_idx].append(host_rep)
        return [torch.stack(c, dim=0) for c in chunks], torch.tensor(labels, dtype=torch.long)

    tr_l, tr_y = _extract_clitic(
        read_task_split("diag_clitic", "train", data_dir), "  Extract clitic (train)"
    )
    dv_l, dv_y = _extract_clitic(
        read_task_split("diag_clitic", "dev", data_dir), "  Extract clitic (dev)"
    )
    te_l, te_y = _extract_clitic(
        read_task_split("diag_clitic", "test", data_dir), "  Extract clitic (test)"
    )

    cached = CachedClassificationData(
        task="diag_clitic",
        num_classes=10,
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=20, batch_size=32),
        device=backbone.device,
    )


def evaluate_para_task(backbone: BackboneAdapter, data_dir: Path) -> dict[str, float]:
    """Evaluate Diagnostic 2: Diag-Para (TaPaCo-kab zero-tune retrieval)."""
    records = read_task_split("diag_para", "test", data_dir)
    rng = random.Random(42)  # noqa: S311
    sample = rng.sample(records, min(2000, len(records)))
    cluster_ids = [int(str(r["cluster_id"])) for r in sample]

    embeddings: list[list[float]] = []
    for r in tqdm(sample, desc="  Extract para embeddings", ncols=80, file=sys.stdout, leave=False):
        ids, mask = backbone.encode_text(str(r["text"]))
        layers = backbone.forward_layers(ids, mask)
        m = mask.float().unsqueeze(-1)
        pooled = (layers[-1] * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
        embeddings.append(pooled[0].tolist())

    return evaluate_paraphrase_retrieval(embeddings, cluster_ids)


def evaluate_spell_task(backbone: BackboneAdapter, data_dir: Path) -> dict[str, float]:
    """Evaluate Diagnostic 3: Diag-Spell minimal pairs."""
    records = read_task_split("diag_spell", "test", data_dir)
    by_cat: dict[str, list[float]] = {}
    total_correct = 0

    for r in tqdm(records, desc="  Scoring minimal pairs", ncols=80, file=sys.stdout, leave=False):
        cat = str(r["category"])
        good_text = str(r["good"])
        bad_text = str(r["bad"])
        loss_good = backbone.score_sentence_loss(good_text)
        loss_bad = backbone.score_sentence_loss(bad_text)
        correct = 1.0 if loss_good < loss_bad else 0.0
        by_cat.setdefault(cat, []).append(correct)
        if correct > 0:
            total_correct += 1

    results: dict[str, float] = {
        "overall_accuracy": round(total_correct / len(records), 4),
    }
    for cat, scores in by_cat.items():
        results[f"acc_{cat.lower()}"] = round(sum(scores) / len(scores), 4)
    return results


def evaluate_ner_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Task 3: KabNER sequence labelling (9 classes)."""
    ner_tags = ("O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC")
    tag2id = {t: i for i, t in enumerate(ner_tags)}
    n_layers = backbone.num_layers + 1

    def _extract_ner_words(records: list[dict[str, Any]], desc: str) -> tuple[list[Tensor], Tensor]:
        all_words_layers: list[list[Tensor]] = [[] for _ in range(n_layers)]
        all_labels: list[int] = []
        for r in tqdm(records, desc=desc, ncols=80, file=sys.stdout, leave=False):
            tokens = r["tokens"]
            labels = [tag2id.get(lbl, 0) for lbl in r["labels"]]
            ids, mask, spans = backbone.encode_words(tokens)
            layers = backbone.forward_layers(ids, mask)
            for l_idx, layer in enumerate(layers):
                for s, e in spans:
                    all_words_layers[l_idx].append(layer[0, s:e].mean(dim=0).cpu())
            all_labels.extend(labels)
        t_layers = [torch.stack(c, dim=0) for c in all_words_layers]
        t_y = torch.tensor(all_labels, dtype=torch.long)
        return t_layers, t_y

    tr_l, tr_y = _extract_ner_words(
        read_task_split("ner", "train", data_dir), "  Extract ner (train)"
    )
    dv_l, dv_y = _extract_ner_words(read_task_split("ner", "dev", data_dir), "  Extract ner (dev)")
    te_l, te_y = _extract_ner_words(
        read_task_split("ner", "test", data_dir), "  Extract ner (test)"
    )

    cached = CachedClassificationData(
        task="ner",
        num_classes=len(ner_tags),
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=15, batch_size=128),
        device=backbone.device,
    )


def evaluate_syntax_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Task 1: KabSyntax (UPOS accuracy & dependency tagging)."""
    upos_tags = (
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    )
    upos2id = {t: i for i, t in enumerate(upos_tags)}
    n_layers = backbone.num_layers + 1

    def _extract_syntax(records: list[dict[str, Any]], desc: str) -> tuple[list[Tensor], Tensor]:
        chunks: list[list[Tensor]] = [[] for _ in range(n_layers)]
        labels: list[int] = []
        for r in tqdm(records, desc=desc, ncols=80, file=sys.stdout, leave=False):
            tokens = r["tokens"]
            upos_list = [upos2id.get(u, upos2id["X"]) for u in r["upos"]]
            ids, mask, spans = backbone.encode_words(tokens)
            layers = backbone.forward_layers(ids, mask)
            for l_idx, layer in enumerate(layers):
                for s, e in spans:
                    chunks[l_idx].append(layer[0, s:e].mean(dim=0).cpu())
            labels.extend(upos_list)
        return [torch.stack(c, dim=0) for c in chunks], torch.tensor(labels, dtype=torch.long)

    tr_l, tr_y = _extract_syntax(
        read_task_split("syntax", "train", data_dir), "  Extract syntax (train)"
    )
    dv_l, dv_y = _extract_syntax(
        read_task_split("syntax", "dev", data_dir), "  Extract syntax (dev)"
    )
    te_l, te_y = _extract_syntax(
        read_task_split("syntax", "test", data_dir), "  Extract syntax (test)"
    )

    cached = CachedClassificationData(
        task="syntax",
        num_classes=len(upos_tags),
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=20, batch_size=128),
        device=backbone.device,
    )


def evaluate_morph_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Task 2: KabMorph (factored aspect/mood classification)."""
    tense_map = {"prétérit": 0, "aoriste": 1, "négatif": 2, "aoriste intensif": 3}
    n_layers = backbone.num_layers + 1

    def _extract_morph(records: list[dict[str, Any]], desc: str) -> tuple[list[Tensor], Tensor]:
        chunks: list[list[Tensor]] = [[] for _ in range(n_layers)]
        labels: list[int] = []
        for r in tqdm(records, desc=desc, ncols=80, file=sys.stdout, leave=False):
            form = str(r["form"])
            t_raw = str(r.get("tense_raw", "prétérit")).lower()
            lbl = tense_map.get(t_raw, 0)
            labels.append(lbl)
            ids, mask = backbone.encode_text(form)
            layers = backbone.forward_layers(ids, mask)
            m = mask.float().unsqueeze(-1)
            for l_idx, layer in enumerate(layers):
                pooled = (layer * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)
                chunks[l_idx].append(pooled[0].cpu())
        return [torch.stack(c, dim=0) for c in chunks], torch.tensor(labels, dtype=torch.long)

    tr_l, tr_y = _extract_morph(
        read_task_split("morph", "train", data_dir)[:3000], "  Extract morph (train)"
    )
    dv_l, dv_y = _extract_morph(
        read_task_split("morph", "dev", data_dir)[:500], "  Extract morph (dev)"
    )
    te_l, te_y = _extract_morph(
        read_task_split("morph", "test", data_dir)[:500], "  Extract morph (test)"
    )

    cached = CachedClassificationData(
        task="morph",
        num_classes=4,
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=20, batch_size=64),
        device=backbone.device,
    )


def evaluate_punct_task(backbone: BackboneAdapter, data_dir: Path) -> TaskRunResult:
    """Evaluate Task 6: KabPunct (punctuation marks prediction, 5 classes)."""
    n_layers = backbone.num_layers + 1

    def _extract_punct(records: list[dict[str, Any]], desc: str) -> tuple[list[Tensor], Tensor]:
        chunks: list[list[Tensor]] = [[] for _ in range(n_layers)]
        labels: list[int] = []
        for r in tqdm(records, desc=desc, ncols=80, file=sys.stdout, leave=False):
            words = r["words"]
            punct = [PUNCT_TAG_TO_ID.get(p, 0) for p in r["punct"]]
            ids, mask, spans = backbone.encode_words(words)
            layers = backbone.forward_layers(ids, mask)
            for l_idx, layer in enumerate(layers):
                for s, e in spans:
                    chunks[l_idx].append(layer[0, s:e].mean(dim=0).cpu())
            labels.extend(punct)
        return [torch.stack(c, dim=0) for c in chunks], torch.tensor(labels, dtype=torch.long)

    tr_l, tr_y = _extract_punct(
        read_task_split("punct", "train", data_dir)[:2500], "  Extract punct (train)"
    )
    dv_l, dv_y = _extract_punct(
        read_task_split("punct", "dev", data_dir)[:500], "  Extract punct (dev)"
    )
    te_l, te_y = _extract_punct(
        read_task_split("punct", "test", data_dir)[:500], "  Extract punct (test)"
    )

    cached = CachedClassificationData(
        task="punct",
        num_classes=len(PUNCT_TAGS),
        train_layers=tr_l,
        train_labels=tr_y,
        dev_layers=dv_l,
        dev_labels=dv_y,
        test_layers=te_l,
        test_labels=te_y,
    )
    return evaluate_cached_classification(
        cached,
        lr_grid=FROZEN_LR_GRID,
        seeds=DEFAULT_SEEDS,
        schedule=FitSchedule(epochs=15, batch_size=128),
        device=backbone.device,
    )


# ---------------------------------------------------------------------------
# Runner & CLI
# ---------------------------------------------------------------------------


def run_benchmark_for_model(  # noqa: PLR0915
    adapter: BackboneAdapter, data_dir: Path
) -> dict[str, Any]:
    """Run all 6 core tasks and 3 diagnostic tasks for one encoder."""
    print("=" * 70, flush=True)
    print(
        f"Evaluating: {adapter.model_id} "
        f"({adapter.num_layers} layers, hidden={adapter.hidden_size})",
        flush=True,
    )
    print("=" * 70, flush=True)

    results: dict[str, Any] = {}
    t_start = time.time()

    print("--> [1/6] Task: KabSyntax (UPOS macro-F1 & syntax)...", flush=True)
    t0 = time.time()
    res_syntax = evaluate_syntax_task(adapter, data_dir)
    results["syntax"] = res_syntax.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] UPOS Macro-F1: "
        f"{res_syntax.primary_metric:.4f} +/- {res_syntax.primary_std:.4f}",
        flush=True,
    )

    print("--> [2/6] Task: KabMorph (factored aspect/mood)...", flush=True)
    t0 = time.time()
    res_morph = evaluate_morph_task(adapter, data_dir)
    results["morph"] = res_morph.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] Morph Macro-F1: "
        f"{res_morph.primary_metric:.4f} +/- {res_morph.primary_std:.4f}",
        flush=True,
    )

    print("--> [3/6] Task: KabNER (named entity recognition)...", flush=True)
    t0 = time.time()
    res_ner = evaluate_ner_task(adapter, data_dir)
    results["ner"] = res_ner.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] NER Macro-F1: "
        f"{res_ner.primary_metric:.4f} +/- {res_ner.primary_std:.4f}",
        flush=True,
    )

    print("--> [4/6] Task: KabSentiment (sentiment classification)...", flush=True)
    t0 = time.time()
    res_sent = evaluate_sent_task(adapter, data_dir)
    results["sent"] = res_sent.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] Sentiment Macro-F1: "
        f"{res_sent.primary_metric:.4f} +/- {res_sent.primary_std:.4f}",
        flush=True,
    )

    print("--> [5/6] Task: KabTopic (SIB-200 7-way topic)...", flush=True)
    t0 = time.time()
    res_topic = evaluate_topic_task(adapter, data_dir)
    results["topic"] = res_topic.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] Topic Macro-F1: "
        f"{res_topic.primary_metric:.4f} +/- {res_topic.primary_std:.4f}",
        flush=True,
    )

    print("--> [6/6] Task: KabPunct (punctuation marks prediction)...", flush=True)
    t0 = time.time()
    res_punct = evaluate_punct_task(adapter, data_dir)
    results["punct"] = res_punct.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] Punct Macro-F1: "
        f"{res_punct.primary_metric:.4f} +/- {res_punct.primary_std:.4f}",
        flush=True,
    )

    print("--> [Diag 1/3] Diag-Clitic (host attachment)...", flush=True)
    t0 = time.time()
    res_clitic = evaluate_clitic_task(adapter, data_dir)
    results["diag_clitic"] = res_clitic.as_dict()
    print(
        f"    [{time.time() - t0:.1f}s] Clitic Macro-F1: {res_clitic.primary_metric:.4f}",
        flush=True,
    )

    print("--> [Diag 2/3] Diag-Para (TaPaCo-kab paraphrase retrieval)...", flush=True)
    t0 = time.time()
    res_para = evaluate_para_task(adapter, data_dir)
    results["diag_para"] = res_para
    print(
        f"    [{time.time() - t0:.1f}s] mAP@R: {res_para['map_at_r']:.4f}, "
        f"MRR@10: {res_para['mrr_at_10']:.4f}",
        flush=True,
    )

    print("--> [Diag 3/3] Diag-Spell (minimal pairs accuracy)...", flush=True)
    t0 = time.time()
    res_spell = evaluate_spell_task(adapter, data_dir)
    results["diag_spell"] = res_spell
    print(
        f"    [{time.time() - t0:.1f}s] Minimal Pairs Accuracy: "
        f"{res_spell['overall_accuracy']:.4f}",
        flush=True,
    )

    print(f"Finished {adapter.model_id} in {time.time() - t_start:.1f}s total.\n", flush=True)

    return {
        "model_id": adapter.model_id,
        "results": results,
        "raw_runs": {
            "syntax": res_syntax,
            "morph": res_morph,
            "ner": res_ner,
            "sent": res_sent,
            "topic": res_topic,
            "punct": res_punct,
        },
    }


ALIAS_MAP: dict[str, str] = {
    "masinissa-73m": "agbalu/Masinissa-73M",
    "masinissa-31m": "agbalu/Masinissa-31M",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "xlm-roberta-large": "FacebookAI/xlm-roberta-large",
    "serengeti-e110": "UBC-NLP/serengeti-E110",
    "serengeti-e250": "UBC-NLP/serengeti-E250",
    "serengeti": "UBC-NLP/serengeti",
    "mmbert-small": "jhu-clsp/mmBERT-small",
    "mmbert-base": "jhu-clsp/mmBERT-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "afro-xlmr-base": "Davlan/afro-xlmr-base-76L_script",
    "afro-xlmr-large": "Davlan/afro-xlmr-large-76L",
    "rembert": "google/rembert",
}


def _model_slug(name: str) -> str:
    return name.replace("/", "_").replace("-", "_").lower()


def main() -> int:  # noqa: PLR0915
    reconfig = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig):
        reconfig(line_buffering=True)
    parser = argparse.ArgumentParser(description="KabEncBench multi-task evaluation runner")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tasks/encbench"))
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/encoder/measurements/encbench_results.json")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["masinissa-73m", "masinissa-31m", "xlm-roberta-base"],
        help="Models to evaluate (e.g. masinissa-73m, masinissa-31m, xlm-roberta-base)",
    )
    parser.add_argument(
        "--snapshot-73m",
        type=Path,
        default=Path(
            "artifacts/runs/encoder-pretrain/step-0027231-1793464372253148324/training.pt"
        ),
    )
    parser.add_argument(
        "--tokenizer-73m",
        type=Path,
        default=Path("artifacts/encoder/measurements/mammeri-tok-base-16k.model"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"KabEncBench Runner initialized on {device}.", flush=True)
    print(f"Models scheduled for evaluation: {args.models}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    runs_by_model: dict[str, dict[str, TaskRunResult]] = {}

    for idx, raw_model_name in enumerate(args.models, 1):
        model_name = ALIAS_MAP.get(raw_model_name, raw_model_name)
        print(f"\n>>> [{idx}/{len(args.models)}] Loading {model_name}...", flush=True)
        adapter: BackboneAdapter
        if model_name in ("masinissa-73m", "agbalu/Masinissa-73M"):
            adapter = Masinissa73MAdapter(args.snapshot_73m, args.tokenizer_73m, device)
        elif model_name in ("masinissa-31m", "agbalu/Masinissa-31M"):
            adapter = Masinissa31MAdapter("agbalu/Masinissa-31M", device)
        else:
            adapter = HFAutoModelAdapter(model_name, device)

        m_res = run_benchmark_for_model(adapter, args.data_dir)
        all_results[adapter.model_id] = m_res["results"]
        runs_by_model[adapter.model_id] = m_res["raw_runs"]

        # Save intermediate per-model JSON immediately
        slug = _model_slug(adapter.model_id)
        per_model_path = args.out.parent / f"encbench_{slug}.json"
        item_scores = {
            t: [list(scores) for scores in res.seed_item_scores]
            for t, res in m_res["raw_runs"].items()
        }
        per_model_payload = (
            json.dumps(
                {
                    "model_id": adapter.model_id,
                    "results": m_res["results"],
                    "item_scores": item_scores,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        per_model_path.write_text(per_model_payload, encoding="utf-8")
        print(f"Saved checkpoint results for {adapter.model_id} to {per_model_path}", flush=True)

    # Compute Bradley-Terry Elo Leaderboard
    if len(runs_by_model) >= 1:
        print("\nComputing Bradley-Terry Elo Leaderboard (Paired Cluster-Bootstrap)...", flush=True)
        anchor = (
            "FacebookAI/xlm-roberta-base"
            if "FacebookAI/xlm-roberta-base" in runs_by_model
            else next(iter(runs_by_model.keys()))
        )
        model_task_items = {
            m: {t: res.seed_item_scores for t, res in t_runs.items()}
            for m, t_runs in runs_by_model.items()
        }
        leaderboard = compute_leaderboard(
            "frozen",
            model_task_items,
            anchor_model=anchor,
            num_bootstraps=2000,
        )

        all_results["leaderboard"] = {
            "track": "frozen",
            "anchor_model": anchor,
            "ratings": [asdict(r) for r in leaderboard.ratings],
        }

        print("\n" + "=" * 75, flush=True)
        print("KabEncBench Leaderboard (Track A: Frozen Scale-Normalized Probe)", flush=True)
        print("=" * 75, flush=True)
        print(f"{'Rank':<5} {'Model':<30} {'Elo (95% CI)':<22} {'Tier':<6}", flush=True)
        print("-" * 75, flush=True)
        for r in leaderboard.ratings:
            elo_str = f"{r.elo_median:.1f} [{r.ci_low:.1f}, {r.ci_high:.1f}]"
            print(f"{r.rank:<5} {r.model:<30} {elo_str:<22} {r.tier:<6}", flush=True)
        print("=" * 75 + "\n", flush=True)

    payload = json.dumps(all_results, indent=2, ensure_ascii=False) + "\n"
    args.out.write_text(payload, encoding="utf-8")
    print(f"Wrote full results to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
