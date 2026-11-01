"""Stage Belaid-75M for release on Hugging Face."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


CONFIGURATION_BELAID = '''"""Configuration for the Belaid-75M punctuation and casing restorer."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig

PUNCTUATION_LABELS = [
    "NONE",
    "COMMA",
    "PERIOD",
    "QUESTION",
    "EXCLAMATION",
    "COLON",
    "SEMICOLON",
    "ELLIPSIS",
    "INTERROBANG",
]
CASE_LABELS = ["LOWER", "INITIAL_CAP", "ALL_CAP"]
TEMPLATES = ["@"]


class BelaidConfig(PreTrainedConfig):
    model_type = "belaid"

    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 512,
        intermediate_size: int = 1408,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 20,
        block_layers: int = 4,
        position_buckets: int = 32,
        max_position_embeddings: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        epsilon: float = 1e-6,
        punctuation_labels: list[str] | None = None,
        case_labels: list[str] | None = None,
        templates: list[str] | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.block_layers = block_layers
        self.position_buckets = position_buckets
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.layer_norm_eps = layer_norm_eps
        self.epsilon = epsilon
        self.punctuation_labels = punctuation_labels or list(PUNCTUATION_LABELS)
        self.case_labels = case_labels or list(CASE_LABELS)
        self.templates = templates or list(TEMPLATES)

        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("pad_token_id", pad_token_id)
        kwargs.setdefault("bos_token_id", bos_token_id)
        kwargs.setdefault("eos_token_id", eos_token_id)
        super().__init__(**kwargs)

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads


__all__ = ["CASE_LABELS", "PUNCTUATION_LABELS", "TEMPLATES", "BelaidConfig"]
'''

MODELING_BELAID = '''"""Belaid-75M: punctuation and casing restoration over Kabyle ASR output."""

from __future__ import annotations

import bisect
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel

from .configuration_belaid import BelaidConfig

MARKS = ("", ",", ".", "?", "!", ":", ";", "…", "?!")
CASES = ("LOWER", "INITIAL_CAP", "ALL_CAP")
PRIMARY = "@"
TERMINAL_LABELS = frozenset({2, 3, 4, 8})  # . ? ! ?!
OPENERS = {"«": "»", "“": "”", "(": ")", "[": "]"}
CASE_COMPONENT = re.compile(r"[^\\-\'’]+")


def is_letter_or_number(char: str) -> bool:
    return unicodedata.category(char)[0] in "LMN"


def lexical_spans(text: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        if not is_letter_or_number(text[cursor]):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < length:
            char = text[cursor]
            connected = (
                char in "-\'’" and cursor + 1 < length and is_letter_or_number(text[cursor + 1])
            )
            numeric_separator = (
                char in ".,:/"
                and cursor + 1 < length
                and text[cursor - 1].isdecimal()
                and text[cursor + 1].isdecimal()
            )
            if not is_letter_or_number(char) and not connected and not numeric_separator:
                break
            cursor += 1
        result.append((start, cursor))
    return tuple(result)


def apply_case(word: str, label: int | tuple[int, ...]) -> str:
    matches = list(CASE_COMPONENT.finditer(word))
    if not matches:
        return word
    labels = label if isinstance(label, tuple) else (label,) * len(matches)
    result: list[str] = []
    cursor = 0
    for match, item in zip(matches, labels, strict=False):
        start, end = match.span()
        result.append(word[cursor:start])
        part = word[start:end]
        if item == 1:
            part = part[:1].upper() + part[1:].lower()
        elif item == 2:
            part = part.upper()
        else:
            part = part.lower()
        result.append(part)
        cursor = end
    result.append(word[cursor:])
    return "".join(result)


def log_bucket_position(offsets: Tensor, buckets: int, max_distance: int) -> Tensor:
    sign = torch.sign(offsets)
    distance = torch.abs(offsets)
    mid = buckets // 2
    boundary = float(max_distance - 1)
    scaled = torch.clamp(distance, min=1).float()
    log_bucket = (
        torch.ceil(torch.log(scaled / mid) / math.log(boundary / mid) * (mid - 1)).long() + mid
    )
    clipped = torch.clamp(log_bucket, max=buckets - 1)
    bucket = torch.where(distance < mid, distance, clipped)
    return buckets - 1 + bucket * sign


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight.to(value.dtype)


class DepthAttention(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.norm = RMSNorm(width, eps)

    def weights(self, values: Tensor) -> Tensor:
        keys = self.norm.forward(values)
        return (keys.float() * self.query.float()).sum(-1).softmax(0)

    def forward(self, blocks: list[Tensor], partial: Tensor | None) -> Tensor:
        values = torch.stack([*blocks, *([] if partial is None else [partial])])
        weights = self.weights(values).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(0)


class Attention(nn.Module):
    def __init__(self, config: BelaidConfig) -> None:
        super().__init__()
        self.config = config
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(3 * self.head_dim)
        width = config.hidden_size

        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.epsilon)
        self.k_norm = RMSNorm(self.head_dim, config.epsilon)

    def forward(
        self,
        hidden: Tensor,
        relative: tuple[Tensor, Tensor],
        positions: Tensor,
        valid: Tensor,
    ) -> Tensor:
        qkv: Tensor = self.qkv(hidden)
        q, k, v = qkv.view(-1, 3, self.heads, self.head_dim).unbind(1)
        q = self.q_norm(q).transpose(0, 1).unsqueeze(0)
        k = self.k_norm(k).transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        position_q, position_k = relative
        with torch.autocast(q.device.type, enabled=False):
            c2p = torch.matmul(q[0].float(), position_k.float().transpose(-1, -2))
            p2c = torch.matmul(k[0].float(), position_q.float().transpose(-1, -2))

        length = hidden.size(0)
        qi = torch.arange(length, device=hidden.device)[:, None]
        ki = torch.arange(length, device=hidden.device)[None, :]
        heads = torch.arange(self.heads, device=hidden.device)[:, None, None]

        distance = positions[qi] - positions[ki]
        forward_buckets = log_bucket_position(
            distance, self.config.position_buckets, self.config.max_position_embeddings
        )
        reverse_buckets = log_bucket_position(
            -distance, self.config.position_buckets, self.config.max_position_embeddings
        )

        with torch.autocast(q.device.type, enabled=False):
            score = q[0].float() @ k[0].float().transpose(-1, -2)
            combined = (
                score + c2p[heads, qi, forward_buckets] + p2c[heads, ki, reverse_buckets]
            ) * self.scale
            allowed = valid[qi] & valid[ki]
            allowed = allowed | ((~valid[qi]) & (qi == ki))
            score = torch.where(allowed, combined, float("-inf"))
            context = score.softmax(-1) @ v[0].float()

        merged = context.transpose(0, 1).reshape(-1, self.config.hidden_size).to(hidden.dtype)
        projected: Tensor = self.output(merged)
        return projected * valid.unsqueeze(-1)


class Layer(nn.Module):
    def __init__(self, config: BelaidConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.attention = Attention(config)
        self.attention_route = DepthAttention(width, config.epsilon)
        self.feedforward_route = DepthAttention(width, config.epsilon)
        self.attention_norm = RMSNorm(width, config.epsilon)
        self.feedforward_norm = RMSNorm(width, config.epsilon)
        self.gate_up = nn.Linear(width, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, width, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        blocks: list[Tensor],
        partial: Tensor | None,
        relative: tuple[Tensor, Tensor],
        positions: Tensor,
        valid: Tensor,
    ) -> Tensor:
        mixed = self.attention_route(blocks, partial)
        delta = self.dropout(
            self.attention(self.attention_norm(mixed), relative, positions, valid)
        )
        partial = delta if partial is None else partial + delta
        mixed = self.feedforward_route(blocks, partial)
        projected: Tensor = self.gate_up(self.feedforward_norm(mixed))
        gate, value = projected.chunk(2, -1)
        change: Tensor = self.down(functional.silu(gate) * value)
        return partial + self.dropout(change)


class Encoder(nn.Module):
    def __init__(self, config: BelaidConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.relative = nn.Parameter(
            torch.empty(2 * config.position_buckets - 1, config.hidden_size)
        )
        self.position_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.position_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_hidden_layers))
        self.final_route = DepthAttention(config.hidden_size, config.epsilon)
        self.final_norm = RMSNorm(config.hidden_size, config.epsilon)

    def forward(self, input_ids: Tensor, positions: Tensor, valid: Tensor) -> Tensor:
        hidden = self.embedding(input_ids)
        rel_q = (
            self.position_q(self.relative)
            .view(-1, self.config.num_attention_heads, self.config.head_size)
            .transpose(0, 1)
        )
        rel_k = (
            self.position_k(self.relative)
            .view(-1, self.config.num_attention_heads, self.config.head_size)
            .transpose(0, 1)
        )
        relative = (rel_q, rel_k)

        blocks = [hidden]
        partial: Tensor | None = None
        for i, layer in enumerate(self.layers):
            partial = layer(blocks, partial, relative, positions, valid)
            if (i + 1) % self.config.block_layers == 0:
                blocks.append(partial)
                partial = None

        routed = self.final_route(blocks, partial)
        return self.final_norm(routed)


def projection(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(0.1),
    )


class BelaidForTokenClassification(PreTrainedModel):
    config_class = BelaidConfig
    base_model_prefix = "belaid"

    def __init__(self, config: BelaidConfig) -> None:
        super().__init__(config)
        self.config = config
        self.encoder = Encoder(config)
        width = config.hidden_size
        self.word_projection = projection(3 * width, width)
        self.boundary_projection = projection(4 * width, width)
        self.document_edges = nn.Embedding(2, width)
        self.presence = nn.Linear(width, 1)
        self.punctuation = nn.Linear(width, len(config.punctuation_labels) - 1)
        self.casing = nn.Sequential(
            nn.Linear(2 * width + 3, width // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width // 2, len(config.case_labels)),
        )
        self.structure = nn.Linear(width, len(config.templates))
        self.post_init()

    def forward_features(
        self,
        input_ids: Tensor,
        word_spans: Tensor,
        case_spans: Tensor,
        case_words: Tensor,
        case_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        length = input_ids.size(0)
        positions = torch.arange(length, device=input_ids.device)
        valid = torch.ones(length, dtype=torch.bool, device=input_ids.device)

        hidden = self.encoder(input_ids, positions, valid)
        starts, ends = word_spans.unbind(-1)
        first = hidden[starts]
        last = hidden[(ends - 1).clamp(min=0)]

        prefix = functional.pad(hidden.float().cumsum(0), (0, 0, 1, 0))
        mean = (prefix[ends] - prefix[starts]) / (ends - starts).clamp(min=1).unsqueeze(-1)
        words: Tensor = self.word_projection(torch.cat((first, last, mean.to(hidden.dtype)), -1))

        width = hidden.size(-1)
        start_edge = self.document_edges.weight[0:1]
        end_edge = self.document_edges.weight[1:2]
        left = torch.cat((start_edge, words), dim=0)
        right = torch.cat((words, end_edge), dim=0)
        boundary: Tensor = self.boundary_projection(
            torch.cat((left, right, left * right, right - left), -1)
        )

        p = self.presence(boundary).squeeze(-1)
        types = self.punctuation(boundary)
        struct = self.structure(boundary)

        case_starts, case_ends = case_spans.unbind(-1)
        c_first = hidden[case_starts]
        c_last = hidden[(case_ends - 1).clamp(min=0)]
        word_ctx = words[case_words]
        case_input = torch.cat(
            (c_first + c_last, word_ctx, case_features.to(hidden.dtype)), dim=-1
        )
        cases = self.casing(case_input)

        return p, types, cases, struct

    @torch.inference_mode()
    def restore(self, texts: str | list[str], tokenizer: Any) -> list[str]:
        wanted = [texts] if isinstance(texts, str) else list(texts)
        device = next(self.parameters()).device
        results: list[str] = []

        for raw_text in wanted:
            normed = unicodedata.normalize("NFC", raw_text)
            spans = lexical_spans(normed)
            words = tuple(normed[s:e] for s, e in spans)
            if not words:
                results.append("")
                continue

            # Encode subwords
            sub_ids: list[int] = []
            word_sub_spans: list[tuple[int, int]] = []
            case_sub_spans: list[tuple[int, int]] = []
            case_word_idx: list[int] = []
            case_feats: list[list[float]] = []

            for w_idx, word in enumerate(words):
                w_start_sub = len(sub_ids) + 1  # account for BOS
                encoded_w = tokenizer.encode(word.lower(), add_special_tokens=False)
                if not encoded_w:
                    encoded_w = [1]  # UNK
                sub_ids.extend(encoded_w)
                w_end_sub = len(sub_ids) + 1
                word_sub_spans.append((w_start_sub, w_end_sub))

                # Components
                matches = list(CASE_COMPONENT.finditer(word))
                c_total = len(matches)
                for c_idx, match in enumerate(matches):
                    # byte offsets approx to subword span
                    c_span_start = w_start_sub
                    c_span_end = w_end_sub
                    case_sub_spans.append((c_span_start, c_span_end))
                    case_word_idx.append(w_idx)
                    inv_c = 1.0 / max(1, c_total)
                    case_feats.append([c_idx * inv_c, (c_idx + 1) * inv_c, inv_c])

            full_ids = [2, *sub_ids, 3]  # BOS, ..., EOS
            input_ids = torch.tensor(full_ids, dtype=torch.long, device=device)
            w_spans = torch.tensor(word_sub_spans, dtype=torch.long, device=device)
            c_spans = (
                torch.tensor(case_sub_spans, dtype=torch.long, device=device)
                if case_sub_spans
                else torch.zeros((0, 2), dtype=torch.long, device=device)
            )
            c_words = (
                torch.tensor(case_word_idx, dtype=torch.long, device=device)
                if case_word_idx
                else torch.zeros((0,), dtype=torch.long, device=device)
            )
            c_feats = (
                torch.tensor(case_feats, dtype=torch.float32, device=device)
                if case_feats
                else torch.zeros((0, 3), dtype=torch.float32, device=device)
            )

            p, types, cases, struct = self.forward_features(
                input_ids, w_spans, c_spans, c_words, c_feats
            )

            # Compute logits
            p_np = p.float().cpu().numpy()
            t_np = types.float().cpu().numpy()
            c_np = cases.float().cpu().numpy() if cases.size(0) > 0 else np.zeros((0, 3))

            # Joint logits
            log_present = -np.logaddexp(0, -p_np)
            log_none = -np.logaddexp(0, p_np)
            shifted = t_np - t_np.max(-1, keepdims=True)
            log_types = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            joint = np.concatenate(
                (log_none[:, None], log_present[:, None] + log_types), axis=1
            )
            marks_pred = joint.argmax(-1)

            # Assign case predictions per word
            c_preds = c_np.argmax(-1) if c_np.size > 0 else np.zeros((0,), dtype=int)
            word_cases: list[int | tuple[int, ...]] = []
            curr_c = 0
            for word in words:
                comp_count = len(list(CASE_COMPONENT.finditer(word)))
                if comp_count == 1:
                    word_cases.append(int(c_preds[curr_c]))
                    curr_c += 1
                elif comp_count > 1:
                    word_cases.append(
                        tuple(int(c_preds[i]) for i in range(curr_c, curr_c + comp_count))
                    )
                    curr_c += comp_count
                else:
                    word_cases.append(0)

            # Render text
            pieces: list[str] = []
            for index, word in enumerate(words):
                mark_before = MARKS[marks_pred[index]]
                if mark_before:
                    pieces.append(mark_before + " ")
                elif index > 0:
                    pieces.append(" ")

                w_case = word_cases[index]
                if marks_pred[index] in TERMINAL_LABELS:
                    if isinstance(w_case, tuple):
                        w_case = (max(w_case[0], 1), *w_case[1:])
                    else:
                        w_case = max(w_case, 1)

                pieces.append(apply_case(word, w_case))

            final_mark = MARKS[marks_pred[-1]]
            if final_mark:
                pieces.append(final_mark)

            results.append("".join(pieces).strip())

        return results


__all__ = ["BelaidConfig", "BelaidForTokenClassification"]
'''


def stage_belaid_75m(root: Path) -> Path:
    release_dir = root / "artifacts" / "release" / "Belaid-75M"
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    ckpt_path = (
        root
        / "artifacts"
        / "runs"
        / "belaid-75m-v3"
        / "step-0012071-1793564922048870524"
        / "training.pt"
    )
    card_path = root / "docs" / "cards" / "belaid-75m.md"
    hub_belaid = root / "artifacts" / "hub_models" / "belaid"

    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ckpt["model"]

    # Save weights as safetensors
    safetensors_path = release_dir / "model.safetensors"
    print(f"Saving safetensors to {safetensors_path}...")
    save_file(weights, safetensors_path)

    # Copy card as README.md
    print("Writing README.md...")
    shutil.copyfile(card_path, release_dir / "README.md")

    # Copy tokenizer and git attributes
    for fname in (
        ".gitattributes",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        src_f = hub_belaid / fname
        if src_f.is_file():
            shutil.copyfile(src_f, release_dir / fname)
            print(f"Copied {fname}")

    # Copy Mammeri-Tok model file
    mammeri_model = root / "artifacts" / "encoder" / "measurements" / "mammeri-tok-base-16k.model"
    shutil.copyfile(mammeri_model, release_dir / "mammeri-tok-base-16k.model")
    print("Copied mammeri-tok-base-16k.model")

    # Write configuration and modeling
    print("Writing configuration_belaid.py and modeling_belaid.py...")
    (release_dir / "configuration_belaid.py").write_text(CONFIGURATION_BELAID, encoding="utf-8")
    (release_dir / "modeling_belaid.py").write_text(MODELING_BELAID, encoding="utf-8")
    (release_dir / "__init__.py").write_text(
        "from .configuration_belaid import BelaidConfig\n"
        "from .modeling_belaid import BelaidForTokenClassification\n\n"
        '__all__ = ["BelaidConfig", "BelaidForTokenClassification"]\n',
        encoding="utf-8",
    )

    # Write config.json
    config_dict = {
        "architectures": ["BelaidForTokenClassification"],
        "attention_probs_dropout_prob": 0.1,
        "auto_map": {
            "AutoConfig": "configuration_belaid.BelaidConfig",
            "AutoModelForTokenClassification": "modeling_belaid.BelaidForTokenClassification",
            "AutoModel": "modeling_belaid.BelaidForTokenClassification",
        },
        "block_layers": 4,
        "bos_token_id": 2,
        "case_labels": ["LOWER", "INITIAL_CAP", "ALL_CAP"],
        "classifier_dropout": 0.1,
        "eos_token_id": 3,
        "epsilon": 1e-06,
        "hidden_dropout_prob": 0.1,
        "hidden_size": 512,
        "intermediate_size": 1408,
        "layer_norm_eps": 1e-06,
        "max_position_embeddings": 1024,
        "model_type": "belaid",
        "num_attention_heads": 8,
        "num_hidden_layers": 20,
        "pad_token_id": 0,
        "position_buckets": 32,
        "punctuation_labels": [
            "NONE",
            "COMMA",
            "PERIOD",
            "QUESTION",
            "EXCLAMATION",
            "COLON",
            "SEMICOLON",
            "ELLIPSIS",
            "INTERROBANG",
        ],
        "templates": ["@"],
        "tie_word_embeddings": True,
        "transformers_version": "5.12.1",
        "vocab_size": 16000,
    }
    (release_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Write export.stats.json
    files_stats = [
        {"name": f.name, "bytes": f.stat().st_size, "sha256": digest(f)}
        for f in sorted(release_dir.iterdir())
        if f.name != "export.stats.json"
    ]
    (release_dir / "export.stats.json").write_text(
        json.dumps({"staged": files_stats}, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Staged {len(files_stats)} files in {release_dir}.")
    return release_dir


if __name__ == "__main__":
    stage_belaid_75m(Path.cwd())
