"""Load the native Qwen3 codec generation graph with explicit tensor accounting."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional
from transformers import Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

from agbalu.codec_tts.configuration import Geometry
from agbalu.codec_tts.layout import CODEBOOKS, VOCABULARY


class Generator(nn.Module):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        self.geometry = geometry
        config = Qwen3Config(
            hidden_size=geometry.hidden_size,
            intermediate_size=geometry.intermediate_size,
            num_hidden_layers=geometry.layers,
            num_attention_heads=geometry.heads,
            num_key_value_heads=geometry.kv_heads,
            head_dim=geometry.head_dim,
            vocab_size=geometry.text_vocabulary,
            rms_norm_eps=geometry.rms_epsilon,
            rope_parameters={"rope_type": "default", "rope_theta": geometry.rope_theta},
            max_position_embeddings=32768,
            attention_dropout=0,
            use_cache=False,
        )
        self.body = Qwen3Model(config)
        self.audio_embedding = nn.Embedding(CODEBOOKS * VOCABULARY, geometry.hidden_size)

    def embed_audio(self, codes: Tensor) -> Tensor:
        offsets = torch.arange(CODEBOOKS, device=codes.device) * VOCABULARY
        embedded: Tensor = self.audio_embedding(codes.long() + offsets)
        return embedded.sum(dim=-2)

    def enable_activation_checkpointing(self) -> None:
        self.body.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def hidden(self, text_ids: Tensor, codes: Tensor, audio_mask: Tensor, valid: Tensor) -> Tensor:
        embedded: Tensor = self.body.embed_tokens(text_ids)
        embedded = torch.where(audio_mask.unsqueeze(-1), self.embed_audio(codes), embedded)
        output = self.body(inputs_embeds=embedded, attention_mask=valid, use_cache=False)
        if not isinstance(output, BaseModelOutputWithPast):
            message = f"unexpected Qwen3 output: {type(output).__name__}"
            raise TypeError(message)
        hidden = output.last_hidden_state
        if not isinstance(hidden, Tensor):
            message = "Qwen3 returned no hidden-state tensor"
            raise TypeError(message)
        return hidden

    def classify(self, hidden: Tensor) -> Tensor:
        logits = functional.linear(hidden, self.audio_embedding.weight)
        return logits.reshape(*hidden.shape[:-1], CODEBOOKS, VOCABULARY)


def map_weight(name: str) -> str | None:
    if name.startswith("tied.embedding.modality_embeddings.0.model."):
        return None
    if name.startswith("tied.head."):
        return None
    prefixes = (
        ("tied.embedding.text_embedding.", "body.embed_tokens."),
        ("tied.embedding.modality_embeddings.0.embedding.", "audio_embedding."),
        ("body.", "body."),
    )
    for source, destination in prefixes:
        if name.startswith(source):
            return destination + name.removeprefix(source)
    message = f"unrecognized native checkpoint tensor {name!r}"
    raise ValueError(message)


def load_generator(
    path: Path, geometry: Geometry, device: torch.device
) -> tuple[Generator, dict[str, object]]:
    with torch.device("meta"):
        model = Generator(geometry)
    expected = model.state_dict()
    seen: set[str] = set()
    dropped: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        source_keys = set(checkpoint.keys())
        for source in source_keys:
            destination = map_weight(source)
            if destination is None:
                dropped.append(source)
                continue
            if destination not in expected or destination in seen:
                message = f"unexpected/duplicate tensor {source!r} -> {destination!r} at {path}"
                raise ValueError(message)
            tensor = checkpoint.get_tensor(source)
            if tensor.shape != expected[destination].shape:
                message = f"shape mismatch {path}:{source}: {tensor.shape}"
                message += f" != {expected[destination].shape}"
                raise ValueError(message)
            model.load_state_dict({destination: tensor.to(device)}, strict=False, assign=True)
            seen.add(destination)
        missing = set(expected) - seen
        if missing:
            message = f"missing native tensors at {path}: {sorted(missing)}"
            raise ValueError(message)
        tied_pairs = (
            ("tied.head.text_head.weight", "tied.embedding.text_embedding.weight"),
            (
                "tied.head.modality_heads.0.weight",
                "tied.embedding.modality_embeddings.0.embedding.weight",
            ),
        )
        for left, right in tied_pairs:
            if (
                left not in source_keys
                or right not in source_keys
                or not torch.equal(checkpoint.get_tensor(left), checkpoint.get_tensor(right))
            ):
                message = f"missing or unequal tied tensors at {path}: {left}, {right}"
                raise ValueError(message)
    # Derived frequencies are not serialized and remain meta after assign-only loading.
    model.body.rotary_emb = Qwen3RotaryEmbedding(model.body.config).to(device)
    model.requires_grad_(False)
    return model, {"loaded": sorted(seen), "dropped": dropped, "ties_verified": True}
