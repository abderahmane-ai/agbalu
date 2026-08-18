"""Decoding: greedy in batch, and beam search for a single string.

Both are free-running — the model is fed its own output. That is not the same
measurement as a teacher-forced pass, which scores every position against the gold
prefix and so cannot expose an error that compounds. The first evaluation of this model
reported a teacher-forced number under a beam-search heading; the two are separated here
so a caller has to pick one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from agbalu.tifinagh.config import ModelConfig
from agbalu.tifinagh.model import CharTransformer
from agbalu.tifinagh.tokenizer import BOS_ID, EOS_ID, PAD_ID, CharTokenizer

log: Final = logging.getLogger("agbalu.tifinagh.infer")

MAX_LENGTH: Final = 256
"""Longest output in characters. The corpus caps at 35 words, so this is slack, not a
budget — a decode that reaches it has degenerated and the caller should see that."""

LENGTH_PENALTY: Final = 0.6

DEFAULT_CHECKPOINT: Final = Path("artifacts/checkpoints/Juba-27M/juba_final.pt")


class CheckpointError(Exception):
    """A checkpoint that cannot produce a model."""


class Transliterator:
    """A loaded model, decoding strings to strings."""

    def __init__(
        self, model: CharTransformer, tokenizer: CharTokenizer, device: torch.device
    ) -> None:
        if tokenizer.vocab_size > model.config.vocab_size:
            # Otherwise the mismatch surfaces as `IndexError: index out of range in self`
            # from inside `nn.Embedding`, which names neither the tokenizer nor the model.
            message = (
                f"the tokenizer has {tokenizer.vocab_size} ids and the model has "
                f"{model.config.vocab_size} embedding rows"
            )
            raise CheckpointError(message)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> Transliterator:
        """A training checkpoint, or a published directory. Both, because both exist.

        The shapes come from the config the source carries, never from `ModelConfig()`:
        defaults load a differently-shaped checkpoint only when they happen to match, and
        otherwise raise inside `load_state_dict` naming a tensor rather than the cause.
        """
        target = torch.device(device)
        if path.is_dir():
            return cls(_from_directory(path, target), CharTokenizer(), target)
        if not path.is_file():
            message = f"no checkpoint or published directory at {path}"
            raise CheckpointError(message)
        state = torch.load(path, map_location=target, weights_only=True)
        if "model_state_dict" not in state:
            message = f"{path} carries no model_state_dict"
            raise CheckpointError(message)
        model = CharTransformer(ModelConfig(**state.get("config", {})))
        model.load_state_dict(state["model_state_dict"])
        return cls(model, CharTokenizer(), target)

    @torch.no_grad()
    def greedy_batch(self, texts: Sequence[str], max_length: int = MAX_LENGTH) -> list[str]:
        """Free-running greedy decoding over a whole batch at once.

        The single-string loop it replaces re-ran the decoder once per sentence per
        character; batching turns 49,795 sentences from a job that does not finish on a
        CPU into one that does. Sequences that have emitted `[EOS]` are padded rather
        than removed, so a row's position in the output matches its position in the
        input for the whole decode.
        """
        if not texts:
            return []
        context, _ = self._encode_batch(texts)
        batch = len(texts)

        tokens = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=self.device)
        for _ in range(max_length):
            logits = self.model.decode(tokens, context)
            following = logits[:, -1, :].argmax(dim=-1)
            following = torch.where(finished, torch.full_like(following, PAD_ID), following)
            tokens = torch.cat([tokens, following.unsqueeze(1)], dim=1)
            finished |= following == EOS_ID
            if bool(finished.all()):
                break
        return [self.tokenizer.decode(row.tolist()) for row in tokens]

    @torch.no_grad()
    def transliterate(self, text: str, num_beams: int = 4, max_length: int = MAX_LENGTH) -> str:
        """One string. `num_beams` of 1 or less is greedy."""
        if num_beams <= 1:
            return self.greedy_batch([text], max_length)[0]
        context, _ = self._encode_batch([text])
        return self._beam(context, num_beams, max_length)

    def _encode_batch(self, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
        encoded = [self.tokenizer.encode(text) for text in texts]
        width = max(len(ids) for ids in encoded)
        padded = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
        for row, ids in enumerate(encoded):
            padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded = padded.to(self.device)
        return self.model.encode(padded), padded

    def _beam(self, context: Tensor, num_beams: int, max_length: int) -> str:
        """Beam search, expanding every live beam in one decoder call.

        Beams are stacked into the batch dimension and the encoder output is expanded to
        match, so the cost per step is one forward rather than one per beam.
        """
        beams: list[tuple[list[int], float]] = [([BOS_ID], 0.0)]
        completed: list[tuple[list[int], float]] = []

        for _ in range(max_length):
            live = [beam for beam in beams if beam[0][-1] != EOS_ID]
            completed.extend(beam for beam in beams if beam[0][-1] == EOS_ID)
            if not live or len(completed) >= num_beams:
                break

            prefixes = torch.tensor([sequence for sequence, _ in live], device=self.device)
            logits = self.model.decode(prefixes, context.expand(len(live), -1, -1))
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            top_scores, top_ids = torch.topk(log_probs, k=num_beams, dim=-1)

            candidates: list[tuple[list[int], float]] = []
            for index, (sequence, score) in enumerate(live):
                for step_score, token in zip(
                    top_scores[index].tolist(), top_ids[index].tolist(), strict=True
                ):
                    candidates.append(([*sequence, token], score + step_score))
            beams = sorted(candidates, key=_normalised, reverse=True)[:num_beams]

        best = max([*completed, *beams], key=_normalised)[0]
        return self.tokenizer.decode(best)


def _normalised(beam: tuple[list[int], float]) -> float:
    """Log probability divided by length, so a short beam does not win by being short."""
    sequence, score = beam
    normalised: float = score / len(sequence) ** LENGTH_PENALTY
    return normalised


def _from_directory(directory: Path, device: torch.device) -> CharTransformer:
    """A published release: `config.json` plus `model.safetensors`.

    The export drops the output projection because it is the embedding under a second
    name, and safetensors refuses to write shared storage. `assign=False` then re-ties it
    on load, so `strict=False` here is not slack — it is the one key the export is
    documented to omit, and any other missing key is reported.
    """
    from safetensors.torch import load_file  # noqa: PLC0415

    config_path, weights_path = directory / "config.json", directory / "model.safetensors"
    for required in (config_path, weights_path):
        if not required.is_file():
            message = f"{directory} is not a published model: no {required.name}"
            raise CheckpointError(message)

    # Only the fields this dataclass declares. A published `config.json` is a transformers
    # config — it also carries `auto_map`, `architectures`, `model_type` and whatever else
    # the library writes — and a frozen dataclass rejects every one of them by name.
    declared = {field.name for field in fields(ModelConfig)}
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    model = CharTransformer(ModelConfig(**{k: v for k, v in saved.items() if k in declared}))
    # `remove_duplicate=False`, or the tied head never appears and every published model
    # reads as missing a key it was correct to omit.
    tied = {
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter is model.embed.weight
    }
    incompatible = model.load_state_dict(load_file(weights_path, device=str(device)), strict=False)
    unexpected = set(incompatible.missing_keys) - tied
    if unexpected or incompatible.unexpected_keys:
        message = (
            f"{weights_path} does not match the config: missing {sorted(unexpected)}, "
            f"unexpected {sorted(incompatible.unexpected_keys)}"
        )
        raise CheckpointError(message)
    return model
