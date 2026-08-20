"""Inference engine for orthography standardisation and diacritic restoration.

Provides fast greedy decoding and batch standardization on CPU, MPS, and CUDA.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from agbalu.standardise.model import CharTransformer
from agbalu.standardise.tokenizer import Tokenizer

DEFAULT_CHECKPOINT = Path("artifacts/boulifa/boulifa_best.pt")


def resolve_checkpoint(checkpoint_path: Path | str = DEFAULT_CHECKPOINT) -> Path:
    """Resolve a checkpoint path, falling back to known artifact locations."""
    path = Path(checkpoint_path)
    if path.is_file():
        return path
    candidates = (
        Path("artifacts/boulifa/boulifa_best.pt"),
        Path("artifacts/boulifa/boulifa_final.pt"),
        Path("artifacts/checkpoints/Boulifa-48M/boulifa_best.pt"),
        Path("artifacts/checkpoints/Boulifa-48M/boulifa_final.pt"),
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return path


class Standardiser:
    """Inference wrapper for the 47.8M standardisation model."""

    def __init__(
        self,
        model: CharTransformer,
        tokenizer: Tokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: torch.device | None = None,
    ) -> Standardiser:
        path = resolve_checkpoint(checkpoint_path)
        tokenizer = Tokenizer.build()
        model = CharTransformer()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        return cls(model=model, tokenizer=tokenizer, device=device)

    @torch.no_grad()
    def standardise(self, text: str, *, max_length: int = 512) -> str:
        """Standardise a single text from informal/French typing to canonical Kabyle Latin."""
        if not text.strip():
            return text

        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        memory = self.model.encode(input_tensor)

        # Autoregressive greedy decoding
        generated = [self.tokenizer.bos_id]
        max_gen = min(max_length, len(token_ids) * 2 + 10)

        for _ in range(max_gen):
            dec_in = torch.tensor([generated], dtype=torch.long, device=self.device)
            seq_len = dec_in.shape[1]
            causal_mask = (
                torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=self.device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            decoded = self.model.decode(dec_in, memory, self_mask=causal_mask)
            logits = self.model.output_projection(decoded[:, -1:, :])
            next_token = int(logits.argmax(dim=-1).item())

            if next_token == self.tokenizer.eos_id:
                break
            generated.append(next_token)

        return self.tokenizer.decode(generated[1:], skip_special_tokens=True)

    @torch.no_grad()
    def standardise_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> list[str]:
        """Standardise a batch of texts."""
        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend([self.standardise(t, max_length=max_length) for t in batch])
        return results


def standardise(text: str, *, checkpoint: Path | str = DEFAULT_CHECKPOINT) -> str:
    """Convenience helper to standardise a sentence."""
    engine = Standardiser.load(checkpoint)
    return engine.standardise(text)
