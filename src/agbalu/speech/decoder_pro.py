"""Fadhma-Pro: Tri-Level Decoding Engine (pyctcdecode + KabLex Lexicon + KenLM + Masinissa-31M)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from numpy.typing import NBitBase, NDArray
from pyctcdecode.decoder import build_ctcdecoder

from agbalu.speech.vocabulary_pro import WORD_DELIMITER, VocabularyPro

if TYPE_CHECKING:
    from pyctcdecode.decoder import BeamSearchDecoderCTC
    from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

# The width `pyctcdecode` declares. numpy's dtype parameter is invariant, so `float32` is
# not assignable to it.
type LogitArray = NDArray[np.floating[NBitBase]]

DEFAULT_ALPHA: Final[float] = 0.5
DEFAULT_BETA: Final[float] = 1.5
DEFAULT_BEAM_WIDTH: Final[int] = 100

KABLEX_PATH: Final = Path("data/raw/hf.boffire.kablex-lexicon-kab/kablex.json")


def load_kablex_unigrams(lexicon_path: Path = KABLEX_PATH) -> list[str]:
    """Extract distinct verified words from KabLex lexicon as unigram constraints."""
    if not lexicon_path.exists():
        log.warning("KabLex path %s does not exist; using open vocabulary", lexicon_path)
        return []

    with lexicon_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        words = list(data.keys())
    elif isinstance(data, list):
        words = [
            item["word"] if isinstance(item, dict) and "word" in item else str(item)
            for item in data
        ]
    else:
        words = []
    return sorted({w.strip().lower() for w in words if w.strip()})


@dataclass(frozen=True, slots=True)
class DecoderConfigPro:
    """Configuration for Tri-Level Decoder."""

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    beam_width: int = DEFAULT_BEAM_WIDTH


class TriLevelDecoderPro:
    """Fadhma-Pro Tri-Level CTC Decoder:

    Stage 1: Frame-level CTC acoustic emission (Wav2Vec2 / omniASR-v2).
    Stage 2: pyctcdecode Beam Search with 6-gram KenLM shallow fusion & KabLex lexicon constraint.
    Stage 3: Masinissa-31M pseudo-log-likelihood neural rescoring over top 50 beam candidates.
    """

    def __init__(
        self,
        vocab: VocabularyPro | None = None,
        kenlm_path: Path | None = None,
        unigrams: list[str] | None = None,
        config: DecoderConfigPro | None = None,
    ) -> None:
        self.vocab = vocab or VocabularyPro()
        self.config = config or DecoderConfigPro()

        # pyctcdecode expects blank as "" at index 0 and space as " "
        tokens = self.vocab.get_tokens()
        labels = [
            "" if i == 0 else (" " if t == WORD_DELIMITER else t) for i, t in enumerate(tokens)
        ]

        lm_file = str(kenlm_path) if kenlm_path and kenlm_path.exists() else None
        self.decoder: BeamSearchDecoderCTC = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=lm_file,
            unigrams=unigrams,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )

    def decode_greedy(self, logits: LogitArray | torch.Tensor) -> str:
        """Standard greedy CTC argmax collapse."""
        arr = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
        pred_ids = np.argmax(arr, axis=-1)
        tokens = self.vocab.get_tokens()

        out_chars: list[str] = []
        prev_id = -1
        for idx in pred_ids:
            if idx not in (prev_id, 0):  # 0 is [PAD] blank
                tok = tokens[idx]
                if tok == WORD_DELIMITER:
                    out_chars.append(" ")
                elif tok != "[UNK]":
                    out_chars.append(tok)
            prev_id = idx

        return " ".join("".join(out_chars).split())

    def decode_beam(self, logits: LogitArray | torch.Tensor, beam_width: int | None = None) -> str:
        """Decode a single utterance with pyctcdecode beam search."""
        arr = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
        width = beam_width or self.config.beam_width
        text = self.decoder.decode(arr, beam_width=width)
        return " ".join(text.split())

    def decode_beams_top_k(
        self,
        logits: LogitArray | torch.Tensor,
        top_k: int = 50,
        beam_width: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k candidate hypotheses with acoustic+LM log-probabilities."""
        arr = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
        width = beam_width or max(self.config.beam_width, top_k)
        beams = self.decoder.decode_beams(arr, beam_width=width)
        return [(" ".join(beam[0].split()), float(beam[3])) for beam in beams[:top_k]]

    def rescore_with_masinissa(
        self,
        candidate_beams: list[tuple[str, float]],
        masinissa_model: torch.nn.Module,
        masinissa_tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        lm_weight: float = 0.3,
    ) -> str:
        """Rescore top-k candidate hypotheses with Masinissa-31M masked language model."""
        if not candidate_beams:
            return ""
        if len(candidate_beams) == 1:
            return candidate_beams[0][0]

        best_hyp = candidate_beams[0][0]
        best_combined_score = -float("inf")

        masinissa_model.eval()
        with torch.no_grad():
            for text, beam_score in candidate_beams:
                if not text.strip():
                    continue
                encoded = masinissa_tokenizer(text, return_tensors="pt").to(device)
                input_ids = encoded["input_ids"]
                logits = masinissa_model(**encoded).logits
                log_probs = torch.log_softmax(logits, dim=-1)

                target_ids = input_ids[0, 1:-1]
                target_log_probs = log_probs[0, 1:-1, :]
                pll = target_log_probs.gather(1, target_ids.unsqueeze(1)).sum().item()

                combined_score = beam_score + lm_weight * pll
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_hyp = text

        return best_hyp
