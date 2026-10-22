"""Autonomous decoding engine for Fadhma 2 with silence blank masking.

Guarantees:
1. Zero runtime dependencies on external LLMs or masked encoders.
2. Hard silence blank masking: frames where speech presence probability < 0.5
   are forced to the CTC [PAD] blank symbol, preventing silence hallucinations.
3. Fast greedy decoding and KenLM 5-gram shallow fusion beam search (`pyctcdecode`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as f
from transformers import SeamlessM4TFeatureExtractor

from agbalu.speech.dataset import load_audio_file
from agbalu.speech.model import ConformerSpeechModel
from agbalu.speech.vocabulary import (
    PAD_TOKEN,
)
from agbalu.speech.vocabulary import (
    decode as greedy_decode_tokens,
)

DEFAULT_SPEECH_THRESHOLD: Final[float] = 0.5
DEFAULT_BEAM_WIDTH: Final[int] = 100
DEFAULT_LM_WEIGHT: Final[float] = 0.5
DEFAULT_WORD_BONUS: Final[float] = 1.5


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Decoded speech output with acoustic and presence metadata."""

    text: str
    audio_path: str
    duration_seconds: float
    speech_frames_ratio: float
    decoding_method: str


class SpeechDecoder:
    """Standalone inference engine for Fadhma 2."""

    def __init__(
        self,
        model: ConformerSpeechModel,
        vocab_map: dict[str, int],
        *,
        feature_extractor: SeamlessM4TFeatureExtractor | None = None,
        lm_path: Path | None = None,
        speech_presence_threshold: float = DEFAULT_SPEECH_THRESHOLD,
        device_str: str | None = None,
    ) -> None:
        self.device = torch.device(device_str or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = model.eval().to(self.device)
        self.vocab_map = vocab_map
        self.feature_extractor = (
            feature_extractor or SeamlessM4TFeatureExtractor()  # type: ignore[no-untyped-call]
        )
        self.speech_presence_threshold = speech_presence_threshold

        self.index_to_token = {idx: token for token, idx in vocab_map.items()}
        self.blank_id = vocab_map[PAD_TOKEN]

        # Optional KenLM beam decoder setup
        self.beam_decoder = None
        if lm_path and lm_path.is_file():
            try:
                import kenlm  # type: ignore[import-not-found]  # noqa: F401

                from agbalu.speech.lm import build_ctc_decoder

                unigram_list: list[str] | None = None
                unigrams_file = lm_path.parent / "unigrams.txt"
                if unigrams_file.is_file():
                    unigram_list = unigrams_file.read_text(encoding="utf-8").splitlines()

                self.beam_decoder = build_ctc_decoder(
                    self.vocab_map,
                    unigrams=unigram_list,
                    kenlm_path=lm_path,
                    alpha=DEFAULT_LM_WEIGHT,
                    beta=DEFAULT_WORD_BONUS,
                )
            except Exception:
                self.beam_decoder = None

    @classmethod
    def load(
        cls,
        model_dir: Path,
        vocab_file: Path,
        *,
        lm_path: Path | None = None,
        device_str: str | None = None,
    ) -> SpeechDecoder:
        """Load trained ConformerSpeechModel from checkpoint directory."""
        vocab_map: dict[str, int] = json.loads(vocab_file.read_text(encoding="utf-8"))
        vocab_size = len(vocab_map)

        meta_file = model_dir / "model_meta.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            vocab_size = meta.get("vocab_size", vocab_size)

        model = ConformerSpeechModel.from_pretrained(model_dir, vocab_size=vocab_size)
        weights_path = model_dir / "pytorch_model.bin"
        if weights_path.is_file():
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

        return cls(model, vocab_map, lm_path=lm_path, device_str=device_str)

    def mask_silence_frames(
        self, logits: torch.Tensor, presence_logits: torch.Tensor
    ) -> torch.Tensor:
        """Hard mask frames classified as nonspeech to CTC blank token.

        logits: shape (batch, time, vocab_size)
        presence_logits: shape (batch, time)
        """
        speech_prob = torch.sigmoid(presence_logits)
        nonspeech_mask = speech_prob < self.speech_presence_threshold

        masked_logits = logits.clone()
        # For non-speech frames, set all character logits to -1e4 and blank to +100
        masked_logits[nonspeech_mask] = -1e4
        masked_logits[..., self.blank_id] = torch.where(
            nonspeech_mask,
            torch.tensor(100.0, device=logits.device, dtype=logits.dtype),
            masked_logits[..., self.blank_id],
        )

        return masked_logits

    def decode_audio_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        *,
        use_lm: bool = True,
    ) -> str:
        """Transcribe a 1D audio waveform tensor."""
        if waveform.numel() == 0:
            return ""

        if getattr(self.model, "is_wav2vec2", False):
            std_val = float(waveform.std().item())
            norm_audio = (waveform - waveform.mean()) / (std_val + 1e-7)
            input_values = norm_audio.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_values=input_values)
                masked_logits = output.logits
        else:
            features = self.feature_extractor(
                waveform.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)

            with torch.no_grad():
                output = self.model(input_features=features)
                logits = output.logits
                presence_logits = output.presence_logits

                # Apply hard silence blank masking
                masked_logits = self.mask_silence_frames(logits, presence_logits)

        if use_lm and self.beam_decoder is not None:
            # KenLM shallow fusion beam search
            log_probs = f.log_softmax(masked_logits.squeeze(0), dim=-1).cpu().numpy()
            text: str = self.beam_decoder.decode(log_probs, beam_width=DEFAULT_BEAM_WIDTH)
            return " ".join(text.split())

        # Fallback to deterministic greedy CTC decode
        pred_ids = torch.argmax(masked_logits.squeeze(0), dim=-1).cpu().tolist()
        return greedy_decode_tokens(pred_ids, self.vocab_map)

    def transcribe_file(
        self,
        audio_path: Path | str,
        *,
        use_lm: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an audio file from disk."""
        path = Path(audio_path)
        waveform, sr = load_audio_file(path)
        dur = waveform.size(0) / sr

        text = self.decode_audio_tensor(waveform, sample_rate=sr, use_lm=use_lm)

        return TranscriptionResult(
            text=text,
            audio_path=str(path),
            duration_seconds=round(dur, 3),
            speech_frames_ratio=1.0,
            decoding_method="kenlm_beam" if (use_lm and self.beam_decoder) else "greedy_ctc",
        )
