"""Speech acoustic model for Fadhma 2 with pretrained Omnilingual head slicing.

Backbone: `aadel4/omniASR-CTC-300M-v2` (Meta Omnilingual ASR v2, Apache-2.0, 315M parameters).
Trained on 1,668 languages including Kabyle (`kab_Latn`).
Head: 40 CTC grapheme classes sliced directly from Omnilingual's trained 10,288-class head.

Also maintains backward-compatible support for Wav2Vec2BertConfig architectures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as f
from torch import nn
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    Wav2Vec2BertConfig,
    Wav2Vec2BertModel,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
)

DEFAULT_BLANK_ID: Final[int] = 0
DEFAULT_OMNI_BASE: Final[str] = "aadel4/omniASR-CTC-300M-v2"


@dataclass(frozen=True, slots=True)
class SpeechModelOutput:
    """Output container for speech model forward pass."""

    loss: torch.Tensor | None
    ctc_loss: torch.Tensor | None
    presence_loss: torch.Tensor | None
    logits: torch.Tensor
    presence_logits: torch.Tensor
    hidden_states: torch.Tensor | None = None


def slice_omni_head(
    omni_lm_head: nn.Linear,
    omni_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> nn.Linear:
    """Slice target 40 Kabyle classes out of Omnilingual's trained 10,288-class head."""
    vocab_size = len(target_vocab)
    indices = [0] * vocab_size

    for ch, our_id in target_vocab.items():
        if ch == "|":
            omni_id = omni_vocab.get(" ", 4)
        elif ch == "[PAD]":
            omni_id = 0  # Row 0 in omniASR is the CTC blank token
        elif ch == "[UNK]":
            omni_id = omni_vocab.get("<unk>", 3)
        else:
            omni_id = omni_vocab.get(ch, 3)
        indices[our_id] = omni_id

    indices_t = torch.tensor(indices, dtype=torch.long)
    hidden_size = omni_lm_head.weight.size(1)
    new_head = nn.Linear(hidden_size, vocab_size)
    new_head.weight.data = omni_lm_head.weight[indices_t].clone()
    new_head.bias.data = omni_lm_head.bias[indices_t].clone()
    return new_head


class ConformerSpeechModel(nn.Module):
    """Speech acoustic model for Fadhma 2 with dual Wav2Vec2/Conformer support."""

    ctc_model: Wav2Vec2ForCTC | None
    backbone: nn.Module
    lm_head: nn.Linear
    dropout: nn.Dropout
    presence_head: nn.Linear

    def __init__(
        self,
        config: PretrainedConfig,
        vocab_size: int,
        *,
        ctc_model: Wav2Vec2ForCTC | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        if ctc_model is not None:
            self.ctc_model = ctc_model
            self.backbone = ctc_model.wav2vec2
            self.lm_head = ctc_model.lm_head
            self.dropout = ctc_model.dropout
            self.is_wav2vec2 = True
        elif isinstance(config, Wav2Vec2BertConfig):
            self.is_wav2vec2 = False
            self.ctc_model = None
            self.backbone = Wav2Vec2BertModel(config)
            self.dropout = nn.Dropout(config.feat_proj_dropout)
            self.lm_head = nn.Linear(config.hidden_size, vocab_size)
            self.presence_head = nn.Linear(config.hidden_size, 1)
        else:
            # Fallback standard Wav2Vec2
            self.is_wav2vec2 = True
            raw_dict = config.to_dict() if hasattr(config, "to_dict") else {}
            wav_config = Wav2Vec2Config.from_dict(raw_dict) if raw_dict else config
            wav_config.vocab_size = vocab_size
            wav_config.pad_token_id = DEFAULT_BLANK_ID
            self.ctc_model = Wav2Vec2ForCTC(wav_config)
            self.backbone = self.ctc_model.wav2vec2
            self.lm_head = self.ctc_model.lm_head
            self.dropout = self.ctc_model.dropout

        if not hasattr(self, "presence_head"):
            hidden_dim = int(getattr(config, "hidden_size", 1024))
            self.presence_head = nn.Linear(hidden_dim, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        vocab_size: int = 40,
        *,
        vocab_path: Path | None = None,
        feature_dim: int = 160,
    ) -> ConformerSpeechModel:
        """Instantiate model with backbone loaded from checkpoint, hub, or sliced omniASR."""
        path_str = str(model_name_or_path)
        path_obj = Path(model_name_or_path)

        # 1. Local saved checkpoint directory
        if path_obj.is_dir() and (path_obj / "config.json").is_file():
            try:
                ctc = Wav2Vec2ForCTC.from_pretrained(path_str)
                actual_vocab = int(ctc.config.vocab_size or vocab_size)
                return cls(ctc.config, vocab_size=actual_vocab, ctc_model=ctc)
            except Exception:
                config = Wav2Vec2BertConfig.from_pretrained(path_str)
                config.feature_projection_input_dim = feature_dim
                model = cls(config, vocab_size=vocab_size)
                model.backbone = Wav2Vec2BertModel.from_pretrained(path_str, config=config)
                return model

        # 2. Meta Omnilingual ASR v2 base model
        if "omniASR" in path_str or "wav2vec2" in path_str.lower() or path_str == DEFAULT_OMNI_BASE:
            ctc = Wav2Vec2ForCTC.from_pretrained(path_str)
            target_vocab_path = vocab_path or Path("artifacts/asr/vocab.json")

            if target_vocab_path.is_file() and ctc.config.vocab_size != vocab_size:
                tokenizer = AutoTokenizer.from_pretrained(path_str)
                omni_vocab = tokenizer.get_vocab()
                target_vocab: dict[str, int] = json.loads(
                    target_vocab_path.read_text(encoding="utf-8")
                )
                new_head = slice_omni_head(ctc.lm_head, omni_vocab, target_vocab)
                ctc.lm_head = new_head
                ctc.config.vocab_size = len(target_vocab)
                ctc.config.pad_token_id = DEFAULT_BLANK_ID
                ctc.config.ctc_loss_reduction = "mean"
                ctc.config.ctc_zero_infinity = True

            return cls(ctc.config, vocab_size=vocab_size, ctc_model=ctc)

        # 3. Wav2Vec2Bert (w2v-bert-2.0 fallback)
        config = Wav2Vec2BertConfig.from_pretrained(path_str)
        config.feature_projection_input_dim = feature_dim
        model = cls(config, vocab_size=vocab_size)
        model.backbone = Wav2Vec2BertModel.from_pretrained(path_str, config=config)
        return model

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the acoustic backbone for head warmup."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the acoustic backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def _forward_wav2vec2(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> SpeechModelOutput:
        """Forward through Wav2Vec2ForCTC backbone."""
        if self.ctc_model is None:
            msg = "ctc_model is not initialized"
            raise RuntimeError(msg)

        outputs = self.ctc_model(
            input_values=audio,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = outputs.logits
        loss = outputs.loss
        ctc_loss = loss
        batch_sz, time_sz = logits.size(0), logits.size(1)
        presence_logits = torch.zeros(batch_sz, time_sz, device=logits.device)
        p_loss = torch.tensor(0.0, device=logits.device) if loss is not None else None

        return SpeechModelOutput(
            loss=loss,
            ctc_loss=ctc_loss,
            presence_loss=p_loss,
            logits=logits,
            presence_logits=presence_logits,
            hidden_states=outputs.hidden_states[-1] if outputs.hidden_states else None,
        )

    def _forward_bert(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        presence_labels: torch.Tensor | None,
    ) -> SpeechModelOutput:
        """Forward through Wav2Vec2BertModel conformer backbone."""
        if not isinstance(self.backbone, Wav2Vec2BertModel):
            msg = "Expected Wav2Vec2BertModel backbone"
            raise TypeError(msg)

        orig_mask_prob = getattr(self.backbone.config, "mask_time_prob", 0.0)
        if input_features.size(1) < 100:
            self.backbone.config.mask_time_prob = 0.0

        try:
            backbone_outputs = self.backbone(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            self.backbone.config.mask_time_prob = orig_mask_prob

        hidden_states = backbone_outputs.last_hidden_state
        dropped = self.dropout(hidden_states)
        logits = self.lm_head(dropped)
        presence_logits = self.presence_head(dropped).squeeze(-1)

        loss: torch.Tensor | None = None
        ctc_loss: torch.Tensor | None = None
        p_loss: torch.Tensor | None = None

        if labels is not None:
            log_probs = f.log_softmax(logits, dim=-1).transpose(0, 1)
            if attention_mask is not None:
                input_lengths = self.backbone._get_feat_extract_output_lengths(
                    attention_mask.sum(dim=-1).long()  # type: ignore[arg-type]
                )
            else:
                input_lengths = torch.full(
                    size=(logits.size(0),),
                    fill_value=logits.size(1),
                    dtype=torch.long,
                    device=logits.device,
                )

            target_mask = (labels >= 0) & (labels != -100)
            target_lengths = target_mask.sum(dim=-1).long()
            cleaned_targets = torch.where(target_mask, labels, torch.zeros_like(labels))

            ctc_loss = f.ctc_loss(
                log_probs,
                cleaned_targets,
                input_lengths,
                target_lengths,
                blank=DEFAULT_BLANK_ID,
                reduction="mean",
                zero_infinity=True,
            )
            loss = ctc_loss

        if presence_labels is not None and presence_logits.shape == presence_labels.shape:
            p_loss = f.binary_cross_entropy_with_logits(
                presence_logits, presence_labels.float(), reduction="mean"
            )
            if loss is not None:
                loss = loss + (0.15 * p_loss)

        return SpeechModelOutput(
            loss=loss,
            ctc_loss=ctc_loss,
            presence_loss=p_loss,
            logits=logits,
            presence_logits=presence_logits,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        presence_labels: torch.Tensor | None = None,
    ) -> SpeechModelOutput:
        """Forward pass emitting CTC logits and frame-level outputs."""
        if self.is_wav2vec2:
            audio = input_values if input_values is not None else input_features
            if audio is None:
                msg = "Either input_values or input_features must be provided."
                raise ValueError(msg)
            if audio.ndim == 3 and audio.size(-1) == 1:
                audio = audio.squeeze(-1)
            return self._forward_wav2vec2(audio, attention_mask, labels)

        # Conformer / Wav2Vec2Bert path
        if input_features is None:
            if input_values is None:
                msg = "input_features or input_values required for speech model."
                raise ValueError(msg)
            feat_dim = getattr(self.config, "feature_projection_input_dim", 160)
            if input_values.ndim == 2:
                feat_len = max(1, input_values.size(1) // 320)
                downsampled = f.interpolate(
                    input_values.unsqueeze(1), size=feat_len, mode="linear", align_corners=False
                ).squeeze(1)
                input_features = downsampled.unsqueeze(-1).repeat(1, 1, feat_dim)
            else:
                input_features = input_values

        if attention_mask is not None and attention_mask.size(1) != input_features.size(1):
            attention_mask = torch.ones(
                input_features.size(0),
                input_features.size(1),
                device=input_features.device,
                dtype=torch.long,
            )

        return self._forward_bert(input_features, attention_mask, labels, presence_labels)

    def save_pretrained(self, save_directory: Path) -> None:
        """Save model weights and configuration to disk."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.is_wav2vec2 and self.ctc_model is not None:
            self.ctc_model.save_pretrained(str(save_dir))
        else:
            self.config.save_pretrained(str(save_dir))
            torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        meta = {
            "vocab_size": self.vocab_size,
            "model_type": "wav2vec2" if self.is_wav2vec2 else "conformer",
        }
        (save_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
