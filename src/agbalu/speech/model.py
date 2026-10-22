"""Conformer acoustic model with dual-head CTC and speech presence supervision.

Backbone: `facebook/w2v-bert-2.0` (600M parameter Conformer self-supervised model)
Input: 80-channel log-mel filterbanks with delta features (160 dimensions)
Heads:
1. Linear grapheme CTC projection head (emits Kabyle text classes)
2. Linear binary speech-presence auxiliary head (emits frame-level voice activity)

Joint objective:
L_total = L_CTC + 0.15 * L_presence
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as f
from torch import nn
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

PRESENCE_LOSS_WEIGHT: Final[float] = 0.15
DEFAULT_BLANK_ID: Final[int] = 0


@dataclass(frozen=True, slots=True)
class SpeechModelOutput:
    """Output container for ConformerSpeechModel forward pass."""

    loss: torch.Tensor | None
    ctc_loss: torch.Tensor | None
    presence_loss: torch.Tensor | None
    logits: torch.Tensor
    presence_logits: torch.Tensor
    hidden_states: torch.Tensor | None = None


class ConformerSpeechModel(nn.Module):
    """600M Conformer speech model with CTC and auxiliary speech-presence heads."""

    def __init__(self, config: Wav2Vec2BertConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.backbone = Wav2Vec2BertModel(config)
        self.dropout = nn.Dropout(config.feat_proj_dropout)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
        self.presence_head = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        vocab_size: int,
        *,
        feature_dim: int = 160,
    ) -> ConformerSpeechModel:
        """Instantiate model with backbone loaded from checkpoint or hub."""
        config = Wav2Vec2BertConfig.from_pretrained(str(model_name_or_path))
        config.feature_projection_input_dim = feature_dim
        model = cls(config, vocab_size=vocab_size)
        model.backbone = Wav2Vec2BertModel.from_pretrained(str(model_name_or_path), config=config)
        return model

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the Conformer backbone for initial head warmup."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the Conformer backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        presence_labels: torch.Tensor | None = None,
    ) -> SpeechModelOutput:
        """Forward pass emitting CTC logits and frame-level speech-presence logits."""
        backbone_outputs = self.backbone(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state
        dropped = self.dropout(hidden_states)

        logits = self.lm_head(dropped)
        presence_logits = self.presence_head(dropped).squeeze(-1)

        loss: torch.Tensor | None = None
        ctc_loss: torch.Tensor | None = None
        p_loss: torch.Tensor | None = None

        if labels is not None:
            # log_probs for CTC loss must be (time, batch, vocab_size)
            log_probs = f.log_softmax(logits, dim=-1).transpose(0, 1)

            # Determine input lengths from attention mask or full temporal dimension
            if attention_mask is not None:
                # 4x temporal downsampling is handled inside Wav2Vec2BertModel
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

            # Target lengths: exclude padding tokens (id < 0 or == -100)
            target_mask = (labels >= 0) & (labels != -100)
            target_lengths = target_mask.sum(dim=-1).long()
            # Clean labels replacing -100 with blank for gather safety
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

        if presence_labels is not None:
            # Binary cross-entropy with logits over frame speech labels
            p_loss = f.binary_cross_entropy_with_logits(
                presence_logits, presence_labels.float(), reduction="mean"
            )
            if loss is not None:
                loss = loss + (PRESENCE_LOSS_WEIGHT * p_loss)
            else:
                loss = PRESENCE_LOSS_WEIGHT * p_loss

        return SpeechModelOutput(
            loss=loss,
            ctc_loss=ctc_loss,
            presence_loss=p_loss,
            logits=logits,
            presence_logits=presence_logits,
            hidden_states=hidden_states,
        )

    def save_pretrained(self, save_directory: Path) -> None:
        """Save model weights and configuration to disk."""
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(str(save_directory))
        state_dict = self.state_dict()
        torch.save(state_dict, save_directory / "pytorch_model.bin")
        meta = {"vocab_size": self.vocab_size}
        (save_directory / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
