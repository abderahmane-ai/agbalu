"""Standalone Hugging Face Modeling class for Fadhma-Pro ASR."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel, Wav2Vec2Model

from .configuration_fadhma_pro import FadhmaProConfig


class FadhmaProForCTC(PreTrainedModel):
    """Fadhma-Pro ASR model for standalone Hugging Face Hub inference."""

    config_class = FadhmaProConfig
    base_model_prefix = "wav2vec2"

    def __init__(self, config: FadhmaProConfig) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full(
                size=(logits.shape[0],),
                fill_value=logits.shape[1],
                dtype=torch.long,
                device=logits.device,
            )
            target_lengths = torch.sum(labels >= 0, dim=-1)
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )

        return {"loss": loss, "logits": logits}
