"""The classification head Masinissa is fine-tuned with, and the model it is trained as.

The head is duplicated in `agbalu.hub.masinissa_sentiment`, which the published repository
loads and which cannot import this package. `tests/unit/test_hub_masinissa_sentiment.py`
holds the two together by running both over the same weights and comparing the logits; a
change made on one side without the other publishes a model whose head is not the one that
was trained.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.model.modeling import Encoder
from agbalu.sentiment.config import SentimentConfig, SentimentConfigError


def masked_mean(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean over the real positions.

    Not `[CLS]`: this encoder is masked-language-model pretrained with no sentence-level
    objective, so nothing ever trained that position to summarise the sequence.
    """
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class MultiSampleHead(nn.Module):
    """A normalised bottleneck read through several dropout masks at once.

    The encoder runs before the branch, so averaging the loss over the masks costs one
    extra Linear per mask and buys the gradient of an ensemble of that many sub-networks.
    Inference takes the undropped path, which is what the published head evaluates.
    """

    def __init__(self, config: SentimentConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_in = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.bottleneck_size)
        self.activation = nn.GELU(approximate="tanh")
        self.norm_mid = nn.LayerNorm(config.bottleneck_size, eps=config.layer_norm_eps)
        self.dropouts = nn.ModuleList(nn.Dropout(rate) for rate in config.dropouts)
        self.classifier = nn.Linear(config.bottleneck_size, config.num_classes)

    def features(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        pooled = masked_mean(hidden, attention_mask)
        bottleneck: Tensor = self.norm_mid(self.activation(self.dense(self.norm_in(pooled))))
        return bottleneck

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        feature = self.features(hidden, attention_mask)
        if not self.training:
            logits = self.classifier(feature)
            loss = None if labels is None else functional.cross_entropy(logits, labels)
            return logits, loss

        sampled = [self.classifier(dropout(feature)) for dropout in self.dropouts]
        logits = torch.stack(sampled, dim=0).mean(dim=0)
        loss = None
        if labels is not None:
            loss = torch.stack(
                [
                    functional.cross_entropy(
                        sample, labels, label_smoothing=self.config.label_smoothing
                    )
                    for sample in sampled
                ]
            ).mean()
        return logits, loss


class SequenceClassifier(nn.Module):
    """The pretrained encoder with the head on top, which is what the fine-tune optimises."""

    def __init__(self, encoder: Encoder, config: SentimentConfig | None = None) -> None:
        super().__init__()
        head_config = config or SentimentConfig()
        if encoder.config.hidden_size != head_config.hidden_size:
            message = (
                f"the encoder is {encoder.config.hidden_size}-dimensional and the head is "
                f"sized for {head_config.hidden_size}"
            )
            raise SentimentConfigError(message)
        self.encoder = encoder
        self.config = head_config
        self.head = MultiSampleHead(head_config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden = self.encoder.contextualise(input_ids, attention_mask)
        scored: tuple[Tensor, Tensor | None] = self.head(hidden, attention_mask, labels=labels)
        return scored
