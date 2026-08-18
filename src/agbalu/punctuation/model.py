"""Two token-classification heads over the Kabyle encoder.

The heads are separate because the tasks are: casing is lexical — whether a word is a name —
and punctuation is syntactic. Sharing a single softmax over their product would make the
model spend capacity on combinations that never occur, and would make the two error rates
impossible to read apart.

The encoder's own masked-token classifier is loaded, because the checkpoint contains it, and
frozen: restoring punctuation never predicts a word.

**The tied weight is exempt from that freeze, and it has to be.** The classifier's output
weight *is* the input embedding matrix — one tensor under two names — so freezing the head by
`parameters()` silently froze 6,144,000 embedding parameters too, and the first run trained
25,116,296 of 31,424,136 without anyone choosing that. Standard fine-tuning adapts the
embeddings, and proper nouns are the task where word identity carries the answer, so the
freeze is applied by identity and skips the shared tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn

from agbalu.model.checkpoint import load as load_checkpoint
from agbalu.model.config import PRESETS, Preset
from agbalu.model.modeling import Encoder
from agbalu.punctuation.labels import CASE, PUNCTUATION

DEFAULT_CHECKPOINT: Final = Path("artifacts/runs/agbalu-encoder-v1")
BEST: Final = "best.pt"


@dataclass(frozen=True, slots=True)
class Heads:
    punctuation: Tensor
    case: Tensor


class _Head(nn.Module):
    def __init__(self, hidden_size: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, classes)
        nn.init.trunc_normal_(self.dense.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.trunc_normal_(self.out.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        projected: Tensor = self.out(self.dropout(self.activation(self.dense(self.norm(hidden)))))
        return projected


class Restorer(nn.Module):
    """The encoder plus a punctuation head and a casing head, both over every position."""

    def __init__(self, encoder: Encoder, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.punctuation = _Head(hidden_size, len(PUNCTUATION), dropout)
        self.case = _Head(hidden_size, len(CASE), dropout)
        tied = self.encoder.embedding.word_embedding.weight
        for parameter in self.encoder.classifier.parameters():
            if parameter is not tied:
                parameter.requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Heads:
        hidden = self.encoder.contextualise(input_ids, attention_mask)
        return Heads(self.punctuation(hidden), self.case(hidden))

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    *,
    preset: Preset = "kab",
    dropout: float = 0.1,
    name: str = BEST,
    device: torch.device | None = None,
) -> Restorer:
    """Restore the pretrained encoder through the checksum-verifying loader, then attach heads.

    The RNG is deliberately not restored: replaying the pretraining stream here would make
    this run's shuffling and dropout a function of where that one stopped.
    """
    target = device or torch.device("cpu")
    encoder = Encoder(PRESETS[preset])
    load_checkpoint(checkpoint, encoder, None, name=name, restore_rng=False)
    return Restorer(encoder, dropout).to(target)
