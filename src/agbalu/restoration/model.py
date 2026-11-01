"""Predict ordered word-boundary edits over the native pretrained encoder."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from agbalu.encoder.attention import Layout, make_layout
from agbalu.encoder.checkpoint import verify_snapshot
from agbalu.encoder.configuration import Geometry
from agbalu.encoder.corpus import read_json
from agbalu.encoder.model import Encoder
from agbalu.restoration.annotations import CASES, MARKS
from agbalu.restoration.configuration import Settings
from agbalu.restoration.dataset import Batch


@dataclass(frozen=True)
class Scores:
    presence: Tensor
    punctuation: Tensor
    casing: Tensor
    structure: Tensor


def projection(in_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_features), nn.LayerNorm(out_features), nn.GELU(), nn.Dropout(0.1)
    )


class Restorer(nn.Module):
    def __init__(self, geometry: Geometry, templates: tuple[str, ...]) -> None:
        super().__init__()
        self.geometry = geometry
        self.templates = templates
        self.encoder = Encoder(geometry)
        del self.encoder.head
        del self.encoder.head_norm
        del self.encoder.vocabulary_bias
        width = geometry.hidden_size
        self.word_projection = projection(3 * width, width)
        self.boundary_projection = projection(4 * width, width)
        self.document_edges = nn.Embedding(2, width)
        self.presence = nn.Linear(width, 1)
        self.punctuation = nn.Linear(width, len(MARKS) - 1)
        self.casing = nn.Sequential(
            nn.Linear(2 * width + 3, width // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width // 2, len(CASES)),
        )
        self.structure = nn.Linear(width, len(templates))
        self.features: Callable[[Tensor, Layout], Tensor] = self.encoder.contextualise_packed

    @classmethod
    def from_settings(cls, settings: Settings, templates: tuple[str, ...]) -> "Restorer":
        return cls(settings.geometry, templates)

    def load_encoder(self, directory: Path) -> str:
        pointer = read_json(directory / "best_mlm.json")
        snapshot = directory / str(pointer["snapshot"])
        verify_snapshot(snapshot)
        source_geometry = Geometry.model_validate(read_json(snapshot / "geometry.json"))
        left = source_geometry.model_dump(exclude={"backend", "dropout"})
        right = self.geometry.model_dump(exclude={"backend", "dropout"})
        if left != right:
            msg = f"encoder geometry mismatch: {left} != {right}"
            raise ValueError(msg)
        payload = torch.load(snapshot / "training.pt", map_location="cpu", weights_only=True)
        state = {
            name: value
            for name, value in payload["weights"].items()
            if not name.startswith(("head.", "head_norm.", "vocabulary_bias"))
        }
        self.encoder.load_state_dict(state, strict=True)
        return str(snapshot)

    def contextualise(self, batch: Batch, hidden: Tensor) -> Scores:
        batch_size = batch.words.size(0)
        width = hidden.size(-1)
        starts, ends = batch.spans.unbind(-1)
        first = hidden[starts]
        last = hidden[(ends - 1).clamp(min=0)]
        prefix = functional.pad(hidden.float().cumsum(0), (0, 0, 1, 0))
        mean = (prefix[ends] - prefix[starts]) / (ends - starts).clamp(min=1).unsqueeze(-1)
        words: Tensor = self.word_projection(torch.cat((first, last, mean.to(hidden.dtype)), -1))
        words = words * batch.words.unsqueeze(-1)
        start_edge = self.document_edges.weight[0].expand(batch_size, 1, -1)
        end_edge = self.document_edges.weight[1].expand(batch_size, 1, -1)
        left = torch.cat((start_edge * batch.edges[:, :1, None], words), 1)
        right = torch.cat((words, end_edge), 1)
        indices = batch.words.sum(1).long()
        right = right.scatter(
            1, indices[:, None, None].expand(-1, 1, width), end_edge * batch.edges[:, 1:, None]
        )
        boundary: Tensor = self.boundary_projection(
            torch.cat((left, right, left * right, right - left), -1)
        )
        presence: Tensor = self.presence(boundary).squeeze(-1)
        punctuation: Tensor = self.punctuation(boundary)
        component = hidden[(batch.case_spans[..., 1] - 1).clamp(min=0)]
        context = words.gather(1, batch.case_words[..., None].expand(-1, -1, width))
        casing: Tensor = self.casing(
            torch.cat((component, context, batch.case_features.to(hidden.dtype)), -1)
        )
        structure: Tensor = self.structure(boundary)
        return Scores(presence, punctuation, casing, structure)

    def forward(self, batch: Batch) -> Scores:
        hidden: list[Tensor] = []
        for index in range(batch.ids.size(0)):
            layout = make_layout(
                batch.documents[index],
                batch.positions[index],
                batch.attention[index],
                flex=self.geometry.backend == "flex",
            )
            hidden.append(self.features(batch.ids[index], layout))
        return self.contextualise(batch, torch.cat(hidden))

    def enable_compile(self) -> None:
        self.features = torch.compile(self.encoder.contextualise_packed, dynamic=False)


def joint_probabilities(scores: Scores) -> Tensor:
    presence = scores.presence.float().sigmoid().unsqueeze(-1)
    return torch.cat((1 - presence, presence * scores.punctuation.float().softmax(-1)), -1)
