"""Decode what the model predicts at masked positions, as Kabyle text.

A falling loss says the model is improving; it does not say what it learned. The same
held-out sentences are masked with a fixed seed at every evaluation, so successive
previews are comparable and the filled-in words can be read.

Training windows pack several sentences end to end, so a whole window is unreadable. A
preview shows one sentence: it starts after `[CLS]` and stops at the first `[SEP]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import torch
from torch import Tensor

from agbalu.model.modeling import IGNORE_INDEX, Encoder

METASPACE: Final = "▁"
OPEN: Final = "‹"
CLOSE: Final = "›"
"""Wrap a predicted piece, so masked positions are visible without drowning the line."""

MAX_PIECES: Final = 40


@dataclass(frozen=True, slots=True)
class MaskedPreview:
    gold: str
    predicted: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


class Decoder(Protocol):
    def id_to_piece(self, piece_id: int) -> str: ...


def render(pieces: list[str]) -> str:
    return "".join(pieces).replace(METASPACE, " ").strip()


def _mark(piece: str) -> str:
    """Keep the metaspace outside the marker so word boundaries survive rendering."""
    leading = METASPACE if piece.startswith(METASPACE) else ""
    return f"{leading}{OPEN}{piece.removeprefix(METASPACE)}{CLOSE}"


class Previewer:
    """Holds a fixed batch so every evaluation previews the same sentences."""

    def __init__(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        decoder: Decoder,
        *,
        stop_ids: frozenset[int],
        skip_ids: frozenset[int] = frozenset(),
        max_pieces: int = MAX_PIECES,
    ) -> None:
        self.ids, self.attention, self.labels = batch
        self.decoder = decoder
        self.stop_ids = stop_ids
        self.skip_ids = skip_ids
        self.max_pieces = max_pieces

    def _row(self, row: int, predictions: Tensor) -> MaskedPreview | None:
        gold_pieces: list[str] = []
        predicted_pieces: list[str] = []
        correct = 0
        total = 0
        for position in range(int(self.ids.shape[1])):
            if len(gold_pieces) >= self.max_pieces:
                break
            label = int(self.labels[row, position])
            token = int(self.ids[row, position])
            unmasked = label == IGNORE_INDEX
            if (token if unmasked else label) in self.stop_ids and gold_pieces:
                break
            if unmasked:
                if token in self.skip_ids or token in self.stop_ids:
                    continue
                piece = self.decoder.id_to_piece(token)
                gold_pieces.append(piece)
                predicted_pieces.append(piece)
                continue
            total += 1
            prediction = int(predictions[row, position])
            correct += prediction == label
            gold_pieces.append(self.decoder.id_to_piece(label))
            predicted_pieces.append(_mark(self.decoder.id_to_piece(prediction)))
        if not total:
            return None
        return MaskedPreview(render(gold_pieces), render(predicted_pieces), correct, total)

    @torch.no_grad()
    def preview(self, model: Encoder, device: torch.device) -> list[MaskedPreview]:
        was_training = model.training
        model.eval()
        predictions = model.predict_masked(
            self.ids.to(device), self.attention.to(device), self.labels.to(device)
        ).cpu()
        model.train(was_training)
        rows = (self._row(row, predictions) for row in range(int(self.ids.shape[0])))
        return [row for row in rows if row is not None]
