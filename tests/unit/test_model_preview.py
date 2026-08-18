from __future__ import annotations

from typing import Final

import torch

from agbalu.model.config import ModelConfig
from agbalu.model.modeling import IGNORE_INDEX, Encoder
from agbalu.model.preview import (
    CLOSE,
    MAX_PIECES,
    OPEN,
    MaskedPreview,
    Previewer,
    _mark,
    render,
)

VOCAB: Final = 64
CLS: Final = 2
SEP: Final = 3
PAD: Final = 0
STOP: Final = frozenset({SEP, PAD})
SKIP: Final = frozenset({CLS})

TINY: Final = ModelConfig(
    vocab_size=VOCAB,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_hidden_layers=2,
)


class Pieces:
    """Ids 10+ are word-initial; the rest are continuations, as SentencePiece renders."""

    def id_to_piece(self, piece_id: int) -> str:
        if piece_id == CLS:
            return "[CLS]"
        if piece_id == SEP:
            return "[SEP]"
        if piece_id == PAD:
            return "[PAD]"
        return f"▁w{piece_id}" if piece_id >= 10 else f"c{piece_id}"


def previewer(
    ids: list[list[int]], labels: list[list[int]], max_pieces: int = MAX_PIECES
) -> tuple[Previewer, Encoder]:
    id_tensor = torch.tensor(ids)
    batch = (id_tensor, torch.ones_like(id_tensor, dtype=torch.bool), torch.tensor(labels))
    built = Previewer(batch, Pieces(), stop_ids=STOP, skip_ids=SKIP, max_pieces=max_pieces)
    return built, Encoder(TINY)


def run(
    ids: list[list[int]], labels: list[list[int]], max_pieces: int = MAX_PIECES
) -> list[MaskedPreview]:
    preview, model = previewer(ids, labels, max_pieces)
    return preview.preview(model, torch.device("cpu"))


class TestRendering:
    def test_metaspace_becomes_a_space(self) -> None:
        assert render(["▁w1", "c2", "▁w3"]) == "w1c2 w3"

    def test_leading_metaspace_is_stripped(self) -> None:
        assert render(["▁w1"]) == "w1"

    def test_an_empty_piece_list_renders_empty(self) -> None:
        assert render([]) == ""


class TestPreview:
    def test_a_masked_position_is_marked_and_an_unmasked_one_is_not(self) -> None:
        previews = run([[CLS, 11, 12, 13, SEP]], [[IGNORE_INDEX] * 2 + [12] + [IGNORE_INDEX] * 2])
        assert len(previews) == 1
        assert previews[0].gold == "w11 w12 w13"
        assert OPEN in previews[0].predicted
        assert previews[0].predicted.count(OPEN) == 1
        assert previews[0].predicted.count(CLOSE) == 1

    def test_the_marker_keeps_the_word_boundary_outside_it(self) -> None:
        assert _mark("▁azul") == f"▁{OPEN}azul{CLOSE}"
        assert render([_mark("▁azul")]) == f"{OPEN}azul{CLOSE}"
        assert render(["▁w1", _mark("▁azul")]) == f"w1 {OPEN}azul{CLOSE}"

    def test_a_continuation_piece_is_marked_without_gaining_a_boundary(self) -> None:
        assert _mark("kra") == f"{OPEN}kra{CLOSE}"
        assert render(["▁w1", _mark("kra")]) == f"w1{OPEN}kra{CLOSE}"

    def test_it_stops_at_the_first_separator(self) -> None:
        previews = run(
            [[CLS, 11, 12, SEP, 13, 14, SEP]],
            [[IGNORE_INDEX, IGNORE_INDEX, 12] + [IGNORE_INDEX] * 4],
        )
        assert previews[0].gold == "w11 w12"

    def test_the_class_token_is_skipped(self) -> None:
        previews = run([[CLS, 11, SEP]], [[IGNORE_INDEX, 11, IGNORE_INDEX]])
        assert "[CLS]" not in previews[0].gold
        assert "[CLS]" not in previews[0].predicted

    def test_a_row_with_nothing_masked_is_dropped(self) -> None:
        assert run([[CLS, 11, 12, SEP]], [[IGNORE_INDEX] * 4]) == []

    def test_rows_are_previewed_independently(self) -> None:
        previews = run(
            [[CLS, 11, 12, SEP], [CLS, 13, 14, SEP]],
            [[IGNORE_INDEX, IGNORE_INDEX, 12, IGNORE_INDEX], [IGNORE_INDEX] * 4],
        )
        assert len(previews) == 1, "the second row masks nothing"

    def test_max_pieces_truncates_a_long_window(self) -> None:
        ids = [[CLS, *range(11, 51)]]
        labels = [[IGNORE_INDEX, 11, *[IGNORE_INDEX] * 39]]
        previews = run(ids, labels, max_pieces=5)
        assert len(previews[0].gold.split()) == 5

    def test_the_total_counts_only_what_is_shown(self) -> None:
        ids = [[CLS, 11, 12, SEP, 13, 14]]
        labels = [[IGNORE_INDEX, IGNORE_INDEX, 12, IGNORE_INDEX, 13, 14]]
        assert run(ids, labels)[0].total == 1, "the masks after [SEP] are outside the window"

    def test_correct_never_exceeds_total(self) -> None:
        previews = run([[CLS, 11, 12, 13, SEP]], [[IGNORE_INDEX, 11, 12, 13, IGNORE_INDEX]])
        assert 0 <= previews[0].correct <= previews[0].total == 3

    def test_it_is_deterministic_for_a_fixed_model(self) -> None:
        preview, model = previewer([[CLS, 11, 12, SEP]], [[IGNORE_INDEX, IGNORE_INDEX, 12, 0]])
        first = preview.preview(model, torch.device("cpu"))
        second = preview.preview(model, torch.device("cpu"))
        assert first == second

    def test_it_leaves_the_model_in_training_mode(self) -> None:
        preview, model = previewer([[CLS, 11, 12, SEP]], [[IGNORE_INDEX, IGNORE_INDEX, 12, 0]])
        model.train()
        preview.preview(model, torch.device("cpu"))
        assert model.training

    def test_it_leaves_an_evaluating_model_evaluating(self) -> None:
        preview, model = previewer([[CLS, 11, 12, SEP]], [[IGNORE_INDEX, IGNORE_INDEX, 12, 0]])
        model.eval()
        preview.preview(model, torch.device("cpu"))
        assert not model.training


class TestAccuracy:
    def test_it_is_the_ratio(self) -> None:
        assert MaskedPreview("a", "b", correct=3, total=4).accuracy == 0.75

    def test_an_empty_preview_does_not_divide_by_zero(self) -> None:
        assert MaskedPreview("", "", correct=0, total=0).accuracy == 0.0


class TestPredictMasked:
    def test_it_predicts_only_at_masked_positions(self) -> None:
        model = Encoder(TINY).eval()
        ids = torch.tensor([[CLS, 11, 12, SEP]])
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 12, IGNORE_INDEX]])
        predictions = model.predict_masked(ids, torch.ones_like(ids, dtype=torch.bool), labels)
        assert predictions.shape == labels.shape
        assert (predictions[labels == IGNORE_INDEX] == IGNORE_INDEX).all()
        assert 0 <= int(predictions[0, 2]) < VOCAB

    def test_nothing_masked_yields_no_predictions(self) -> None:
        model = Encoder(TINY).eval()
        ids = torch.tensor([[CLS, 11, 12, SEP]])
        labels = torch.full_like(ids, IGNORE_INDEX)
        predictions = model.predict_masked(ids, torch.ones_like(ids, dtype=torch.bool), labels)
        assert (predictions == IGNORE_INDEX).all()
