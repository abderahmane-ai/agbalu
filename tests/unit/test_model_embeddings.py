"""Geometry of the word-embedding table: what the analogy actually returns."""

from __future__ import annotations

import pytest
import torch

from agbalu.model.embeddings import ANALOGY_TERMS, analogy, neighbours, piece_for

PIECES = ("king", "man", "woman", "queen", "stone", "river")


class FakeTokenizer:
    """One row per word, so an id is its index in `PIECES`."""

    def __init__(self, unknown: int = 0) -> None:
        self._unknown = unknown

    def encode(self, text: str, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [PIECES.index(text)] if text in PIECES else []

    def unk_id(self) -> int:
        return self._unknown

    def id_to_piece(self, piece_id: int) -> str:
        return PIECES[piece_id]


def table() -> torch.Tensor:
    """`queen` is exactly `king - man + woman`, so the analogy has one right answer."""
    king = torch.tensor([1.0, 1.0, 0.0])
    man = torch.tensor([1.0, 0.0, 0.0])
    woman = torch.tensor([0.0, 0.0, 1.0])
    rows = {
        "king": king,
        "man": man,
        "woman": woman,
        "queen": king - man + woman,
        "stone": torch.tensor([-1.0, -1.0, -1.0]),
        "river": torch.tensor([0.5, -0.5, 0.0]),
    }
    return torch.stack([rows[piece] for piece in PIECES])


class TestPieceFor:
    def test_a_word_in_the_vocabulary_is_its_own_row(self) -> None:
        assert piece_for("queen", FakeTokenizer()) == PIECES.index("queen")

    def test_a_word_that_produces_no_piece_falls_back_to_unk(self) -> None:
        assert piece_for("absent", FakeTokenizer(unknown=4)) == 4


class TestNeighbours:
    def test_the_closest_row_to_a_vector_is_itself(self) -> None:
        rows = table()
        found = neighbours(rows[PIECES.index("king")], rows, FakeTokenizer(), top_k=1)
        assert found[0].piece == "king"
        assert found[0].similarity == pytest.approx(1.0, abs=1e-6)

    def test_excluded_rows_do_not_appear(self) -> None:
        rows = table()
        found = neighbours(rows[0], rows, FakeTokenizer(), top_k=3, exclude=[0, 1])
        assert {n.piece for n in found}.isdisjoint({"king", "man"})

    def test_the_list_is_still_top_k_when_excluded_rows_ranked_highest(self) -> None:
        """The analogy case: all three operands sit near their own combination, so a
        naive top_k would come back short."""
        rows = table()
        found = neighbours(rows[0], rows, FakeTokenizer(), top_k=3, exclude=[0, 1])
        assert len(found) == 3

    def test_asking_for_more_than_the_vocabulary_holds_does_not_raise(self) -> None:
        rows = table()
        assert len(neighbours(rows[0], rows, FakeTokenizer(), top_k=99)) == len(PIECES)

    def test_similarity_is_cosine_so_magnitude_does_not_matter(self) -> None:
        rows = table()
        scaled = neighbours(rows[PIECES.index("king")] * 7.0, rows, FakeTokenizer(), top_k=1)
        assert scaled[0].piece == "king"


class TestAnalogy:
    def test_king_minus_man_plus_woman_is_queen(self) -> None:
        found, ids = analogy(("king", "man", "woman"), table(), FakeTokenizer())
        assert found[0].piece == "queen"
        assert ids == [PIECES.index(w) for w in ("king", "man", "woman")]

    def test_the_operands_are_never_returned_as_the_answer(self) -> None:
        found, _ = analogy(("king", "man", "woman"), table(), FakeTokenizer())
        assert {n.piece for n in found}.isdisjoint({"king", "man", "woman"})

    def test_the_wrong_number_of_terms_is_refused(self) -> None:
        with pytest.raises(ValueError, match=f"exactly {ANALOGY_TERMS} words"):
            analogy(("king", "man"), table(), FakeTokenizer())

    def test_an_out_of_vocabulary_operand_resolves_through_unk(self) -> None:
        found, ids = analogy(("king", "absent", "woman"), table(), FakeTokenizer(unknown=4))
        assert ids[1] == 4
        assert found
