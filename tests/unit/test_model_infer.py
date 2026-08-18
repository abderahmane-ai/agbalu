"""Fill-mask inference.

The mask id is a property of the vocabulary, not the constant 4: `[MASK]` is a
`user_defined_symbol`, so a tokenizer rebuild can move it. Reading it from the tokenizer is
what these tests pin, along with the failure that would otherwise be silent — a text with
no mask, or a tokenizer with no `[MASK]` piece, must raise rather than score nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from agbalu.model.checkpoint import BEST, LATEST, TrainingState, save
from agbalu.model.config import PRESETS, ModelError
from agbalu.model.infer import Candidate, MaskPrediction, encode, fill_mask, load_encoder, mask_id
from agbalu.model.modeling import Encoder
from agbalu.tokenizer.spec import CLS_ID, MASK_PIECE, SEP_ID

VOCAB = 64
MASK = 4

TINY = replace(
    PRESETS["kab"],
    vocab_size=VOCAB,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_hidden_layers=2,
)


class FakeTokenizer:
    """Encodes whitespace words to fixed ids, with `[MASK]` at `MASK`."""

    def __init__(self, mask_piece: str | None = MASK_PIECE) -> None:
        self.mask_piece = mask_piece

    def piece_to_id(self, piece: str) -> int:
        if piece == self.mask_piece:
            return MASK
        return 1

    def unk_id(self) -> int:
        return 1

    def encode(self, text: str, out_type: type[int] = int) -> list[int]:  # noqa: ARG002
        """`out_type` is part of the SentencePiece signature; this fake only returns ids."""
        return [MASK if word == MASK_PIECE else 10 + (len(word) % 20) for word in text.split()]

    def id_to_piece(self, token_id: int) -> str:
        return f"p{token_id}"


class TestMaskId:
    def test_it_is_read_from_the_tokenizer(self) -> None:
        assert mask_id(FakeTokenizer()) == MASK

    def test_a_tokenizer_without_the_piece_raises(self) -> None:
        """Otherwise `piece_to_id` returns unk and every position looks unmasked."""
        with pytest.raises(ModelError, match=MASK_PIECE.replace("[", r"\[")):
            mask_id(FakeTokenizer(mask_piece=None))


class TestEncode:
    def test_special_tokens_wrap_the_sequence(self) -> None:
        ids, _ = encode(f"azul {MASK_PIECE} ay amdakel", FakeTokenizer())
        assert ids[0] == CLS_ID
        assert ids[-1] == SEP_ID

    def test_positions_account_for_the_leading_cls(self) -> None:
        """An off-by-one here reads the neighbour's hidden state and still returns fluent
        candidates, so it would not look like a bug."""
        ids, positions = encode(f"{MASK_PIECE} ay amdakel", FakeTokenizer())
        assert positions == [1]
        assert ids[positions[0]] == MASK

    def test_every_mask_is_found(self) -> None:
        _, positions = encode(f"a {MASK_PIECE} b {MASK_PIECE} c", FakeTokenizer())
        assert len(positions) == 2

    def test_text_without_a_mask_raises(self) -> None:
        with pytest.raises(ModelError, match=r"no \[MASK\]"):
            encode("azul ay amdakel", FakeTokenizer())

    def test_the_error_shows_the_text(self) -> None:
        with pytest.raises(ModelError, match="azul"):
            encode("azul", FakeTokenizer())


def tiny_model() -> Encoder:
    torch.manual_seed(0)
    return Encoder(TINY).eval()


class TestFillMask:
    def test_one_prediction_per_mask(self) -> None:
        out = fill_mask(f"a {MASK_PIECE} b {MASK_PIECE}", tiny_model(), FakeTokenizer(), top_k=3)
        assert len(out) == 2
        assert all(isinstance(p, MaskPrediction) for p in out)

    def test_top_k_candidates_are_returned_and_ranked(self) -> None:
        out = fill_mask(f"a {MASK_PIECE} b", tiny_model(), FakeTokenizer(), top_k=5)
        scores = [c.probability for c in out[0].candidates]
        assert len(scores) == 5
        assert scores == sorted(scores, reverse=True)

    def test_probabilities_are_a_distribution_slice(self) -> None:
        out = fill_mask(f"a {MASK_PIECE} b", tiny_model(), FakeTokenizer(), top_k=VOCAB)
        assert sum(c.probability for c in out[0].candidates) == pytest.approx(1.0, abs=1e-4)

    def test_candidates_carry_pieces_not_ids(self) -> None:
        out = fill_mask(f"a {MASK_PIECE} b", tiny_model(), FakeTokenizer(), top_k=2)
        assert all(isinstance(c, Candidate) and c.piece.startswith("p") for c in out[0].candidates)

    def test_top_k_above_the_vocabulary_is_clamped(self) -> None:
        out = fill_mask(f"a {MASK_PIECE} b", tiny_model(), FakeTokenizer(), top_k=VOCAB * 10)
        assert len(out[0].candidates) == VOCAB

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_a_non_positive_top_k_raises(self, top_k: int) -> None:
        with pytest.raises(ModelError, match="must be positive"):
            fill_mask(f"a {MASK_PIECE}", tiny_model(), FakeTokenizer(), top_k=top_k)

    def test_the_index_points_at_the_masked_slot(self) -> None:
        out = fill_mask(f"a b {MASK_PIECE}", tiny_model(), FakeTokenizer(), top_k=1)
        ids, positions = encode(f"a b {MASK_PIECE}", FakeTokenizer())
        assert [p.index for p in out] == positions
        assert ids[out[0].index] == MASK


class TestLoadEncoder:
    def test_a_saved_checkpoint_is_restored(self, tmp_path: Path) -> None:
        model = Encoder(PRESETS["kab"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        save(tmp_path, model, optimizer, TrainingState(step=7), name=BEST)

        restored = load_encoder(tmp_path, "kab", name=BEST)
        for before, after in zip(model.parameters(), restored.parameters(), strict=True):
            assert torch.allclose(before, after)

    def test_the_model_comes_back_in_eval_mode(self, tmp_path: Path) -> None:
        """Dropout at inference makes the same prompt give different answers."""
        model = Encoder(PRESETS["kab"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        save(tmp_path, model, optimizer, TrainingState(), name=BEST)
        assert not load_encoder(tmp_path, "kab", name=BEST).training

    def test_a_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError, match="no checkpoint"):
            load_encoder(tmp_path, "kab", name=LATEST)

    def test_a_corrupt_checkpoint_is_refused(self, tmp_path: Path) -> None:
        """Inference goes through the same checksum gate as a resume."""
        model = Encoder(PRESETS["kab"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        save(tmp_path, model, optimizer, TrainingState(), name=BEST)
        (tmp_path / BEST).write_bytes(b"truncated")
        with pytest.raises(ModelError, match="corrupt"):
            load_encoder(tmp_path, "kab", name=BEST)
