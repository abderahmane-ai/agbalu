"""Held-out likelihood of a causal model (task 11.2).

Two things can silently corrupt a perplexity and neither shows up as an error: scoring the
padding, and shifting the labels the wrong way. Both are asserted against models whose exact
loss is known — uniform logits give `ln(vocabulary)` per token, and an oracle that puts its
mass on the next token gives zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch

from agbalu.llm.baseline import (
    IGNORE_INDEX,
    LOSS_CHUNK_BYTES,
    BaselineError,
    Likelihood,
    Scored,
    SpecialTokenError,
    batches,
    chunk_rows,
    leading_and_pad,
    likelihoods,
    measure,
    prepare,
    summed_nll,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

VOCAB = 32
BOS = 1
PAD = 0


@dataclass
class Output:
    logits: torch.Tensor


class Uniform:
    """Every token equally likely, so the loss of one token is exactly `ln(VOCAB)`."""

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Output:
        assert attention_mask.shape == input_ids.shape
        return Output(logits=torch.zeros(*input_ids.shape, VOCAB))


class Oracle:
    """All mass on the token that actually follows. A shift in either direction breaks it."""

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Output:
        assert attention_mask.shape == input_ids.shape
        rows, width = input_ids.shape
        logits = torch.zeros(rows, width, VOCAB)
        for row in range(rows):
            for position in range(width - 1):
                logits[row, position, int(input_ids[row, position + 1])] = 60.0
        return Output(logits=logits)


def encoder(table: dict[str, list[int]]) -> Callable[[list[str]], list[list[int]]]:
    def encode(texts: list[str]) -> list[list[int]]:
        return [table[text] for text in texts]

    return encode


class TestScored:
    def test_perplexity_is_the_exponential_of_the_mean_loss(self) -> None:
        scored = Scored("s", "kab_Latn", 2, 10, 40, nll=20.0, truncated=0)
        assert scored.nll_per_token == 2.0
        assert scored.perplexity == pytest.approx(math.exp(2.0))

    def test_bits_per_character_is_the_tokenizer_free_figure(self) -> None:
        """Task 11.12 changes the tokenizer, and perplexity per token moves with it."""
        scored = Scored("s", "kab_Latn", 1, 10, 100, nll=math.log(2) * 100, truncated=0)
        assert scored.bits_per_character == pytest.approx(1.0)

    def test_a_set_with_no_tokens_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="0 tokens"):
            Scored("s", "kab_Latn", 1, 0, 10, nll=0.0, truncated=0)

    def test_a_set_with_no_characters_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="characters"):
            Scored("s", "kab_Latn", 1, 5, 0, nll=1.0, truncated=0)

    def test_the_row_carries_every_figure_it_is_derived_from(self) -> None:
        row = Scored("s", "kab_Latn", 2, 10, 40, nll=20.0, truncated=1).as_dict()
        assert set(row) == {
            "name",
            "language",
            "documents",
            "tokens",
            "characters",
            "nll",
            "truncated",
            "nll_per_token",
            "perplexity",
            "bits_per_character",
        }


class TestLikelihood:
    def test_a_document_with_no_scored_token_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="0 tokens"):
            Likelihood(nll=0.0, tokens=0, truncated=False)


class TestMeasure:
    def test_characters_come_from_the_documents_not_the_tokens(self) -> None:
        texts = ["azul", "aɣbalu"]
        values = [Likelihood(2.0, 4, False), Likelihood(3.0, 6, False)]
        scored = measure("s", "kab_Latn", texts, values)
        assert scored.characters == len("azul") + len("aɣbalu")
        assert scored.tokens == 10
        assert scored.nll == pytest.approx(5.0)

    def test_truncation_is_counted_rather_than_hidden(self) -> None:
        values = [Likelihood(1.0, 2, True), Likelihood(1.0, 2, False)]
        assert measure("s", "kab_Latn", ["ab", "cd"], values).truncated == 1

    def test_a_length_mismatch_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="likelihoods for"):
            measure("s", "kab_Latn", ["a", "b"], [Likelihood(1.0, 1, False)])

    def test_an_empty_population_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="nothing to score"):
            measure("s", "kab_Latn", [], [])


class TestPrepare:
    def test_every_document_gains_a_leading_bos(self) -> None:
        """No tokenizer here adds one where it is needed, so the first real token would
        otherwise be scored with no context at all."""
        prepared = prepare(["a"], encoder({"a": [7, 8]}), BOS, 512)
        assert prepared == [[BOS, 7, 8]]

    def test_a_long_document_is_truncated_to_the_window(self) -> None:
        prepared = prepare(["a"], encoder({"a": list(range(20))}), BOS, 8)
        assert len(prepared[0]) == 8
        assert prepared[0][0] == BOS

    def test_a_tokenizer_dropping_a_row_is_caught(self) -> None:
        def encode(texts: list[str]) -> list[list[int]]:
            return [[2]] * (len(texts) - 1)

        with pytest.raises(BaselineError, match="rows for"):
            prepare(["a", "b"], encode, BOS, 512)

    def test_a_document_encoding_to_nothing_is_caught(self) -> None:
        with pytest.raises(BaselineError, match="no tokens"):
            prepare(["a"], encoder({"a": []}), BOS, 512)

    def test_a_window_with_no_room_to_score_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="at least one scored token"):
            prepare(["a"], encoder({"a": [3]}), BOS, 1)


class TestBatches:
    def test_the_last_batch_is_short(self) -> None:
        assert [len(b) for b in batches([[1]] * 7, 3)] == [3, 3, 1]

    def test_a_zero_size_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="batch size must be positive"):
            list(batches([[1]], 0))


class TestLikelihoods:
    def table(self, texts: Sequence[str], ids: Sequence[list[int]]) -> dict[str, list[int]]:
        return dict(zip(texts, ids, strict=True))

    def test_uniform_logits_cost_exactly_the_log_of_the_vocabulary(self) -> None:
        texts = ["a"]
        table = self.table(texts, [[5, 6, 7]])
        values = likelihoods(texts, encoder(table), Uniform(), bos_id=BOS, pad_id=PAD)
        assert values[0].tokens == 3
        assert values[0].nll == pytest.approx(3 * math.log(VOCAB), abs=1e-4)

    def test_an_oracle_costs_nothing_which_pins_the_label_shift(self) -> None:
        texts = ["a"]
        table = self.table(texts, [[5, 6, 7, 8]])
        values = likelihoods(texts, encoder(table), Oracle(), bos_id=BOS, pad_id=PAD)
        assert values[0].nll == pytest.approx(0.0, abs=1e-3)

    def test_padding_is_neither_scored_nor_counted(self) -> None:
        """A ragged batch. Every extra token would add `ln(VOCAB)` to the shorter row."""
        texts = ["short", "long"]
        table = self.table(texts, [[5], [5, 6, 7, 8, 9]])
        values = likelihoods(
            texts,
            encoder(table),
            Uniform(),
            bos_id=BOS,
            pad_id=PAD,
            batch_size=2,
        )
        assert [v.tokens for v in values] == [1, 5]
        assert values[0].nll == pytest.approx(math.log(VOCAB), abs=1e-4)

    def test_the_batch_size_does_not_change_the_result(self) -> None:
        texts = [f"t{i}" for i in range(5)]
        table = self.table(texts, [list(range(2, 2 + i + 1)) for i in range(5)])
        one = likelihoods(texts, encoder(table), Oracle(), bos_id=BOS, pad_id=PAD, batch_size=1)
        many = likelihoods(texts, encoder(table), Oracle(), bos_id=BOS, pad_id=PAD, batch_size=4)
        assert [v.tokens for v in one] == [v.tokens for v in many]
        assert [round(v.nll, 3) for v in one] == [round(v.nll, 3) for v in many]

    def test_results_come_back_in_input_order(self) -> None:
        texts = ["a", "b"]
        table = self.table(texts, [[5, 6, 7], [5]])
        values = likelihoods(texts, encoder(table), Uniform(), bos_id=BOS, pad_id=PAD)
        assert [v.tokens for v in values] == [3, 1]

    def test_a_document_filling_the_window_is_flagged_as_truncated(self) -> None:
        texts = ["a"]
        table = self.table(texts, [list(range(2, 20))])
        values = likelihoods(
            texts,
            encoder(table),
            Uniform(),
            bos_id=BOS,
            pad_id=PAD,
            max_length=8,
        )
        assert values[0].truncated
        assert values[0].tokens == 7


class TestChunkRows:
    """The slice width that keeps the fp32 upcast bounded whatever the vocabulary is."""

    def test_a_248k_vocabulary_gives_a_270_token_slice(self) -> None:
        """262,144 classes make one token's fp32 row exactly 1 MiB, so the 256 MiB budget
        buys 256 positions. This is the arithmetic that the A10G OOM came down to."""
        assert 262_144 * 4 == 1024 * 1024
        assert chunk_rows(262_144) == 256

    def test_a_small_vocabulary_takes_the_whole_batch_in_one_slice(self) -> None:
        assert chunk_rows(32) == LOSS_CHUNK_BYTES // (32 * 4)

    def test_an_absurd_vocabulary_still_yields_a_usable_slice(self) -> None:
        """Never zero: a slice of no positions would score nothing and report it as 0.0."""
        assert chunk_rows(LOSS_CHUNK_BYTES) == 1

    def test_a_non_positive_vocabulary_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="vocabulary must be positive"):
            chunk_rows(0)


class TestSummedNll:
    """Chunking is an optimisation, so its only defensible property is that it changes
    nothing. Every case below compares it against the whole-batch computation it replaced."""

    def reference(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The unchunked form, kept here as the oracle."""
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().transpose(1, 2),
            targets,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        return losses.sum(dim=1)

    @pytest.mark.parametrize(("rows", "width", "vocab"), [(1, 2, 8), (3, 9, 16), (4, 17, 64)])
    def test_it_equals_the_unchunked_computation(self, rows: int, width: int, vocab: int) -> None:
        generator = torch.Generator().manual_seed(20260810)
        logits = torch.randn(rows, width, vocab, generator=generator)
        targets = torch.randint(0, vocab, (rows, width - 1), generator=generator)
        assert torch.allclose(summed_nll(logits, targets), self.reference(logits, targets))

    def test_ignored_positions_stay_out_of_the_sum(self) -> None:
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(2, 6, 16, generator=generator)
        targets = torch.randint(0, 16, (2, 5), generator=generator)
        targets[0, 3:] = IGNORE_INDEX
        targets[1, 1:] = IGNORE_INDEX
        assert torch.allclose(summed_nll(logits, targets), self.reference(logits, targets))

    def test_a_row_that_is_entirely_padding_sums_to_zero(self) -> None:
        logits = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(3))
        targets = torch.full((1, 3), IGNORE_INDEX)
        assert summed_nll(logits, targets).item() == 0.0

    def test_a_slice_narrower_than_the_row_is_still_exact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forces several chunks per row, which the default budget never would at this size."""
        generator = torch.Generator().manual_seed(99)
        logits = torch.randn(2, 13, 8, generator=generator)
        targets = torch.randint(0, 8, (2, 12), generator=generator)
        expected = self.reference(logits, targets)
        monkeypatch.setattr("agbalu.llm.baseline.LOSS_CHUNK_BYTES", 8 * 4 * 3)
        assert chunk_rows(8, 8 * 4 * 3) == 3
        assert torch.allclose(summed_nll(logits, targets), expected)

    def test_a_shape_mismatch_between_logits_and_targets_is_named(self) -> None:
        logits = torch.randn(2, 5, 8)
        with pytest.raises(BaselineError, match="off by exactly one"):
            summed_nll(logits, torch.zeros(2, 5, dtype=torch.long))

    def test_a_row_count_mismatch_is_named(self) -> None:
        logits = torch.randn(2, 5, 8)
        with pytest.raises(BaselineError, match="2 logit rows against 3 target rows"):
            summed_nll(logits, torch.zeros(3, 4, dtype=torch.long))

    def test_the_upcast_slice_never_exceeds_the_budget(self) -> None:
        """The property the OOM violated: fp32 surface per slice is bounded by the budget,
        independently of batch size, sequence length and vocabulary."""
        for vocab in (32, 4096, 262_144, 1_000_000):
            assert chunk_rows(vocab) * vocab * 4 <= max(LOSS_CHUNK_BYTES, vocab * 4)


class TestLeadingAndPad:
    """The base has no `<bos>`, and refusing that is what would have failed on the GPU.

    `Qwen/Qwen3.5-2B` carries no bos, eos 248044 and pad 248044, read off its published
    tokenizer config. The other rows are the shapes a checkpoint can present, since the
    resolution has to be total over them.
    """

    def test_uses_bos_and_pad_when_the_checkpoint_has_them(self) -> None:
        assert leading_and_pad(2, 1, 0) == (2, 0)

    def test_falls_back_to_eos_for_both_when_neither_exists(self) -> None:
        assert leading_and_pad(None, 248044, None) == (248044, 248044)

    def test_falls_back_only_for_the_missing_one(self) -> None:
        assert leading_and_pad(None, 151645, 151643) == (151645, 151643)

    def test_keeps_a_zero_id_rather_than_treating_it_as_absent(self) -> None:
        # `if not bos_id` would silently swap a real id 0 for eos.
        assert leading_and_pad(0, 9, 0) == (0, 0)

    def test_refuses_a_checkpoint_with_no_eos(self) -> None:
        with pytest.raises(SpecialTokenError, match="no eos token"):
            leading_and_pad(None, None, None)

    def test_refuses_no_eos_even_when_bos_and_pad_exist(self) -> None:
        with pytest.raises(SpecialTokenError, match="no eos token"):
            leading_and_pad(2, None, 0)
