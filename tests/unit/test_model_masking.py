from __future__ import annotations

import pytest
import torch

from agbalu.model.config import ModelError
from agbalu.model.masking import SpanMasker
from agbalu.model.modeling import IGNORE_INDEX

VOCAB = 1_000
MASK_ID = 4
SPECIALS = 5


def masker(**overrides: float | int) -> SpanMasker:
    settings: dict[str, float | int] = {
        "vocab_size": VOCAB,
        "mask_token_id": MASK_ID,
        "n_special_tokens": SPECIALS,
        "random_p": 0.1,
        "keep_p": 0.1,
        "max_span_length": 3,
    }
    settings.update(overrides)
    return SpanMasker(**settings)  # type: ignore[arg-type]


def window(length: int = 128, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(SPECIALS, VOCAB, (length,), generator=generator)


def generator(seed: int = 7) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestValidation:
    def test_rejects_a_zero_span_length(self) -> None:
        with pytest.raises(ModelError, match="max_span_length"):
            masker(max_span_length=0)

    def test_rejects_a_special_count_covering_the_vocabulary(self) -> None:
        with pytest.raises(ModelError, match="ordinary piece"):
            masker(n_special_tokens=VOCAB)

    def test_rejects_replacement_probabilities_above_one(self) -> None:
        with pytest.raises(ModelError, match="proportion"):
            masker(random_p=0.8, keep_p=0.8)

    def test_rejects_a_three_dimensional_input(self) -> None:
        with pytest.raises(ModelError, match="1-D window or a 2-D batch"):
            masker()(torch.zeros(2, 3, 4, dtype=torch.long), 0.15, generator=generator())

    def test_a_batch_keeps_its_shape(self) -> None:
        batch = torch.randint(5, VOCAB, (8, 64), dtype=torch.long)
        inputs, labels = masker()(batch, 0.15, generator=generator())
        assert inputs.shape == labels.shape == batch.shape

    def test_a_batch_masks_every_row(self) -> None:
        batch = torch.randint(5, VOCAB, (8, 64), dtype=torch.long)
        _, labels = masker()(batch, 0.15, generator=generator())
        assert bool(((labels != IGNORE_INDEX).sum(dim=1) > 0).all())

    def test_rows_of_a_batch_are_masked_independently(self) -> None:
        batch = torch.arange(5, 5 + 64, dtype=torch.long).expand(8, 64).contiguous()
        _, labels = masker()(batch, 0.3, generator=generator())
        positions = [
            tuple((labels[row] != IGNORE_INDEX).nonzero().flatten().tolist()) for row in range(8)
        ]
        assert len(set(positions)) > 1, "identical rows masked identically means one shared draw"


class TestRate:
    @pytest.mark.parametrize("probability", [0.15, 0.3])
    def test_hits_the_requested_rate(self, probability: float) -> None:
        _, labels = masker()(window(512), probability, generator=generator())
        observed = float((labels != IGNORE_INDEX).float().mean())
        assert observed == pytest.approx(probability, abs=0.05)

    def test_always_masks_at_least_one_position(self) -> None:
        _, labels = masker()(window(64), 0.0, generator=generator())
        assert int((labels != IGNORE_INDEX).sum()) >= 1

    def test_a_probability_of_one_masks_everything_maskable(self) -> None:
        tokens = window(64)
        _, labels = masker()(tokens, 1.0, generator=generator())
        assert int((labels != IGNORE_INDEX).sum()) == 64


class TestSpecialTokens:
    def test_special_ids_are_never_masked(self) -> None:
        tokens = window(128)
        tokens[:SPECIALS] = torch.arange(SPECIALS)
        _, labels = masker()(tokens, 0.5, generator=generator())
        assert bool((labels[:SPECIALS] == IGNORE_INDEX).all())

    def test_random_replacements_are_never_special_ids(self) -> None:
        inputs, labels = masker(random_p=1.0, keep_p=0.0)(window(256), 0.5, generator=generator())
        replaced = (labels != IGNORE_INDEX) & (inputs != MASK_ID)
        assert bool((inputs[replaced] >= SPECIALS).all())

    def test_a_window_of_only_specials_masks_nothing(self) -> None:
        tokens = torch.zeros(16, dtype=torch.long)
        inputs, labels = masker()(tokens, 0.3, generator=generator())
        assert bool((labels == IGNORE_INDEX).all())
        assert torch.equal(inputs, tokens)


class TestReplacementPolicy:
    def test_all_mask_when_neither_random_nor_keep(self) -> None:
        tokens = window(128)
        inputs, labels = masker(random_p=0.0, keep_p=0.0)(tokens, 0.3, generator=generator())
        masked = labels != IGNORE_INDEX
        assert bool((inputs[masked] == MASK_ID).all())

    def test_keep_leaves_the_token_but_still_scores_it(self) -> None:
        tokens = window(128)
        inputs, labels = masker(random_p=0.0, keep_p=1.0)(tokens, 0.3, generator=generator())
        masked = labels != IGNORE_INDEX
        assert int(masked.sum()) > 0
        assert torch.equal(inputs[masked], tokens[masked])

    def test_labels_always_carry_the_original_token(self) -> None:
        tokens = window(128)
        _, labels = masker()(tokens, 0.3, generator=generator())
        masked = labels != IGNORE_INDEX
        assert torch.equal(labels[masked], tokens[masked])

    def test_unmasked_positions_are_left_alone(self) -> None:
        tokens = window(128)
        inputs, labels = masker()(tokens, 0.3, generator=generator())
        untouched = labels == IGNORE_INDEX
        assert torch.equal(inputs[untouched], tokens[untouched])


class TestSpanStructure:
    def test_masked_positions_form_runs_rather_than_singletons(self) -> None:
        """The whole point of span masking: mean run length must exceed 1."""
        _, labels = masker()(window(512), 0.3, generator=generator())
        selected = (labels != IGNORE_INDEX).int()
        runs = int((selected[1:] > selected[:-1]).sum()) + int(selected[0])
        assert int(selected.sum()) / runs > 1.5

    def test_span_ids_are_non_decreasing(self) -> None:
        indices = masker().span_indices(4, 64, generator())
        assert bool((indices[:, 1:] >= indices[:, :-1]).all())

    def test_span_ids_cover_the_window(self) -> None:
        assert masker().span_indices(4, 64, generator()).shape == (4, 64)

    def test_a_span_length_of_one_degenerates_to_token_masking(self) -> None:
        indices = masker(max_span_length=1).span_indices(3, 32, generator())
        assert torch.equal(indices, torch.arange(32).expand(3, 32))

    def test_the_first_position_always_opens_the_first_span(self) -> None:
        assert bool((masker().span_indices(4, 64, generator())[:, 0] == 0).all())


class TestDeterminism:
    def test_the_same_generator_seed_gives_the_same_masking(self) -> None:
        tokens = window(128)
        first = masker()(tokens, 0.2, generator=generator(3))
        second = masker()(tokens, 0.2, generator=generator(3))
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])

    def test_a_different_seed_gives_different_masking(self) -> None:
        tokens = window(128)
        first = masker()(tokens, 0.2, generator=generator(1))
        second = masker()(tokens, 0.2, generator=generator(2))
        assert not torch.equal(first[1], second[1])

    def test_the_input_window_is_not_mutated(self) -> None:
        tokens = window(64)
        original = tokens.clone()
        masker()(tokens, 0.3, generator=generator())
        assert torch.equal(tokens, original)


class TestEdgeCases:
    def test_a_single_token_window(self) -> None:
        tokens = torch.tensor([42])
        inputs, labels = masker()(tokens, 0.5, generator=generator())
        assert inputs.shape == labels.shape == (1,)

    def test_a_window_shorter_than_the_maximum_span(self) -> None:
        tokens = window(2)
        inputs, labels = masker(max_span_length=8)(tokens, 0.5, generator=generator())
        assert inputs.numel() == labels.numel() == 2
