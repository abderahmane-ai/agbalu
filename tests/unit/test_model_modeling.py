from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from agbalu.model.config import PRESETS, ModelConfig
from agbalu.model.modeling import (
    IGNORE_INDEX,
    Attention,
    Encoder,
    GeGLU,
    log_bucket_position,
)

TINY = ModelConfig(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_hidden_layers=2,
    max_position_embeddings=64,
    position_bucket_size=8,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
)


def batch(config: ModelConfig, rows: int = 3, length: int = 12) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    ids = torch.randint(5, config.vocab_size, (rows, length))
    attention = torch.ones(rows, length, dtype=torch.bool)
    labels = torch.full((rows, length), IGNORE_INDEX)
    labels[:, ::3] = ids[:, ::3]
    return ids, attention, labels


class TestLogBucketPosition:
    def test_is_antisymmetric_about_the_diagonal(self) -> None:
        offsets = torch.arange(-20, 21)
        forward = log_bucket_position(offsets, 8, 64)
        backward = log_bucket_position(-offsets, 8, 64)
        assert torch.equal(forward, -backward)

    def test_the_diagonal_is_zero(self) -> None:
        assert int(log_bucket_position(torch.zeros(1, dtype=torch.long), 8, 64)[0]) == 0

    def test_near_offsets_keep_their_own_bucket(self) -> None:
        offsets = torch.arange(-3, 4)
        assert torch.equal(log_bucket_position(offsets, 8, 64), offsets)

    def test_distant_offsets_are_compressed(self) -> None:
        near = log_bucket_position(torch.tensor([10]), 8, 512)
        far = log_bucket_position(torch.tensor([400]), 8, 512)
        assert int(far[0]) - int(near[0]) < 400 - 10

    def test_indices_stay_inside_the_embedding_table(self) -> None:
        bucket_size, max_position = 32, 512
        offsets = torch.arange(max_position).unsqueeze(1) - torch.arange(max_position).unsqueeze(0)
        indices = bucket_size - 1 + log_bucket_position(offsets, bucket_size, max_position)
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 2 * bucket_size - 1


class TestGeGLU:
    def test_halves_the_last_dimension(self) -> None:
        assert GeGLU()(torch.randn(2, 5, 64)).shape == (2, 5, 32)

    def test_a_zero_gate_closes_the_unit(self) -> None:
        x = torch.cat([torch.ones(1, 4), torch.zeros(1, 4)], dim=-1)
        assert torch.allclose(GeGLU()(x), torch.zeros(1, 4))


class TestParameterCount:
    @pytest.mark.parametrize("preset", ["small", "kab", "base"])
    def test_the_analytic_count_matches_the_built_module(self, preset: str) -> None:
        """The budget is computed from `ModelConfig.parameters` before anything is
        allocated, so it has to agree with reality or the cost estimate is fiction."""
        config = PRESETS[preset]  # type: ignore[index]
        assert Encoder(config).parameter_count() == config.parameters

    def test_tied_weights_are_counted_once(self) -> None:
        model = Encoder(TINY)
        assert model.classifier.decoder.weight is model.embedding.word_embedding.weight
        naive = sum(p.numel() for p in model.parameters())
        assert model.parameter_count() <= naive


class TestForward:
    def test_loss_at_initialisation_is_about_log_vocab(self) -> None:
        """A correctly initialised classifier is uniform over the vocabulary."""
        torch.manual_seed(0)
        model = Encoder(TINY).eval()
        loss = model(*batch(TINY)).loss.detach()
        assert abs(float(loss) - math.log(TINY.vocab_size)) < 1.0

    def test_every_parameter_receives_a_gradient(self) -> None:
        model = Encoder(TINY)
        model(*batch(TINY)).loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_no_gradient_is_non_finite(self) -> None:
        model = Encoder(TINY)
        model(*batch(TINY)).loss.backward()
        assert all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)

    def test_only_labelled_positions_are_scored(self) -> None:
        ids, attention, labels = batch(TINY)
        model = Encoder(TINY).eval()
        assert model(ids, attention, labels).num_tokens == int((labels != IGNORE_INDEX).sum())

    def test_a_batch_with_nothing_masked_yields_a_finite_zero_loss(self) -> None:
        """Late in training a window can come back unmasked; it must not produce NaN."""
        ids, attention, _ = batch(TINY)
        empty = torch.full_like(ids, IGNORE_INDEX)
        output = Encoder(TINY).eval()(ids, attention, empty)
        assert output.num_tokens == 0
        assert float(output.loss.detach()) == 0.0
        assert torch.isfinite(output.loss)

    def test_z_loss_is_non_negative(self) -> None:
        assert float(Encoder(TINY).eval()(*batch(TINY)).z_loss.detach()) >= 0.0

    def test_accuracy_is_a_proportion(self) -> None:
        assert 0.0 <= Encoder(TINY).eval()(*batch(TINY)).accuracy <= 1.0


class TestPadding:
    def test_padded_rows_do_not_change_the_real_ones(self) -> None:
        """The load-bearing correctness property of the masked attention."""
        torch.manual_seed(0)
        model = Encoder(TINY).eval()
        ids = torch.randint(5, TINY.vocab_size, (2, 16))
        attention = torch.ones(2, 16, dtype=torch.bool)
        attention[0, 10:] = False
        with torch.no_grad():
            padded = model.contextualise(ids, attention)
            short = model.contextualise(ids[:1, :10], attention[:1, :10])
        assert torch.allclose(padded[0, :10], short[0], atol=1e-4)

    def test_changing_a_padded_token_changes_nothing(self) -> None:
        torch.manual_seed(0)
        model = Encoder(TINY).eval()
        ids = torch.randint(5, TINY.vocab_size, (1, 16))
        attention = torch.ones(1, 16, dtype=torch.bool)
        attention[0, 8:] = False
        altered = ids.clone()
        altered[0, 8:] = 7
        with torch.no_grad():
            before = model.contextualise(ids, attention)[0, :8]
            after = model.contextualise(altered, attention)[0, :8]
        assert torch.allclose(before, after, atol=1e-5)


class TestAttentionGeometry:
    def test_the_scale_accounts_for_three_score_terms(self) -> None:
        attention = Attention(TINY)
        assert attention.scale == pytest.approx(1.0 / math.sqrt(3 * TINY.head_size))

    def test_a_longer_window_than_the_buffer_is_handled(self) -> None:
        """`max_position_embeddings` is a buffer size, not a hard limit."""
        config = replace(TINY, max_position_embeddings=16)
        model = Encoder(config).eval()
        ids = torch.randint(5, config.vocab_size, (1, 24))
        with torch.no_grad():
            hidden = model.contextualise(ids, torch.ones(1, 24, dtype=torch.bool))
        assert hidden.shape == (1, 24, config.hidden_size)
        assert bool(hidden.isfinite().all())
