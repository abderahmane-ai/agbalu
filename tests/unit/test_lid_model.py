from __future__ import annotations

import pytest
import torch

from agbalu.lid.config import ModelConfig
from agbalu.lid.model import (
    AttentivePooling,
    CharClassifier,
    ConvStem,
    MarginHead,
    apply_rope,
    precompute_rope_freqs,
)


@pytest.fixture(name="config")
def fixture_config() -> ModelConfig:
    return ModelConfig(embed_dim=32, num_heads=4, num_layers=2, intermediate_dim=64, conv_dim=8)


class TestConvStem:
    def test_the_stem_preserves_the_sequence_length(self) -> None:
        stem = ConvStem(16, (3, 5, 7), 4)
        assert stem(torch.randn(2, 11, 16)).shape == (2, 11, 16)

    def test_a_sequence_shorter_than_the_widest_kernel_still_runs(self) -> None:
        stem = ConvStem(16, (3, 5, 7), 4)
        assert stem(torch.randn(1, 2, 16)).shape == (1, 2, 16)


class TestRotaryEmbedding:
    def test_rope_preserves_shape_and_dtype(self) -> None:
        freqs = precompute_rope_freqs(8, 16)
        x = torch.randn(2, 3, 5, 8)
        rotated = apply_rope(x, freqs)
        assert rotated.shape == x.shape
        assert rotated.dtype == x.dtype

    def test_rope_leaves_the_first_position_unchanged(self) -> None:
        """Position zero rotates by angle zero; if it moves, the frequency table is off
        by one and every position is mislabelled."""
        freqs = precompute_rope_freqs(8, 16)
        x = torch.randn(1, 1, 4, 8)
        assert torch.allclose(apply_rope(x, freqs)[:, :, 0], x[:, :, 0], atol=1e-6)

    def test_rope_preserves_the_norm_of_each_head_vector(self) -> None:
        freqs = precompute_rope_freqs(8, 16)
        x = torch.randn(2, 2, 6, 8)
        assert torch.allclose(apply_rope(x, freqs).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


class TestAttentivePooling:
    def test_padding_does_not_reach_the_pooled_vector(self) -> None:
        pool = AttentivePooling(8).eval()
        x = torch.randn(1, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False]])
        padded = x.clone()
        padded[0, 3:] = 1e3
        assert torch.allclose(pool(x, mask), pool(padded, mask), atol=1e-3)

    def test_a_fully_masked_row_does_not_produce_nan(self) -> None:
        pool = AttentivePooling(8).eval()
        out = pool(torch.randn(1, 4, 8), torch.zeros(1, 4, dtype=torch.bool))
        assert torch.isfinite(out).all()


class TestMarginHead:
    def test_without_labels_the_logits_are_the_scaled_cosine(self) -> None:
        head = MarginHead(8, 3, scale=10.0)
        logits = head(torch.randn(4, 8))
        assert logits.abs().max() <= 10.0 + 1e-4

    def test_the_margin_lowers_only_the_target_logit(self) -> None:
        head = MarginHead(8, 3, scale=1.0)
        head.set_margins(torch.tensor([0.5, 0.0, 0.0]))
        x = torch.randn(4, 8)
        labels = torch.tensor([0, 0, 1, 2])
        plain = head(x)
        with_margin = head(x, labels)
        target = torch.arange(4), labels
        assert torch.allclose(
            with_margin[target], plain[target] - torch.tensor([0.5, 0.5, 0.0, 0.0]), atol=1e-5
        )
        assert torch.allclose(
            with_margin[:, 1][2:], plain[:, 1][2:] - torch.tensor([0.0, 0.0]), atol=1e-5
        )

    def test_zero_margins_leave_the_logits_alone(self) -> None:
        head = MarginHead(8, 3, scale=4.0)
        x = torch.randn(5, 8)
        labels = torch.tensor([0, 1, 2, 0, 1])
        assert torch.allclose(head(x, labels), head(x), atol=1e-6)

    def test_a_margin_vector_of_the_wrong_length_is_refused(self) -> None:
        head = MarginHead(8, 3, scale=4.0)
        with pytest.raises(ValueError, match="expected \\(3,\\) margins"):
            head.set_margins(torch.zeros(4))


class TestCharClassifier:
    def test_the_forward_produces_the_three_declared_outputs(self, config: ModelConfig) -> None:
        model = CharClassifier(config)
        out = model(torch.randint(2, 258, (3, 9)), pad_mask=torch.ones(3, 9, dtype=torch.bool))
        assert out.logits.shape == (3, config.num_classes)
        assert out.embeddings.shape == (3, config.embed_dim)
        assert out.projections.shape == (3, config.projection_dim)

    def test_the_projection_is_unit_norm(self, config: ModelConfig) -> None:
        model = CharClassifier(config)
        out = model(torch.randint(2, 258, (4, 7)))
        assert torch.allclose(out.projections.norm(dim=1), torch.ones(4), atol=1e-5)

    def test_padding_does_not_change_a_row_s_logits(self, config: ModelConfig) -> None:
        """A short row batched with a long one must score as it does alone, or the batch
        composition becomes a feature."""
        model = CharClassifier(config).eval()
        row = torch.randint(2, 258, (1, 5))
        alone = model(row, pad_mask=torch.ones(1, 5, dtype=torch.bool)).logits
        padded = torch.cat([row, torch.zeros(1, 6, dtype=torch.long)], dim=1)
        mask = torch.tensor([[True] * 5 + [False] * 6])
        assert torch.allclose(alone, model(padded, pad_mask=mask).logits, atol=1e-4)

    def test_a_single_byte_input_runs(self, config: ModelConfig) -> None:
        model = CharClassifier(config).eval()
        assert torch.isfinite(model(torch.randint(2, 258, (1, 1))).logits).all()

    def test_the_rope_table_is_not_in_the_state_dict(self, config: ModelConfig) -> None:
        """A non-persistent buffer comes back from a checkpoint as uninitialised memory
        if it is saved; it is derived, so it is rebuilt instead."""
        assert "rope_freqs" not in CharClassifier(config).state_dict()

    def test_a_checkpoint_round_trips(self, config: ModelConfig) -> None:
        source = CharClassifier(config).eval()
        target = CharClassifier(config).eval()
        target.load_state_dict(source.state_dict())
        ids = torch.randint(2, 258, (2, 6))
        assert torch.allclose(source(ids).logits, target(ids).logits, atol=1e-6)

    def test_the_release_size_is_what_the_name_claims(self) -> None:
        count = CharClassifier(ModelConfig()).trainable_parameters()
        assert 4_500_000 < count < 5_500_000, count

    def test_dropout_is_off_in_eval_so_two_passes_agree(self, config: ModelConfig) -> None:
        model = CharClassifier(config).eval()
        ids = torch.randint(2, 258, (2, 8))
        assert torch.allclose(model(ids).logits, model(ids).logits, atol=1e-6)
