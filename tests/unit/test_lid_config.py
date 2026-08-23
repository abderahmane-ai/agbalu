from __future__ import annotations

import pytest

from agbalu.bench.lid_sources import CLASSES
from agbalu.lid.config import VOCAB_SIZE, ConfigError, ModelConfig, TrainConfig


class TestModelConfig:
    def test_the_default_covers_every_class_the_source_table_defines(self) -> None:
        assert ModelConfig().classes == CLASSES
        assert ModelConfig().num_classes == len(CLASSES)

    def test_the_vocabulary_is_every_byte_plus_the_two_reserved_ids(self) -> None:
        assert VOCAB_SIZE == 258
        assert ModelConfig().vocab_size == VOCAB_SIZE

    def test_an_indivisible_head_count_is_refused_at_construction(self) -> None:
        with pytest.raises(ConfigError, match="not divisible by num_heads"):
            ModelConfig(embed_dim=100, num_heads=8)

    def test_an_even_kernel_is_refused_because_it_would_shift_the_sequence(self) -> None:
        with pytest.raises(ConfigError, match="must be odd"):
            ModelConfig(conv_kernels=(3, 4))

    def test_no_layers_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="num_layers must be at least"):
            ModelConfig(num_layers=0)

    def test_an_empty_class_list_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="nothing to classify"):
            ModelConfig(classes=())

    def test_the_config_is_frozen(self) -> None:
        config = ModelConfig()
        with pytest.raises(AttributeError):
            config.embed_dim = 512  # type: ignore[misc]


class TestTrainConfig:
    def test_the_batch_is_the_product_of_the_group_shape(self) -> None:
        settings = TrainConfig(groups_per_batch=12, samples_per_group=10)
        assert settings.batch_size == 120

    @pytest.mark.parametrize(
        "field", ["epochs", "groups_per_batch", "samples_per_group", "eval_batch_size"]
    )
    def test_a_non_positive_shape_is_refused(self, field: str) -> None:
        with pytest.raises(ConfigError, match=f"{field} must be at least 1"):
            TrainConfig(**{field: 0})

    def test_the_default_batch_holds_more_than_half_the_built_groups(self) -> None:
        """A batch narrower than that never contrasts most cells with each other, which
        is what the contrastive term is for."""
        assert TrainConfig().groups_per_batch >= 10
