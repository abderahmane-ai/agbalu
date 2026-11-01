"""Unit tests for Dihya-22M neural architecture components."""

from __future__ import annotations

import torch

from agbalu.dihya.config import ModelConfig
from agbalu.dihya.model import Classifier, ConvStem, MarginHead, Output


def test_model_config_invariants() -> None:
    """Verify configuration dimensions and parameter computation."""
    config = ModelConfig()
    assert config.embed_dim == 384
    assert config.num_heads == 6
    assert config.head_dim == 64
    assert config.num_layers == 10
    assert config.max_length == 512
    assert config.conv_kernels == (3, 5, 7, 11)


def test_classifier_parameter_count() -> None:
    """Verify published and trainable parameter counts (~22.2M)."""
    config = ModelConfig()
    model = Classifier(config)
    total_params = model.trainable_parameters()
    # Ensure parameter count sits tightly in the 21.8M - 22.5M corridor
    assert total_params == 22_198_528


def test_padding_values_and_batch_neighbors_do_not_change_logits() -> None:
    """Mask embeddings before convolution as well as attention and pooling."""
    config = ModelConfig(embed_dim=16, num_heads=2, num_layers=1, intermediate_dim=32, conv_dim=8)
    model = Classifier(config).eval()
    tokens = torch.tensor([[3, 5, 7]])
    padded = torch.tensor([[3, 5, 7, 99, 100, 101], [3, 8, 4, 6, 5, 7]])
    mask = torch.tensor([[True, True, True, False, False, False], [True] * 6])
    with torch.no_grad():
        alone = model(tokens).logits[0]
        together = model(padded, pad_mask=mask).logits[0]
    assert torch.allclose(alone, together, atol=1e-5, rtol=1e-5)


def test_conv_stem_shape_preservation() -> None:
    """Verify 4-scale ConvStem preserves batch and sequence length."""
    dim = 64
    kernels = (3, 5, 7, 11)
    stem = ConvStem(dim=dim, kernels=kernels, branch_dim=32)

    x = torch.randn(2, 48, dim)
    out = stem(x)
    assert out.shape == x.shape


def test_margin_head_ldam_shift() -> None:
    """Verify MarginHead applies additive margin only to target label."""
    head = MarginHead(in_features=32, num_classes=5, scale=10.0)
    margins = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    head.set_margins(margins)

    x = torch.randn(2, 32)
    labels = torch.tensor([1, 4])

    logits_unmargined = head(x)
    logits_margined = head(x, labels=labels)

    # Off-target logits must be identical
    assert torch.allclose(logits_unmargined[:, 0], logits_margined[:, 0])

    # Target logit must be decreased by margin * scale
    diff_0 = logits_unmargined[0, 1] - logits_margined[0, 1]
    assert torch.isclose(diff_0, margins[1] * head.scale, atol=1e-5)


def test_classifier_forward_pass() -> None:
    """Verify multi-task forward pass yields correct tensor shapes."""
    config = ModelConfig(num_layers=2)
    model = Classifier(config)

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pad_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    pad_mask[:, -5:] = False

    output = model(input_ids, pad_mask=pad_mask)
    assert isinstance(output, Output)

    assert output.logits.shape == (batch_size, config.num_classes)
    assert output.script_logits.shape == (batch_size, config.num_scripts)
    assert output.filter_logits.shape == (batch_size,)
    assert output.embeddings.shape == (batch_size, config.embed_dim)
    assert output.projections.shape == (batch_size, config.projection_dim)
