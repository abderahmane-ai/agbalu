"""Unit tests for ConformerSpeechModel architecture and dual-head loss."""

from __future__ import annotations

import torch
from transformers import Wav2Vec2BertConfig

from agbalu.speech.model import ConformerSpeechModel


def _make_tiny_model(vocab_size: int = 32) -> ConformerSpeechModel:
    config = Wav2Vec2BertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        feat_proj_dropout=0.0,
    )
    config.feature_projection_input_dim = 160
    return ConformerSpeechModel(config, vocab_size=vocab_size)


class TestConformerSpeechModel:
    def test_instantiation_and_freezing(self) -> None:
        model = _make_tiny_model(vocab_size=32)
        assert model.vocab_size == 32

        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        for p in model.lm_head.parameters():
            assert p.requires_grad

        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_forward_pass_shapes(self) -> None:
        model = _make_tiny_model(vocab_size=40)
        batch_size = 2
        time_steps = 25
        feat_dim = 160

        inputs = torch.randn(batch_size, time_steps, feat_dim)
        output = model(input_features=inputs)

        assert output.logits.shape == (batch_size, time_steps, 40)
        assert output.presence_logits.shape == (batch_size, time_steps)
        assert output.loss is None

    def test_forward_pass_with_loss(self) -> None:
        model = _make_tiny_model(vocab_size=40)
        batch_size = 2
        time_steps = 30
        feat_dim = 160

        inputs = torch.randn(batch_size, time_steps, feat_dim)
        labels = torch.randint(low=1, high=39, size=(batch_size, 5))
        presence_labels = torch.ones(batch_size, time_steps)

        output = model(
            input_features=inputs,
            labels=labels,
            presence_labels=presence_labels,
        )

        assert output.loss is not None
        assert output.ctc_loss is not None
        assert output.presence_loss is not None
        assert torch.isfinite(output.loss)

        # Verify gradient flow
        output.loss.backward()
        assert model.lm_head.weight.grad is not None
        assert model.presence_head.weight.grad is not None
