"""The encoder-decoder's shapes, its loss, and the interpolation a line canvas goes through."""

from __future__ import annotations

import torch

from agbalu.ocr.config import ModelConfig
from agbalu.ocr.models import VisionEncoderDecoder
from agbalu.ocr.synthetic import TARGET_HEIGHT, TARGET_WIDTH
from agbalu.ocr.vocabulary import BOS_ID, EOS_ID, VOCAB_SIZE


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=64,
        num_decoder_layers=2,
        num_decoder_heads=2,
        intermediate_size=128,
    )


def test_model_initialization_and_parameter_shapes() -> None:
    config = _tiny_config()
    model = VisionEncoderDecoder(config=config, use_pretrained_encoder=False)
    assert isinstance(model, torch.nn.Module)

    params = sum(p.numel() for p in model.parameters())
    assert params > 0


def test_forward_pass_with_labels_computes_finite_loss() -> None:
    config = _tiny_config()
    model = VisionEncoderDecoder(config=config, use_pretrained_encoder=False)

    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, TARGET_HEIGHT, TARGET_WIDTH)
    labels = torch.tensor(
        [
            [BOS_ID, 10, 25, 30, EOS_ID, -100],
            [BOS_ID, 12, 18, EOS_ID, -100, -100],
        ],
        dtype=torch.long,
    )

    output = model(pixel_values=pixel_values, labels=labels)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (batch_size, labels.size(1), VOCAB_SIZE)
    assert output.encoder_hidden_states.size(0) == batch_size


def test_greedy_generation_returns_strings() -> None:
    config = _tiny_config()
    model = VisionEncoderDecoder(config=config, use_pretrained_encoder=False)

    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, TARGET_HEIGHT, TARGET_WIDTH)

    outputs = model.generate(pixel_values=pixel_values, max_length=10)
    assert len(outputs) == batch_size
    assert all(isinstance(s, str) for s in outputs)


def test_extract_features_with_image_size_interpolation() -> None:
    config = _tiny_config()
    model = VisionEncoderDecoder(config=config, use_pretrained_encoder=False)

    pixel_values = torch.randn(1, 3, 64, 512)
    features = model.extract_features(pixel_values)
    assert features.ndim == 3
    assert features.size(0) == 1
