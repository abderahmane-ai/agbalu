"""The standalone Feraoun-36M against the module its weights were trained under.

Equivalence is the whole point of the package: the published repository ships this code and
not `agbalu.ocr`, so anything the two disagree about is a defect a downloader sees and this
repository never does. The preprocessing is included, because for an OCR model it is half
the contract — a canvas built at the wrong scale still gives logits of the right shape.
"""

from __future__ import annotations

import random
from dataclasses import asdict

import pytest
import torch
from PIL import Image

from agbalu.hub.feraoun.configuration_feraoun import FeraounConfig
from agbalu.hub.feraoun.modeling_feraoun import (
    FeraounForVision2Seq,
    prepare_line_image,
    segment_page_into_lines,
)
from agbalu.ocr.config import ModelConfig
from agbalu.ocr.infer import prepare_line_image as trained_prepare
from agbalu.ocr.infer import segment_page_into_lines as trained_segment
from agbalu.ocr.models import VisionEncoderDecoder
from agbalu.ocr.synthetic import render_text_line
from agbalu.ocr.vocabulary import BOS_ID, EOS_ID, PAD_ID, UNK_ID, decode

SMALL = ModelConfig(
    vocab_size=32,
    hidden_size=32,
    intermediate_size=64,
    num_decoder_layers=2,
    num_decoder_heads=2,
    max_length=64,
)

PROBE = "Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka."


@pytest.fixture
def pair() -> tuple[VisionEncoderDecoder, FeraounForVision2Seq]:
    """The same weights in both implementations, in eval mode."""
    torch.manual_seed(0)
    trained = VisionEncoderDecoder(SMALL, use_pretrained_encoder=False).eval()
    standalone = FeraounForVision2Seq(FeraounConfig(**asdict(SMALL))).eval()
    incompatible = standalone.load_state_dict(trained.state_dict(), strict=True)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys == []
    return trained, standalone


@pytest.fixture
def canvases() -> torch.Tensor:
    lines = [PROBE, "ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ."]
    images = [render_text_line(line, augment=False, rng=random.Random(3)) for line in lines]
    return torch.stack([trained_prepare(image) for image in images])


def test_the_two_implementations_declare_the_same_state_dict() -> None:
    """Key for key, because `from_pretrained` matches on names: one renamed attribute
    publishes a repository whose weights load into nothing."""
    trained = VisionEncoderDecoder(ModelConfig(), use_pretrained_encoder=False)
    standalone = FeraounForVision2Seq(FeraounConfig())
    assert set(standalone.state_dict()) == set(trained.state_dict())


def test_the_release_parameter_count_is_the_published_one() -> None:
    standalone = FeraounForVision2Seq(FeraounConfig())
    assert sum(p.numel() for p in standalone.parameters()) == 36_291_840


def test_preprocessing_is_identical_to_the_training_pipeline() -> None:
    image = render_text_line(PROBE, augment=False, rng=random.Random(3))
    assert torch.equal(prepare_line_image(image), trained_prepare(image))


def test_page_segmentation_is_identical_to_the_training_pipeline() -> None:
    page = Image.new("RGB", (400, 300), color="white")
    for top in (40, 140, 240):
        page.paste(Image.new("RGB", (300, 30), color="black"), (20, top))
    assert [image.size for image in segment_page_into_lines(page)] == [
        image.size for image in trained_segment(page)
    ]


def test_the_logits_are_identical(
    pair: tuple[VisionEncoderDecoder, FeraounForVision2Seq], canvases: torch.Tensor
) -> None:
    trained, standalone = pair
    ids = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 10]])
    with torch.inference_mode():
        assert torch.equal(
            standalone(pixel_values=canvases, decoder_input_ids=ids).logits,
            trained(pixel_values=canvases, decoder_input_ids=ids).logits,
        )


def test_the_loss_is_identical(
    pair: tuple[VisionEncoderDecoder, FeraounForVision2Seq], canvases: torch.Tensor
) -> None:
    trained, standalone = pair
    labels = torch.tensor([[5, 6, 2, -100], [8, 9, 10, 2]])
    with torch.inference_mode():
        standalone_loss = standalone(pixel_values=canvases, labels=labels).loss
        trained_loss = trained(pixel_values=canvases, labels=labels).loss
    assert standalone_loss is not None
    assert trained_loss is not None
    assert torch.equal(standalone_loss, trained_loss)


def test_greedy_decoding_produces_the_same_text(
    pair: tuple[VisionEncoderDecoder, FeraounForVision2Seq], canvases: torch.Tensor
) -> None:
    """Free-running, which is what a caller gets: the two loops agree step for step, not
    only on the first token."""
    trained, standalone = pair
    special = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
    with torch.inference_mode():
        produced = [
            decode([token for token in row.tolist() if token not in special])
            for row in standalone.generate_ids(canvases)
        ]
        expected = trained.generate(canvases)
    assert produced == expected


def test_a_decode_stops_at_the_end_symbol(
    pair: tuple[VisionEncoderDecoder, FeraounForVision2Seq], canvases: torch.Tensor
) -> None:
    _, standalone = pair
    with torch.inference_mode():
        produced = standalone.generate_ids(canvases, max_length=8)
    assert produced.shape[1] <= 9
    assert (produced[:, 0] == standalone.config.decoder_start_token_id).all()
