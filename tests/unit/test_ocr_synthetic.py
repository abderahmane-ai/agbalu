"""Rendering a line, and the seeds that decide whether it renders the same way twice."""

from __future__ import annotations

import torch
from PIL import Image

from agbalu.ocr.augment import augment_line_image
from agbalu.ocr.dataset import SyntheticDataset, chunk_text_into_lines
from agbalu.ocr.synthetic import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    image_to_tensor,
    render_text_line,
)


def test_render_text_line_produces_exact_target_dimensions() -> None:
    text = "Taqbaylit d tutlayt tayemmat nneɣ."
    img = render_text_line(text, augment=False)
    assert isinstance(img, Image.Image)
    assert img.size == (TARGET_WIDTH, TARGET_HEIGHT)


def test_render_text_line_with_augmentations() -> None:
    text = "« Ḥemleɣ ad ɣreɣ idlisen n Dda Lmulud Mammeri. »"
    img = render_text_line(text, augment=True)
    assert isinstance(img, Image.Image)
    assert img.size == (TARGET_WIDTH, TARGET_HEIGHT)


def test_image_to_tensor_shape_and_normalization() -> None:
    text = "Azul fell-awen!"
    img = render_text_line(text, augment=False)
    tensor = image_to_tensor(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, TARGET_HEIGHT, TARGET_WIDTH)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.05
    assert tensor.max() <= 1.05


def test_augment_line_image_preserves_rgb_and_dimensions() -> None:
    base_img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    aug_img = augment_line_image(base_img)
    assert isinstance(aug_img, Image.Image)
    assert aug_img.mode == "RGB"
    assert aug_img.size == (200, 50)


def test_an_unaugmented_render_is_the_same_image_every_time_it_is_scored() -> None:
    """A held-out line has to render identically at every evaluation, or the CER curve
    moves with the renderer's font and offset draw and `best.pt` is chosen by a coin toss.
    """
    dataset = SyntheticDataset(["Ḥemleɣ ad ɣreɣ idlisen n Dda Lmulud Mammeri."], augment=False)
    first, _, _ = dataset[0]
    second, _, _ = dataset[0]
    assert torch.equal(first, second)


def test_two_held_out_lines_do_not_render_identically() -> None:
    """The seed is per index, not per dataset: a single seed for every row would draw the
    same font and offset for all of them and make the holdout one typeface wide."""
    dataset = SyntheticDataset(["azul a yemma", "azul a baba"], augment=False)
    assert not torch.equal(dataset[0][0], dataset[1][0])


def test_an_augmented_render_varies_between_draws() -> None:
    dataset = SyntheticDataset(["Taqbaylit d tutlayt tayemmat nneɣ."] * 2, augment=True)
    assert not torch.equal(dataset[0][0], dataset[1][0])


def test_line_chunking_is_reproducible_across_calls() -> None:
    """The chunk boundaries decide which lines land in the holdout. On the module generator
    the same corpus splits differently on every call, so a resumed run holds out a
    different set from the one it is continuing."""
    sentences = ["Azul fell-awen amek tellam, taqbaylit d tutlayt tayemmat nneɣ i lebda."] * 5
    assert chunk_text_into_lines(sentences) == chunk_text_into_lines(sentences)
    assert chunk_text_into_lines(sentences, seed=1) != chunk_text_into_lines(sentences, seed=2)
