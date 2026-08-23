"""The standalone Dihya-5M against the module its weights were trained under.

Equivalence is the whole point of the package: the published repository ships this code
and not `agbalu.lid.model`, so anything the two disagree about is a defect a downloader
sees and this repository never does.

The byte vocabulary is checked here too. A tokenizer built through the wrong byte table
still encodes every string to ids in range, and the model then reads a different language
than the one on the page — there is no shape, no error and no warning that says so.
"""

from __future__ import annotations

import pytest
import torch
from tools.stage_hub import BYTE_VALUES, build_byte_tokenizer, byte_to_character

from agbalu.hub.dihya.configuration_dihya import CLASSES, DihyaConfig
from agbalu.hub.dihya.modeling_dihya import DihyaForSequenceClassification
from agbalu.lid.config import BYTE_OFFSET, PAD_ID, UNK_ID, ModelConfig
from agbalu.lid.dataset import encode
from agbalu.lid.model import CharClassifier

SMALL_TRAINED = ModelConfig(
    vocab_size=64,
    embed_dim=16,
    conv_kernels=(3, 5),
    conv_dim=8,
    num_layers=2,
    num_heads=2,
    intermediate_dim=32,
    projection_dim=8,
    max_length=64,
    dropout=0.0,
)

SMALL_RELEASED = DihyaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_hidden_layers=2,
    conv_kernels=[3, 5],
    conv_dim=8,
    max_position_embeddings=64,
    dropout_prob=0.0,
    classes=list(SMALL_TRAINED.classes),
)

BYTE_SIZED = DihyaConfig(
    vocab_size=BYTE_VALUES + BYTE_OFFSET,
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_hidden_layers=2,
    conv_kernels=[3, 5],
    conv_dim=8,
    max_position_embeddings=64,
    dropout_prob=0.0,
)
"""The real vocabulary. `identify` encodes text rather than taking ids, so a shrunk
embedding table indexes out of range on any byte above the cut."""

TEXTS = (
    "Azul fell-awen, amek tettilim?",
    "Mayemmi ur teccid ticemrarin nnecm?",
    "Bonjour tout le monde.",
    "",
    "ɣ",
    "﻿Azul fell-awen\u200b",
    "ⵜⴰⵇⴱⴰⵢⵍⵉⵜ",
    "a" * 512,
    "\U0001f642 azul",
    "é vs é",
    "\r\nazul\r\n",
)
"""Kabyle, Tarifit, French, empty, one multi-byte letter, BOM/NBSP/ZWSP, Tifinagh, longer
than `max_length`, a surrogate-pair emoji, NFD against NFC, and mixed line endings."""


@pytest.fixture
def pair() -> tuple[CharClassifier, DihyaForSequenceClassification]:
    """The same weights in both implementations, in eval mode."""
    torch.manual_seed(0)
    trained = CharClassifier(SMALL_TRAINED).eval()
    released = DihyaForSequenceClassification(SMALL_RELEASED).eval()
    incompatible = released.load_state_dict(trained.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert set(incompatible.unexpected_keys) == {
        "head.margins",
        "projection.0.weight",
        "projection.2.weight",
    }
    return trained, released


def test_the_release_keeps_every_key_the_trained_trunk_has(
    pair: tuple[CharClassifier, DihyaForSequenceClassification],
) -> None:
    """A renamed attribute is a `model.safetensors` that no longer loads for anybody who
    downloaded the release, and it fails as a missing-key list naming no cause."""
    trained, released = pair
    assert set(released.state_dict()) < set(trained.state_dict())
    assert set(trained.state_dict()) - set(released.state_dict()) == {
        "head.margins",
        "projection.0.weight",
        "projection.2.weight",
    }


def test_the_two_implementations_agree_on_the_logits(
    pair: tuple[CharClassifier, DihyaForSequenceClassification],
) -> None:
    trained, released = pair
    input_ids = torch.randint(BYTE_OFFSET, 64, (4, 24))
    pad_mask = torch.ones_like(input_ids, dtype=torch.bool)
    pad_mask[2, 12:] = False
    pad_mask[3, 1:] = False

    with torch.inference_mode():
        expected = trained(input_ids, pad_mask=pad_mask).logits
        actual = released(input_ids, attention_mask=pad_mask.long()).logits
    assert torch.allclose(expected, actual, atol=1e-6)


def test_a_row_that_is_entirely_padding_does_not_produce_nan(
    pair: tuple[CharClassifier, DihyaForSequenceClassification],
) -> None:
    """An empty string is a real input to a language identifier, and `-inf` on every
    position of the pooling softmax returns NaN for the whole row."""
    _, released = pair
    input_ids = torch.full((1, 8), PAD_ID, dtype=torch.long)
    with torch.inference_mode():
        logits = released(input_ids, attention_mask=torch.zeros_like(input_ids)).logits
    assert torch.isfinite(logits).all()


def test_the_prior_shift_and_quantile_travel_in_the_config() -> None:
    """Without them the released model is a different classifier from the measured one:
    the argmax minimises the training prior's error, and there is no prediction set."""
    config = DihyaConfig(prior_shift=[0.0] * 7, q_hat=0.9979)
    assert config.q_hat == pytest.approx(0.9979)
    assert len(config.prior_shift) == len(config.classes)
    assert dict(config.id2label or {}) == dict(enumerate(CLASSES))
    assert dict(config.label2id or {})["NOT_AMAZIGH"] == len(CLASSES) - 1


def test_the_byte_table_is_a_bijection_over_every_byte_value() -> None:
    table = byte_to_character()
    assert sorted(table) == list(range(BYTE_VALUES))
    assert len(set(table.values())) == BYTE_VALUES


@pytest.mark.parametrize("text", TEXTS)
def test_the_published_tokenizer_reproduces_the_training_encoder(text: str) -> None:
    """The contract the release rests on: same ids, same model, byte for byte."""
    tokenizer = build_byte_tokenizer()
    produced = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert produced == encode(text, len(text.encode("utf-8")))


def test_the_published_tokenizer_truncates_where_the_training_encoder_does() -> None:
    """Truncation is on bytes and may cut a multi-byte character in half, deliberately:
    at inference text arrives already truncated, and a tokenizer that keeps the last
    character whole would encode a shape the model was never trained on."""
    tokenizer = build_byte_tokenizer()
    text = "ɣ" * 200
    produced = tokenizer(text, add_special_tokens=False, truncation=True, max_length=256)
    assert produced["input_ids"] == encode(text, 256)
    assert len(produced["input_ids"]) == 256


def test_the_tokenizer_holds_the_reserved_ids_and_nothing_more() -> None:
    tokenizer = build_byte_tokenizer()
    assert len(tokenizer) == BYTE_VALUES + BYTE_OFFSET
    assert tokenizer.convert_tokens_to_ids("[PAD]") == PAD_ID
    assert tokenizer.convert_tokens_to_ids("[UNK]") == UNK_ID


@pytest.mark.parametrize("text", [t for t in TEXTS if t and len(t.encode("utf-8")) <= 256])
def test_the_tokenizer_decodes_back_to_the_text(text: str) -> None:
    tokenizer = build_byte_tokenizer()
    assert tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"]) == text


def test_identify_returns_a_prediction_set_only_when_a_quantile_was_fitted() -> None:
    """An uncalibrated repository must report the argmax, never a singleton dressed up as
    a conformal set — the guarantee is the quantile, not the shape of the answer."""
    torch.manual_seed(0)
    uncalibrated = DihyaForSequenceClassification(BYTE_SIZED).eval()
    row = uncalibrated.identify(["azul fell-awen"])[0]
    assert row["prediction_set"] == (row["language"],)

    calibrated = DihyaForSequenceClassification(
        DihyaConfig(**{**BYTE_SIZED.to_dict(), "q_hat": 1.0})
    ).eval()
    covered = calibrated.identify(["azul fell-awen"])[0]
    assert set(covered["prediction_set"]) == set(BYTE_SIZED.classes)


def test_identify_handles_an_empty_batch_and_a_bare_string() -> None:
    torch.manual_seed(0)
    model = DihyaForSequenceClassification(BYTE_SIZED).eval()
    assert model.identify([]) == []
    assert len(model.identify("azul")) == 1
    assert len(model.identify(["azul", ""])) == 2
