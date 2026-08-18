"""The standalone Belaid-31M against the module its weights were trained under.

The encoder body is written out again in `agbalu.hub.belaid` rather than imported, because a
published repository cannot import a sibling one. That duplication is only safe while the two
produce the same tensors from the same weights, which is what this file asserts.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from agbalu.hub.belaid.configuration_belaid import BelaidConfig
from agbalu.hub.belaid.modeling_belaid import (
    BelaidForTokenClassification,
    split_words,
)
from agbalu.model.config import ModelConfig
from agbalu.model.modeling import Encoder
from agbalu.punctuation.labels import CASE, PUNCTUATION
from agbalu.punctuation.labels import split_words as trained_split_words
from agbalu.punctuation.model import Restorer

SMALL = ModelConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_hidden_layers=2,
    max_position_embeddings=32,
    position_bucket_size=8,
)

DERIVED = "position_indices"


def _standalone() -> BelaidForTokenClassification:
    config = BelaidConfig(
        **asdict(SMALL),
        punctuation_labels=list(PUNCTUATION),
        case_labels=list(CASE),
    )
    return BelaidForTokenClassification(config)


@pytest.fixture
def pair() -> tuple[Restorer, BelaidForTokenClassification]:
    torch.manual_seed(0)
    trained = Restorer(Encoder(SMALL)).eval()
    standalone = _standalone().eval()
    incompatible = standalone.load_state_dict(trained.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert all(key.endswith(DERIVED) for key in incompatible.unexpected_keys)
    return trained, standalone


def test_both_heads_are_reproduced_tensor_for_tensor(
    pair: tuple[Restorer, BelaidForTokenClassification],
) -> None:
    trained, standalone = pair
    input_ids = torch.randint(4, SMALL.vocab_size, (2, 9))
    attention = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.inference_mode():
        expected = trained(input_ids, attention)
        produced = standalone(input_ids, attention)

    assert torch.equal(expected.punctuation, produced.punctuation_logits)
    assert torch.equal(expected.case, produced.case_logits)


def test_padding_is_honoured_the_same_way(
    pair: tuple[Restorer, BelaidForTokenClassification],
) -> None:
    """An integer mask from a tokenizer must mask exactly as the trained boolean one does."""
    trained, standalone = pair
    input_ids = torch.randint(4, SMALL.vocab_size, (2, 9))
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    attention[1, 6:] = False

    with torch.inference_mode():
        expected = trained(input_ids, attention)
        produced = standalone(input_ids, attention.long())

    assert torch.equal(expected.punctuation, produced.punctuation_logits)


def test_the_word_definition_is_the_same_one(
    pair: tuple[Restorer, BelaidForTokenClassification],
) -> None:
    """The labels are per word, so a different split silently misaligns every prediction."""
    del pair
    for text in (
        "azul fell-awen amek i tellam",
        "ur\u200b tt-id-iṣaḥ ad terr awal",
        "yenna-yas: «d acu?»",
        "  ",
        "tameṭṭut n wexxam-nni ɣur-s 12 warrac",
    ):
        assert split_words(text) == trained_split_words(text)


def test_the_head_widths_come_from_the_config(
    pair: tuple[Restorer, BelaidForTokenClassification],
) -> None:
    _, standalone = pair
    assert standalone.punctuation.out.out_features == len(PUNCTUATION)
    assert standalone.case.out.out_features == len(CASE)


def test_the_classifier_decoder_is_tied_to_the_input_embedding() -> None:
    """`named_parameters` deduplicates by storage, so the alias is absent exactly when tied.

    The export drops the duplicate for the same reason — `save_file` refuses shared storage —
    so an untied copy would publish weights the module then reports as missing.
    """
    names = set(dict(_standalone().named_parameters()))

    assert "encoder.embedding.word_embedding.weight" in names
    assert "encoder.classifier.decoder.weight" not in names
