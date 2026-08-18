"""The standalone Masinissa-31M against the module its weights were trained under."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from agbalu.hub.masinissa.configuration_masinissa import MasinissaConfig
from agbalu.hub.masinissa.modeling_masinissa import (
    IGNORE_INDEX,
    Attention,
    MasinissaForMaskedLM,
    MasinissaModel,
)
from agbalu.model.config import ModelConfig
from agbalu.model.modeling import Encoder

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


def _standalone() -> MasinissaForMaskedLM:
    return MasinissaForMaskedLM(MasinissaConfig(**asdict(SMALL)))


@pytest.fixture
def pair() -> tuple[Encoder, MasinissaForMaskedLM]:
    torch.manual_seed(0)
    trained = Encoder(SMALL).eval()
    standalone = _standalone().eval()
    incompatible = standalone.load_state_dict(trained.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert all(key.endswith(DERIVED) for key in incompatible.unexpected_keys)
    return trained, standalone


def test_the_two_implementations_have_the_same_learned_tensors() -> None:
    """Only the derived index tables differ, and they differ on purpose: twelve identical
    512x512 int64 lookups would be 25.2 MB of a 124.5 MB download."""
    trained = set(Encoder(SMALL).state_dict())
    standalone = set(_standalone().state_dict())
    assert standalone == {key for key in trained if not key.endswith(DERIVED)}


def test_the_contextualised_representations_are_identical(
    pair: tuple[Encoder, MasinissaForMaskedLM],
) -> None:
    trained, standalone = pair
    ids = torch.randint(5, SMALL.vocab_size, (3, 9))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        expected = trained.contextualise(ids, mask)
        produced = MasinissaModel(MasinissaConfig(**asdict(SMALL)))
        produced.load_state_dict(
            {k: v for k, v in standalone.state_dict().items() if not k.startswith("classifier.")}
        )
        produced.eval()
        assert torch.equal(expected, produced(ids, attention_mask=mask.long()).last_hidden_state)


def test_the_masked_token_logits_are_identical(
    pair: tuple[Encoder, MasinissaForMaskedLM],
) -> None:
    trained, standalone = pair
    ids = torch.randint(5, SMALL.vocab_size, (2, 7))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        expected = trained.classifier(trained.contextualise(ids, mask))
        produced = standalone(input_ids=ids, attention_mask=mask.long()).logits
    assert torch.equal(expected, produced)


def test_an_integer_attention_mask_masks_padding_rather_than_inverting_its_bits(
    pair: tuple[Encoder, MasinissaForMaskedLM],
) -> None:
    """A tokenizer hands back 1/0 integers. `~` on those is a bitwise complement giving -2
    and -1, both truthy, so every position would read as padding and the softmax would see
    a row of `-inf`."""
    trained, standalone = pair
    ids = torch.randint(5, SMALL.vocab_size, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        expected = trained.classifier(trained.contextualise(ids, mask.bool()))
        produced = standalone(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(expected, produced)
    assert bool(torch.isfinite(produced).all())


def test_the_loss_ignores_unmasked_positions(pair: tuple[Encoder, MasinissaForMaskedLM]) -> None:
    _, standalone = pair
    ids = torch.randint(5, SMALL.vocab_size, (2, 6))
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[0, 2] = 7
    with torch.inference_mode():
        loss = standalone(input_ids=ids, labels=labels).loss
    assert loss is not None
    assert torch.isfinite(loss)


def test_the_position_table_survives_an_empty_initialisation() -> None:
    """As a non-persistent buffer this comes back from `from_pretrained` as uninitialised
    memory and indexes out of range — measured, not hypothetical."""
    standalone = _standalone()
    attention = standalone.transformer.attention_layers[0]
    assert isinstance(attention, Attention)
    assert DERIVED not in dict(attention.named_buffers())
    indices = attention.position_indices(SMALL.max_position_embeddings, torch.device("cpu"))
    assert int(indices.min()) >= 0
    assert int(indices.max()) <= 2 * SMALL.position_bucket_size - 2


def test_the_base_model_does_not_report_the_classifier_as_an_unexpected_key() -> None:
    """`AutoModel` on a masked-LM checkpoint is a supported call, and a warning naming
    eight tensors reads as a broken download."""
    assert MasinissaModel._keys_to_ignore_on_load_unexpected == [r"^classifier\."]
