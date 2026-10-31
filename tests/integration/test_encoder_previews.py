"""Verify complete sentence previews with the actual Mammeri tokenizer."""

import pytest
import torch

from agbalu.encoder.configuration import Geometry, Locations
from agbalu.encoder.corpus import tokenizer
from agbalu.encoder.evaluation import preview
from agbalu.encoder.model import Encoder
from agbalu.encoder.previews import SENTENCES, prepare_sentences

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not Locations().tokenizer.is_file(), reason="Mammeri artifact unavailable"),
]


def test_real_sentences_are_complete_repeatable_and_preserve_training_state() -> None:
    locations = Locations()
    processor = tokenizer(locations.tokenizer)
    examples = prepare_sentences(processor, 16000)
    assert len(examples) == 4
    assert all(processor.decode(example.original.tolist()) == example.text for example in examples)
    model = Encoder(
        Geometry(
            hidden_size=8,
            intermediate_size=8,
            heads=1,
            layers=1,
            block_layers=1,
            dropout=0.5,
            backend="reference",
        )
    ).train()
    before = torch.get_rng_state()
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    texts, diagnostics = preview(model, None, locations, torch.device("cpu"))
    repeated, repeated_diagnostics = preview(model, None, locations, torch.device("cpu"))
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert texts == repeated
    assert diagnostics == repeated_diagnostics
    assert len(texts) == 12
    for index, sentence in enumerate(SENTENCES, 1):
        assert texts[f"sentence {index} original"] == sentence
        assert "[MASK]" in texts[f"sentence {index} masked"]
        assert len(texts[f"sentence {index} restored"]) > 90
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name])
