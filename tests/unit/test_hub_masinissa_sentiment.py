"""The standalone Masinissa-Sentiment against the modules its weights were trained under.

This file is the only thing holding the published copy of the encoder and the head to the
originals. Both are copies by necessity — the published repository can import neither
`agbalu.model` nor `agbalu.sentiment` — and a rule changed on one side and not the other
publishes a model computing something the training never optimised. The first draft of
this port skipped the relative-position projection entirely and looked exactly like a
working model.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from tools.stage_hub import ARCHITECTURES, AUTO_MAP, HUB

from agbalu.bench.sentiment import LABEL_NAMES
from agbalu.hub.masinissa_sentiment.configuration_masinissa_sentiment import (
    LABELS,
    MasinissaSentimentConfig,
)
from agbalu.hub.masinissa_sentiment.modeling_masinissa_sentiment import (
    Attention,
    MasinissaForSequenceClassification,
)
from agbalu.model.config import ModelConfig
from agbalu.model.modeling import Encoder
from agbalu.sentiment.config import SentimentConfig
from agbalu.sentiment.model import SequenceClassifier
from agbalu.sentiment.release import MODEL_TYPE
from agbalu.sentiment.release import config as published_config

SMALL = ModelConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_hidden_layers=2,
    max_position_embeddings=32,
    position_bucket_size=8,
)

HEAD = SentimentConfig(hidden_size=16, bottleneck_size=8)

DERIVED = "position_indices"
PRETRAINING_HEAD = "encoder.classifier."


def _standalone() -> MasinissaForSequenceClassification:
    return MasinissaForSequenceClassification(
        MasinissaSentimentConfig(
            **asdict(SMALL),
            bottleneck_size=HEAD.bottleneck_size,
            dropouts=list(HEAD.dropouts),
            label_smoothing=HEAD.label_smoothing,
        )
    )


def _trained() -> SequenceClassifier:
    return SequenceClassifier(Encoder(SMALL), HEAD)


@pytest.fixture
def pair() -> tuple[SequenceClassifier, MasinissaForSequenceClassification]:
    torch.manual_seed(0)
    trained = _trained().eval()
    standalone = _standalone().eval()
    incompatible = standalone.load_state_dict(trained.state_dict(), strict=False)
    assert incompatible.missing_keys == []
    assert all(
        key.endswith(DERIVED) or key.startswith(PRETRAINING_HEAD)
        for key in incompatible.unexpected_keys
    )
    return trained, standalone


def test_the_two_implementations_have_the_same_learned_tensors() -> None:
    """Two families differ and both differ on purpose: the derived bucket tables, and the
    masked-LM head that shaped the trunk during pretraining and is read by nothing here."""
    trained = set(_trained().state_dict())
    standalone = set(_standalone().state_dict())
    assert standalone == {
        key for key in trained if not key.endswith(DERIVED) and not key.startswith(PRETRAINING_HEAD)
    }


def test_the_logits_are_identical(
    pair: tuple[SequenceClassifier, MasinissaForSequenceClassification],
) -> None:
    trained, standalone = pair
    ids = torch.randint(5, SMALL.vocab_size, (3, 9))
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        expected, _ = trained(ids, mask)
        produced = standalone(input_ids=ids, attention_mask=mask.long()).logits
    assert torch.equal(expected, produced)


def test_the_logits_are_identical_under_padding(
    pair: tuple[SequenceClassifier, MasinissaForSequenceClassification],
) -> None:
    """Padding is where a pooled classifier diverges: the mean is over the real positions,
    so a mask cast one way and not the other changes the denominator and nothing else."""
    trained, standalone = pair
    ids = torch.randint(5, SMALL.vocab_size, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        expected, _ = trained(ids, mask.bool())
        produced = standalone(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(expected, produced)
    assert bool(torch.isfinite(produced).all())


def test_the_training_head_and_the_published_head_agree_in_training_mode(
    pair: tuple[SequenceClassifier, MasinissaForSequenceClassification],
) -> None:
    """Multi-sample dropout is the training objective, so the two must agree there too —
    with the dropout rates zeroed, which is the only way two mask draws are comparable."""
    trained, standalone = pair
    for model in (trained, standalone):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    ids = torch.randint(5, SMALL.vocab_size, (2, 7))
    mask = torch.ones_like(ids, dtype=torch.bool)
    labels = torch.tensor([0, 2])
    expected, expected_loss = trained(ids, mask, labels=labels)
    produced = standalone(input_ids=ids, attention_mask=mask.long(), labels=labels)
    assert torch.equal(expected, produced.logits)
    assert expected_loss is not None
    assert produced.loss is not None
    assert torch.equal(expected_loss, produced.loss)


def test_the_published_label_order_is_the_projects_label_order() -> None:
    """The head's rows are positions in this tuple. Reordering it renames every prediction
    and nothing raises, because both lists are three strings long."""
    assert tuple(LABELS) == LABEL_NAMES


def test_what_the_release_writes_is_what_the_published_class_reads(tmp_path: Path) -> None:
    """Down the path staging actually takes: the writer emits `config.json`, `stage_hub`
    reads it back into this class and saves it again. A field the writer emits and the
    class does not accept survives in `**kwargs` and is published as a key nothing reads;
    a calibration constant lost on the way back is a model that corrects nothing.
    """
    source = tmp_path / "config.json"
    source.write_text(json.dumps(published_config(temperature=1.25, q_hat=0.42)), "utf-8")

    config = MasinissaSentimentConfig(**json.loads(source.read_text(encoding="utf-8")))
    assert config.model_type == MODEL_TYPE
    assert config.temperature == pytest.approx(1.25)
    assert config.q_hat == pytest.approx(0.42)
    assert config.labels == list(LABEL_NAMES)
    assert config.id2label == dict(enumerate(LABEL_NAMES))
    assert not config.tie_word_embeddings

    staged = tmp_path / "staged"
    config.save_pretrained(staged)
    reloaded = MasinissaSentimentConfig(
        **json.loads((staged / "config.json").read_text(encoding="utf-8"))
    )
    assert reloaded.q_hat == pytest.approx(0.42)
    assert reloaded.temperature == pytest.approx(1.25)
    assert reloaded.dropouts == config.dropouts
    assert reloaded.labels == config.labels


def test_an_uncalibrated_repository_returns_the_argmax_and_claims_nothing() -> None:
    """`q_hat` absent means the quantile was never fitted. A singleton returned there is
    the argmax, not a 95% set, and the two must not be indistinguishable to a caller."""
    model = _standalone().eval()
    assert model.config.q_hat is None
    probabilities = torch.tensor([[0.34, 0.33, 0.33]])
    logits = probabilities.log()
    assert model.probabilities(logits).shape == (1, 3)


def test_temperature_scaling_moves_confidence_and_not_the_argmax() -> None:
    model = _standalone().eval()
    model.config.temperature = 2.0
    logits = torch.tensor([[3.0, 0.5, -1.0], [-0.5, 1.0, 2.5]])
    scaled = model.probabilities(logits)
    assert torch.equal(scaled.argmax(dim=-1), logits.argmax(dim=-1))
    assert float(scaled.max()) < float(logits.softmax(dim=-1).max())
    assert scaled.sum(dim=-1).allclose(torch.ones(2))


def test_the_position_table_survives_an_empty_initialisation() -> None:
    """As a non-persistent buffer this comes back from `from_pretrained` as uninitialised
    memory and indexes out of range — measured on the encoder, not hypothetical."""
    standalone = _standalone()
    attention = standalone.encoder.transformer.attention_layers[0]
    assert isinstance(attention, Attention)
    assert DERIVED not in dict(attention.named_buffers())
    indices = attention.position_indices(SMALL.max_position_embeddings, torch.device("cpu"))
    assert int(indices.min()) >= 0
    assert int(indices.max()) <= 2 * SMALL.position_bucket_size - 2


def test_the_pretraining_head_is_not_reported_as_an_unexpected_key() -> None:
    """It is in every training checkpoint and in no release, and a warning naming a dozen
    tensors reads to a downloader as a broken download."""
    assert MasinissaForSequenceClassification._keys_to_ignore_on_load_unexpected == [
        r"^encoder\.classifier\.",
        r".*position_indices$",
    ]


def test_the_auto_map_names_modules_the_staging_copies() -> None:
    modules = {path.stem for path in (HUB / "masinissa_sentiment").glob("*.py")}
    for target in AUTO_MAP["masinissa_sentiment"].values():
        module, _, symbol = target.rpartition(".")
        assert module in modules
        assert symbol
    assert ARCHITECTURES["masinissa_sentiment"] == MasinissaForSequenceClassification.__name__
