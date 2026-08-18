"""The standalone Juba-27M against the module its weights were trained under.

Equivalence is the whole point of the package: the published repository ships this code
and not `agbalu.tifinagh.model`, so anything the two disagree about is a defect a
downloader sees and this repository never does.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from agbalu.hub.juba.configuration_juba import JubaConfig
from agbalu.hub.juba.modeling_juba import JubaForSeq2SeqLM
from agbalu.tifinagh.config import ModelConfig
from agbalu.tifinagh.model import CharTransformer

SMALL = ModelConfig(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    max_position_embeddings=64,
)


@pytest.fixture
def pair() -> tuple[CharTransformer, JubaForSeq2SeqLM]:
    """The same weights in both implementations, in eval mode."""
    torch.manual_seed(0)
    trained = CharTransformer(SMALL).eval()
    standalone = JubaForSeq2SeqLM(JubaConfig(**asdict(SMALL))).eval()
    incompatible = standalone.load_state_dict(trained.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) <= {"lm_head.weight"}
    return trained, standalone


def test_the_two_implementations_have_the_same_state_dict_keys() -> None:
    """A renamed attribute is a `model.safetensors` that no longer loads for anybody who
    downloaded the release, and it fails as a missing-key list naming no cause."""
    trained = CharTransformer(SMALL)
    standalone = JubaForSeq2SeqLM(JubaConfig(**asdict(SMALL)))
    assert set(trained.state_dict()) == set(standalone.state_dict())


def test_the_encoder_output_is_identical(pair: tuple[CharTransformer, JubaForSeq2SeqLM]) -> None:
    trained, standalone = pair
    ids = torch.randint(4, SMALL.vocab_size, (3, 11))
    with torch.inference_mode():
        assert torch.equal(trained.encode(ids), standalone.encode(ids))


def test_the_decoder_logits_are_identical(pair: tuple[CharTransformer, JubaForSeq2SeqLM]) -> None:
    trained, standalone = pair
    source = torch.randint(4, SMALL.vocab_size, (3, 11))
    target = torch.randint(4, SMALL.vocab_size, (3, 7))
    with torch.inference_mode():
        context = trained.encode(source)
        expected = trained.decode(target, context)
        produced = standalone(input_ids=source, decoder_input_ids=target).logits
    assert torch.equal(expected, produced)


def test_the_loss_is_the_trained_objective(pair: tuple[CharTransformer, JubaForSeq2SeqLM]) -> None:
    """Label smoothing and the ignored pad index are part of the objective, so a head that
    reports an unsmoothed loss reports a number the training curve cannot be read against."""
    trained, standalone = pair
    source = torch.randint(4, SMALL.vocab_size, (2, 9))
    target = torch.randint(4, SMALL.vocab_size, (2, 6))
    labels = torch.randint(4, SMALL.vocab_size, (2, 6))
    labels[0, -1] = SMALL.pad_token_id
    with torch.inference_mode():
        expected, _ = trained.loss_with_logits(source, target, labels)
        produced = standalone(input_ids=source, decoder_input_ids=target, labels=labels).loss
    assert torch.allclose(expected.loss, produced)


def test_generate_reproduces_the_free_running_greedy_decode(
    pair: tuple[CharTransformer, JubaForSeq2SeqLM],
) -> None:
    """`generate` must walk the same path `Transliterator.greedy_batch` does. A teacher-
    forced or cache-sliced decode would silently be a different measurement."""
    _, standalone = pair
    source = torch.randint(4, SMALL.vocab_size, (1, 9))
    with torch.inference_mode():
        produced = standalone.generate(input_ids=source, max_length=12, num_beams=1)
        assert isinstance(produced, torch.Tensor)
        context = standalone.encode(source)
        tokens = torch.tensor([[SMALL.bos_token_id]])
        for _ in range(produced.shape[1] - 1):
            following = standalone.decode(tokens, context)[:, -1, :].argmax(dim=-1)
            tokens = torch.cat([tokens, following.unsqueeze(1)], dim=1)
            if int(following) == SMALL.eos_token_id:
                break
    assert produced[0].tolist()[: tokens.shape[1]] == tokens[0].tolist()


def test_the_model_declares_that_it_has_no_key_value_cache() -> None:
    """Without this `generate` allocates a `DynamicCache` sized from `num_hidden_layers`,
    which this configuration does not carry, and raises before decoding a token."""
    assert JubaConfig().use_cache is False


def test_the_rotary_frequencies_survive_an_empty_initialisation() -> None:
    """`from_pretrained` allocates buffers empty and fills them from the checkpoint. These
    frequencies are derived and absent from it, so as a registered buffer they come back as
    uninitialised memory that rotates every query by a garbage angle and raises nothing."""
    standalone = JubaForSeq2SeqLM(JubaConfig(**asdict(SMALL)))
    assert "rope.inv_freq" not in dict(standalone.named_buffers())
    frequencies = standalone.rope.inv_freq(torch.device("cpu"))
    assert bool((frequencies > 0).all())
    assert frequencies.shape == (SMALL.hidden_size // SMALL.num_attention_heads // 2,)


def test_padding_is_attended_to_exactly_as_the_published_model_does() -> None:
    """`Transliterator._encode_batch` passes no mask, so the released weights were scored
    attending over padding. Honouring `attention_mask` here would return different logits
    from the model's own evaluation."""
    torch.manual_seed(1)
    standalone = JubaForSeq2SeqLM(JubaConfig(**asdict(SMALL))).eval()
    source = torch.randint(4, SMALL.vocab_size, (1, 9))
    target = torch.randint(4, SMALL.vocab_size, (1, 5))
    with torch.inference_mode():
        unmasked = standalone(input_ids=source, decoder_input_ids=target).logits
        masked = standalone(
            input_ids=source,
            attention_mask=torch.zeros_like(source),
            decoder_input_ids=target,
        ).logits
    assert torch.equal(unmasked, masked)


def test_a_call_without_a_source_or_an_encoding_is_refused() -> None:
    standalone = JubaForSeq2SeqLM(JubaConfig(**asdict(SMALL))).eval()
    with pytest.raises(ValueError, match="input_ids or encoder_outputs"):
        standalone(decoder_input_ids=torch.zeros(1, 2, dtype=torch.long))
