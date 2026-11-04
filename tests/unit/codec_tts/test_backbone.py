"""Verify native codec embeddings and causal hidden states."""

import torch

from agbalu.codec_tts.backbone import Generator, map_weight
from agbalu.codec_tts.configuration import Geometry
from agbalu.codec_tts.layout import CODEBOOKS, VOCABULARY


def test_future_tokens_do_not_change_previous_states() -> None:
    torch.manual_seed(10)
    model = Generator(
        Geometry(
            hidden_size=16,
            intermediate_size=24,
            layers=1,
            heads=2,
            kv_heads=1,
            head_dim=8,
            text_vocabulary=32,
        )
    )
    model.train(False)
    text = torch.tensor([[1, 2, 3]])
    codes = torch.zeros((1, 3, CODEBOOKS), dtype=torch.long)
    audio = torch.tensor([[False, True, True]])
    valid = torch.ones_like(text)
    before = model.hidden(text, codes, audio, valid)
    codes[:, -1] = 3
    after = model.hidden(text, codes, audio, valid)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert not torch.equal(before[:, -1], after[:, -1])
    assert model.classify(after).shape == (1, 3, CODEBOOKS, VOCABULARY)
    manual = sum(model.audio_embedding.weight[stream * VOCABULARY] for stream in range(CODEBOOKS))
    torch.testing.assert_close(model.embed_audio(torch.zeros(CODEBOOKS, dtype=torch.long)), manual)


def test_weight_mapping_distinguishes_codec_and_generator() -> None:
    assert map_weight("tied.embedding.text_embedding.weight") == "body.embed_tokens.weight"
    assert (
        map_weight("body.layers.0.self_attn.q_proj.weight")
        == "body.layers.0.self_attn.q_proj.weight"
    )
    assert map_weight("tied.embedding.modality_embeddings.0.model.encoder.weight") is None
