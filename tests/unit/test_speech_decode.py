"""Unit tests for standalone decoding and silence blank masking."""

from __future__ import annotations

import torch
from transformers import Wav2Vec2BertConfig

from agbalu.speech.decode import SpeechDecoder
from agbalu.speech.evaluate import evaluate_silence_challenge
from agbalu.speech.model import ConformerSpeechModel
from agbalu.speech.vocabulary import PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER


def _make_dummy_decoder() -> tuple[SpeechDecoder, dict[str, int]]:
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        WORD_DELIMITER: 2,
        "a": 3,
        "z": 4,
        "u": 5,
        "l": 6,
    }
    config = Wav2Vec2BertConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
    )
    config.feature_projection_input_dim = 160
    model = ConformerSpeechModel(config, vocab_size=len(vocab))
    decoder = SpeechDecoder(model, vocab, speech_presence_threshold=0.5, device_str="cpu")
    return decoder, vocab


class TestSpeechDecoder:
    def test_silence_masking_forces_blank_token(self) -> None:
        decoder, vocab = _make_dummy_decoder()
        # Shape: batch=1, time=4, vocab=7
        logits = torch.zeros(1, 4, len(vocab))
        # Presence logits: first 2 frames speech (positive), last 2 frames non-speech (negative)
        presence_logits = torch.tensor([[5.0, 5.0, -5.0, -5.0]])

        masked = decoder.mask_silence_frames(logits, presence_logits)

        # First 2 frames should remain unmasked
        assert torch.allclose(masked[0, :2], logits[0, :2])
        # Last 2 frames: blank token (id 0) should be boosted, others suppressed
        assert masked[0, 2, vocab[PAD_TOKEN]].item() == 100.0
        assert masked[0, 2, vocab["a"]].item() < -1000.0
        assert masked[0, 3, vocab[PAD_TOKEN]].item() == 100.0

    def test_pure_digital_silence_yields_empty_string(self) -> None:
        decoder, _ = _make_dummy_decoder()
        # Force presence head bias to strongly negative to simulate silence detector
        decoder.model.presence_head.bias.data.fill_(-50.0)

        silence = torch.zeros(16000)
        result = decoder.decode_audio_tensor(silence, use_lm=False)
        assert result == ""

    def test_silence_challenge_passes_on_silence_detector(self) -> None:
        decoder, _ = _make_dummy_decoder()
        decoder.model.presence_head.bias.data.fill_(-50.0)

        challenge = evaluate_silence_challenge(decoder, durations=[0.5, 1.0])
        assert challenge.insertions == 0
        assert challenge.passed is True
