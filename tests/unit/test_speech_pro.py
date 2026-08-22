"""Unit tests for Fadhma-Pro components."""

from __future__ import annotations

import torch

from agbalu.speech.augment_speech_pro import SpeechAugmentationPro
from agbalu.speech.corpus_pro import compute_frames
from agbalu.speech.decoder_pro import TriLevelDecoderPro
from agbalu.speech.vocabulary_pro import PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER, VocabularyPro


def test_vocabulary_pro_tokens() -> None:
    """Test 40-token canonical inventory."""
    vocab = VocabularyPro()
    tokens = vocab.get_tokens()
    assert len(tokens) == 40
    assert tokens[0] == PAD_TOKEN
    assert tokens[1] == UNK_TOKEN
    assert tokens[2] == WORD_DELIMITER
    assert tokens[3] == "-"
    assert len(tokens[4:]) == 36
    assert "ɣ" in tokens
    assert "ɛ" in tokens
    assert "ḍ" in tokens
    assert "ṭ" in tokens
    assert "ẓ" in tokens
    assert "ṛ" in tokens
    assert "č" in tokens


def test_compute_frames() -> None:
    """Test duration to convolutional frame sequence mapping."""
    # 1000 ms audio = 16000 samples -> ~50 frames
    n_frames = compute_frames(1000)
    assert 48 <= n_frames <= 52

    # 400 ms audio floor
    min_frames = compute_frames(400)
    assert min_frames >= 18


def test_speech_augmentation_pro() -> None:
    """Test online speed perturbation and noise injection."""
    augmenter = SpeechAugmentationPro(speed_perturbation=True, spec_augment=True)
    audio = torch.randn(1, 16000)
    augmented = augmenter(audio)
    assert isinstance(augmented, torch.Tensor)
    assert augmented.ndim in (1, 2)
    assert augmented.shape[-1] > 0


def test_trilevel_decoder_greedy() -> None:
    """Test greedy CTC argmax collapse."""
    vocab = VocabularyPro()
    decoder = TriLevelDecoderPro(vocab=vocab)

    tok_map = vocab.token_to_id()
    logits = torch.zeros(10, 40)
    logits[0, tok_map["a"]] = 10.0
    logits[1, tok_map["z"]] = 10.0
    logits[2, tok_map["u"]] = 10.0
    logits[3, tok_map["l"]] = 10.0
    logits[4, tok_map["[PAD]"]] = 10.0
    logits[5, tok_map["|"]] = 10.0
    logits[6, tok_map["d"]] = 10.0
    logits[7, tok_map["[PAD]"]] = 10.0

    hyp = decoder.decode_greedy(logits)
    assert "azul" in hyp
