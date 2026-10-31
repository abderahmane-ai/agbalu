"""Verify the fixed qualitative sentence contract and reject invalid encodings."""

import pytest
import sentencepiece as spm
import torch

from agbalu.encoder import previews


def test_the_four_operator_sentences_keep_their_exact_boundaries() -> None:
    assert len(previews.SENTENCES) == len(set(previews.SENTENCES)) == 4
    assert [sentence.split()[0] for sentence in previews.SENTENCES] == [
        "Mi",
        "Tutlayt",
        "Argaz-nni",
        "Ur",
    ]
    assert [sentence.split()[-1] for sentence in previews.SENTENCES] == [
        "taddart.",
        "ass.",
        "nezzeh.",
        "gar-aneɣ.",
    ]
    assert "Ǧerjer" in previews.SENTENCES[3]
    assert "tt-id-nessukkes" in previews.SENTENCES[1]


def test_fixed_whole_group_masks_do_not_consume_the_training_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encode(text: str, processor: spm.SentencePieceProcessor) -> tuple[list[int], list[int]]:
        del text, processor
        return list(range(5, 21)), [2 * (index // 2) for index in range(16)]

    monkeypatch.setattr(previews, "encode_groups", encode)
    before = torch.get_rng_state()
    examples = previews.prepare_sentences(spm.SentencePieceProcessor(), 32)
    repeated = previews.prepare_sentences(spm.SentencePieceProcessor(), 32)
    assert torch.equal(before, torch.get_rng_state())
    for example, again in zip(examples, repeated, strict=True):
        assert torch.equal(example.selected, again.selected)
        assert example.masked[0] == 2
        assert example.masked[-1] == 3
        assert example.selected.any()
        assert torch.all(example.masked[example.selected] == 4)
        assert torch.equal(example.masked[~example.selected], example.original[~example.selected])
        for offset in range(1, 17, 2):
            assert example.selected[offset] == example.selected[offset + 1]


@pytest.mark.parametrize("ids", [[], [0], [4], [32]])
def test_invalid_preview_tokens_fail_with_sentence_identity(
    ids: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def encode(text: str, processor: spm.SentencePieceProcessor) -> tuple[list[int], list[int]]:
        del text, processor
        return ids, list(range(len(ids)))

    monkeypatch.setattr(previews, "encode_groups", encode)
    with pytest.raises(ValueError, match="preview sentence 1: invalid token IDs"):
        previews.prepare_sentences(spm.SentencePieceProcessor(), 32)


def test_preview_capacity_fails_instead_of_truncating_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encode(text: str, processor: spm.SentencePieceProcessor) -> tuple[list[int], list[int]]:
        del text, processor
        return [5] * 200, list(range(200))

    monkeypatch.setattr(previews, "encode_groups", encode)
    with pytest.raises(ValueError, match="need 808 tokens, capacity is 512"):
        previews.prepare_sentences(spm.SentencePieceProcessor(), 32)
