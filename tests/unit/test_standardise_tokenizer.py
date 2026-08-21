"""The character table, and the ids a published checkpoint is pinned to."""

from __future__ import annotations

from pathlib import Path

from agbalu.standardise.tokenizer import Tokenizer


def test_tokenizer_build_and_vocab_size() -> None:
    tokenizer = Tokenizer.build()
    assert tokenizer.vocab_size == 128
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3


def test_tokenizer_round_trip() -> None:
    tokenizer = Tokenizer.build()
    sample = "Azul fell-awen, a wid iḥemmlen tamaziɣt! Ǧerǧer d adrar nneɣ."
    encoded = tokenizer.encode(sample, add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id

    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == sample


def test_tokenizer_arabizi_and_diacritics() -> None:
    tokenizer = Tokenizer.build()
    sample = "tamazi3t 7bib-iw tch tch dj 5edmegh"
    encoded = tokenizer.encode(sample, add_bos=False, add_eos=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == sample


def test_tokenizer_save_load(tmp_path: Path) -> None:
    tokenizer = Tokenizer.build()
    save_file = tmp_path / "tokenizer.json"
    tokenizer.save(save_file)
    loaded = Tokenizer.load(save_file)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.char_to_id == tokenizer.char_to_id
