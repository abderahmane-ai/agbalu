"""The 171-symbol table: what it holds, the letter it does not, and the width it fixes."""

from __future__ import annotations

import pytest

from agbalu.ocr.vocabulary import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    VOCABULARY,
    decode,
    encode,
    get_vocab_size,
)


def test_vocabulary_is_non_empty_and_deterministic() -> None:
    assert len(VOCABULARY) == VOCAB_SIZE
    assert VOCAB_SIZE > 100
    assert get_vocab_size() == VOCAB_SIZE
    assert VOCABULARY[PAD_ID] == "<pad>"
    assert VOCABULARY[BOS_ID] == "<s>"
    assert VOCABULARY[EOS_ID] == "</s>"
    assert VOCABULARY[UNK_ID] == "<unk>"


@pytest.mark.parametrize(
    "text",
    [
        "Azul fell-awen, amek tellam?",
        "Taqbaylit d tutlayt tayemmat nneɣ.",
        "Yal amdan yezmer ad yelmed tamaziɣt deg uɣerbaz.",
        "Ḥemleɣ ad ɣreɣ idlisen n Dda Lmulud Mammeri d Belaïd At Ali.",
        "Axxam n taddart yeččur d lferḥ d liser.",
        "1980 d aseggas n Tafsut Imaziɣen.",
        "« Awal yettazzal, tira teqqim. »",
        "ⴰⵣⵓⵍ ⴼⵍⵍⴰⵡⴻⵏ",
    ],
)
def test_character_encoding_decoding_round_trip(text: str) -> None:
    ids = encode(text, add_special_tokens=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert UNK_ID not in ids

    decoded = decode(ids, skip_special_tokens=True)
    assert decoded == text


def test_subdot_characters_are_all_present() -> None:
    subdots = "ḍḌḥḤṛṚṣṢṭṬẓẒɣƔɛƐčČǧǦ"
    for char in subdots:
        ids = encode(char, add_special_tokens=False)
        assert len(ids) == 1
        assert ids[0] != UNK_ID
        assert decode(ids, skip_special_tokens=False) == char


def test_unknown_character_replaces_with_unk() -> None:
    rare_text = "Emoji 😊 and Cyrillic Ж"
    ids = encode(rare_text, add_special_tokens=False)
    assert UNK_ID in ids
    decoded = decode(ids, skip_special_tokens=True)
    assert "😊" not in decoded
    assert "Ж" not in decoded


@pytest.mark.parametrize("char", ["ţ", "Ţ"])
def test_the_missing_kabyle_letter_is_pinned_as_a_known_gap(char: str) -> None:
    """`ţ` U+0163 is real Kabyle — `docs/orthography.md` §4 attests it 21,058 times — and has
    no slot in this table, so every line carrying it is unrecoverable. Adding it changes
    `VOCAB_SIZE` and with it the width of `lm_head`, so the release discloses the gap on its
    card. This test holds the disclosure to the table: the day the letter is added, it fails
    and the card is corrected with it.
    """
    assert encode(char, add_special_tokens=False) == [UNK_ID]


def test_the_vocabulary_holds_no_duplicate_symbols() -> None:
    """`VOCAB_SIZE` is the width of the logit projection. A repeated character would give
    one glyph two classes that can never both be right."""
    assert len(set(VOCABULARY)) == VOCAB_SIZE
