"""The characters a multilingual vocabulary drops, and the refusal that names them."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agbalu.embed.vocabulary import (
    DONORS,
    Encode,
    VocabularyError,
    assert_covered,
    coverage,
    donor_map,
    missing_characters,
)
from agbalu.tokenizer.spec import required_chars

UNK = 0


def fake_encode(vocabulary: str) -> Encode:
    """One id per character, `UNK` for anything the vocabulary does not carry."""

    def encode(text: str) -> list[int]:
        return [vocabulary.index(char) + 1 if char in vocabulary else UNK for char in text]

    return encode


def encodable_with(vocabulary: str) -> Callable[[str], bool]:
    encode = fake_encode(vocabulary)
    return lambda char: UNK not in encode(char)


class TestMissingCharacters:
    def test_empty_vocabulary_misses_every_required_character(self) -> None:
        missing = missing_characters(fake_encode(""), UNK)
        assert len(missing) == len(required_chars())

    def test_full_vocabulary_misses_nothing(self) -> None:
        assert missing_characters(fake_encode(required_chars()), UNK) == ()

    def test_names_only_what_is_absent(self) -> None:
        vocabulary = required_chars().replace("ẓ", "").replace("Ẓ", "")
        assert missing_characters(fake_encode(vocabulary), UNK) == ("Ẓ", "ẓ")

    def test_respects_an_explicit_character_set(self) -> None:
        assert missing_characters(fake_encode("ab"), UNK, chars="abc") == ("c",)

    def test_empty_character_set_is_not_an_error(self) -> None:
        assert missing_characters(fake_encode(""), UNK, chars="") == ()


class TestCoverage:
    def test_counts_tokens_over_whitespace_words(self) -> None:
        result = coverage(fake_encode("abc "), UNK, ["ab c"])
        assert result.words == 2
        assert result.tokens_per_word == 2.0
        assert result.sentences == 1

    def test_unknown_rate_is_over_tokens(self) -> None:
        result = coverage(fake_encode("ab"), UNK, ["abz"])
        assert result.unknown_rate == pytest.approx(1 / 3)

    def test_empty_input_returns_zero_rather_than_dividing(self) -> None:
        result = coverage(fake_encode("ab"), UNK, [])
        assert result.tokens_per_word == 0.0
        assert result.unknown_rate == 0.0
        assert result.words == 0

    def test_whitespace_only_sentence_has_no_words(self) -> None:
        result = coverage(fake_encode("ab "), UNK, ["   "])
        assert result.words == 0
        assert result.tokens_per_word == 0.0

    def test_clean_requires_both_no_missing_and_no_unknown(self) -> None:
        assert coverage(fake_encode(required_chars()), UNK, ["ab"]).clean
        assert not coverage(fake_encode("ab"), UNK, ["ab"]).clean

    def test_clean_is_false_when_a_sentence_carries_an_unknown(self) -> None:
        result = coverage(fake_encode(required_chars()), UNK, ["abԐ"])
        assert result.missing == ()
        assert result.unknown_rate > 0.0
        assert not result.clean

    def test_is_frozen(self) -> None:
        result = coverage(fake_encode("ab"), UNK, ["ab"])
        field = "words"
        with pytest.raises(AttributeError):
            setattr(result, field, 5)


class TestDonorMap:
    def test_uppercase_takes_its_lowercase_when_that_encodes(self) -> None:
        assert donor_map(["Ɣ"], encodable_with("ɣ")) == {"Ɣ": "ɣ"}

    def test_uppercase_shares_the_lowercase_donor_when_both_are_missing(self) -> None:
        chosen = donor_map(["Ẓ", "ẓ"], encodable_with("zṣ"))
        assert chosen == {"Ẓ": "zṣ", "ẓ": "zṣ"}

    def test_undeclared_character_is_refused_by_name(self) -> None:
        with pytest.raises(VocabularyError, match=r"U\+00F1"):
            donor_map(["ñ"], encodable_with("abc"))

    def test_donor_that_is_itself_unencodable_is_refused(self) -> None:
        with pytest.raises(VocabularyError, match="unencodable"):
            donor_map(["ẓ"], encodable_with("z"))

    def test_empty_input_returns_empty(self) -> None:
        assert donor_map([], encodable_with("abc")) == {}

    def test_every_declared_donor_is_lowercase_and_self_consistent(self) -> None:
        for char, donor in DONORS.items():
            assert char.lower() == char
            assert char not in donor


class TestAssertCovered:
    def test_passes_on_a_complete_vocabulary(self) -> None:
        assert_covered(fake_encode(required_chars()), UNK)

    def test_raises_naming_each_missing_codepoint(self) -> None:
        vocabulary = required_chars().replace("ẓ", "")
        with pytest.raises(VocabularyError, match=r"ẓ \(U\+1E93\)"):
            assert_covered(fake_encode(vocabulary), UNK)

    def test_reports_the_count(self) -> None:
        with pytest.raises(VocabularyError, match=f"maps {len(required_chars())} required"):
            assert_covered(fake_encode(""), UNK)
