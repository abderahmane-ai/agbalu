"""The G2P table: the rules it encodes, and what it refuses."""

from __future__ import annotations

import pytest

from agbalu.tts.g2p import (
    BOUNDARY,
    NASAL_ASSIMILATION,
    PLAIN,
    SPIRANTS,
    PhonemeError,
    inventory,
    phonemize,
    phonemize_word,
    tokenize,
    unsupported,
)


class TestSpirantization:
    def test_singleton_stop_is_a_fricative(self) -> None:
        assert phonemize_word("ata") == "æθæ"
        assert phonemize_word("ada") == "æðæ"

    def test_gemination_blocks_it(self) -> None:
        assert phonemize_word("atta") == "ætːæ"
        assert phonemize_word("adda") == "ædːæ"

    @pytest.mark.parametrize(("letter", "short", "stop"), [(k, *v) for k, v in SPIRANTS.items()])
    def test_every_spirant_pair(self, letter: str, short: str, stop: str) -> None:
        """Framed in `e`, which no rule moves, so only spirantization is under test."""
        assert phonemize_word(f"e{letter}e") == f"ə{short}ə"
        assert phonemize_word(f"e{letter}{letter}e") == f"ə{stop}ːə"

    def test_emphatic_d_spirantizes_and_backs_its_neighbours(self) -> None:
        assert phonemize_word("aḍa") == "ɑðˤɑ"
        assert phonemize_word("aḍḍa") == "ɑdˤːɑ"


class TestGemination:
    def test_non_spirant_consonant_lengthens(self) -> None:
        assert phonemize_word("alla") == "ælːæ"

    def test_vowels_never_geminate(self) -> None:
        assert phonemize_word("chafaa").endswith("ææ")
        assert phonemize_word("aa") == "ææ"

    def test_length_follows_pharyngealization(self) -> None:
        assert phonemize_word("aṛṛa") == "ɑrˤːɑ"

    def test_triple_letter_is_a_geminate_then_a_singleton(self) -> None:
        assert phonemize_word("attta") == "ætːθæ"


class TestVowelBacking:
    def test_a_backs_beside_an_emphatic(self) -> None:
        assert phonemize_word("aḍ") == "ɑðˤ"

    def test_a_stays_front_without_one(self) -> None:
        assert phonemize_word("ala") == "ælæ"

    def test_backing_applies_on_either_side(self) -> None:
        assert phonemize_word("ṛa") == "rˤɑ"
        assert phonemize_word("aṛ") == "ɑrˤ"

    def test_lax_allophones_are_not_produced(self) -> None:
        """`i`/`u` laxing scores 75.59% against a 74.99% baseline, so it is not modelled."""
        assert "ɪ" not in phonemize_word("abiskwi")
        assert "ʊ" not in phonemize_word("abuchid")


class TestNasalAssimilation:
    @pytest.mark.parametrize(("following", "nasal"), sorted(NASAL_ASSIMILATION.items()))
    def test_place_assimilation(self, following: str, nasal: str) -> None:
        assert phonemize_word(f"an{following}a").startswith(f"æ{nasal}")

    def test_plain_n_elsewhere(self) -> None:
        assert phonemize_word("ana") == "ænæ"

    def test_geminate_n_does_not_assimilate(self) -> None:
        assert phonemize_word("annfa") == "ænːfæ"


class TestLegacyTenseT:
    def test_t_cedilla_is_not_dropped(self) -> None:
        """The published source emits nothing for it in all 73 of its words."""
        assert phonemize_word("aţan") == "ætːæn"

    def test_it_maps_to_the_geminate(self) -> None:
        assert phonemize_word("ţ") == "tː"


class TestRefusals:
    @pytest.mark.parametrize("text", ["3d", "androïd", "rosé", "muḥ€nd", "ṭeyyeb‟"])
    def test_unmapped_characters_raise(self, text: str) -> None:
        with pytest.raises(PhonemeError):
            phonemize_word(text)

    def test_the_message_names_the_codepoint(self) -> None:
        with pytest.raises(PhonemeError, match=r"U\+00E9"):
            phonemize_word("rosé")

    def test_o_is_mapped_rather_than_dropped(self) -> None:
        """`o` is not Kabyle, but the source deleted it silently in 128 words."""
        assert phonemize_word("adolf") == "æðolf"

    def test_a_combining_mark_is_not_silently_absorbed(self) -> None:
        with pytest.raises(PhonemeError):
            phonemize_word("xelleṣ̣")


class TestTokenize:
    def test_clitic_hyphen_is_a_boundary(self) -> None:
        assert tokenize("deg-s") == ["deg", "s"]

    def test_punctuation_is_stripped(self) -> None:
        assert tokenize("Azul, a·y?") == ["Azul", "a·y"]

    @pytest.mark.parametrize("space", [" ", " ", "\t", "\n", "  "])
    def test_unicode_whitespace_separates(self, space: str) -> None:
        assert tokenize(f"azul{space}fell") == ["azul", "fell"]

    def test_empty_and_punctuation_only(self) -> None:
        assert tokenize("") == []
        assert tokenize("  ...  ") == []

    def test_zero_width_space_is_not_a_separator_and_does_not_pass_silently(self) -> None:
        """U+200B is category Cf, not whitespace, so it survives tokenization and must
        then be refused rather than becoming an inaudible gap in a training target."""
        word = "az\u200bul"
        assert tokenize(word) == [word]
        with pytest.raises(PhonemeError, match=r"U\+200B"):
            phonemize_word(word)


class TestPhonemize:
    def test_words_are_boundary_separated(self) -> None:
        assert phonemize("ala ala") == f"ælæ{BOUNDARY}ælæ"

    def test_empty_text(self) -> None:
        assert phonemize("") == ""

    def test_case_is_folded(self) -> None:
        assert phonemize_word("ALA") == phonemize_word("ala")

    def test_lexicon_supplies_attested_readings(self) -> None:
        assert phonemize("abiskwi", lexicon={"abiskwi": "æβɪsçwi"}) == "æβɪsçwi"

    def test_table_covers_what_the_lexicon_lacks(self) -> None:
        assert phonemize("ala", lexicon={"other": "x"}) == "ælæ"

    def test_empty_lexicon_reading_falls_back(self) -> None:
        assert phonemize("ala", lexicon={"ala": ""}) == "ælæ"

    def test_lexicon_lookup_is_case_folded(self) -> None:
        assert phonemize("ALA", lexicon={"ala": "ZZZ"}) == "ZZZ"


class TestInventory:
    def test_every_plain_symbol_is_covered(self) -> None:
        emitted = inventory()
        for symbol in PLAIN.values():
            assert set(symbol) <= emitted

    def test_backed_vowel_is_in_the_inventory(self) -> None:
        assert "ɑ" in inventory()

    def test_unsupported_names_what_a_vocabulary_lacks(self) -> None:
        assert unsupported(dict.fromkeys(inventory(), 0)) == frozenset()
        assert "ʕ" in unsupported(dict.fromkeys(inventory() - {"ʕ"}, 0))

    def test_kokoro_shortfall_is_the_four_known_symbols(self) -> None:
        """Measured against `hexgrad/Kokoro-82M` config.json on 2026-08-14."""
        kokoro = set(
            ';:,.!?—…"()“” ̃ʣʥʦʨᵝꭧAIOQSTWYᵊabcdefhijklmnopqrstuvwxyz'
            "ɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ"
        )
        assert unsupported(dict.fromkeys(kokoro, 0)) == frozenset({"ʕ", "ħ", "ˤ", "͡"})
