"""The CTC output vocabulary (task 5.2).

The class this file exists for is the last one: a vocabulary taken from raw Common
Voice text contains Greek epsilon *and* Latin open-e as separate CTC classes, so the
model learns to emit both and reproduces the corpus defect at 571 hours of scale.
Normalising first is what prevents it, and the test asserts the difference rather
than the fixed number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.normalise import normalise
from agbalu.speech import vocabulary
from agbalu.speech.vocabulary import PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER, Vocabulary, ctc_target


class TestTarget:
    def test_casefolds(self) -> None:
        assert ctc_target("Azul Fell-awen") == "azul fell-awen"

    def test_drops_sentence_punctuation(self) -> None:
        assert ctc_target("Azul, ay amdan!") == "azul ay amdan"

    def test_keeps_the_hyphen(self) -> None:
        """`docs/orthography.md` §2: a letter of the writing system, never stripped."""
        assert ctc_target("yenna-yas-d") == "yenna-yas-d"

    def test_keeps_every_kabyle_specific_letter(self) -> None:
        assert ctc_target("ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ") == "ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ"

    def test_capitals_fold_onto_their_own_letter(self) -> None:
        assert ctc_target("Ɣ Ɛ Ḥ Ḍ Ṣ Ṭ Ẓ Ṛ Č Ǧ") == "ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ"

    def test_collapses_whitespace(self) -> None:
        assert ctc_target("  azul \t\n  fell-awen  ") == "azul fell-awen"

    def test_punctuation_becomes_a_boundary_not_a_join(self) -> None:
        assert ctc_target("azul,fell-awen") == "azul fell-awen"

    @pytest.mark.parametrize("text", ["", "   ", "?!.", "123", "€ 42"])
    def test_nothing_speakable_is_empty(self, text: str) -> None:
        assert ctc_target(text) == ""

    def test_digits_are_dropped(self) -> None:
        """An unexpanded numeral cannot be read aloud; it is a defect, not a class."""
        assert ctc_target("deg 2026 n useggas") == "deg n useggas"


class TestUnspeakable:
    def test_punctuation_is_not_reported(self) -> None:
        assert list(vocabulary.unspeakable("azul, ay amdan!")) == []

    def test_a_foreign_letter_is_reported(self) -> None:
        assert list(vocabulary.unspeakable("café")) == ["é"]

    def test_a_digit_is_reported(self) -> None:
        assert list(vocabulary.unspeakable("2026")) == ["2", "0", "2", "6"]

    def test_a_currency_sign_is_reported(self) -> None:
        assert list(vocabulary.unspeakable("10 €")) == ["1", "0", "€"]

    def test_a_kabyle_letter_is_not_reported(self) -> None:
        assert list(vocabulary.unspeakable("aɣbalu")) == []


class TestBuild:
    def test_inventory_counts_characters(self) -> None:
        built = vocabulary.build(["azul", "azul"])
        assert built.counts == {"a": 2, "z": 2, "u": 2, "l": 2}

    def test_space_is_a_class(self) -> None:
        assert " " in vocabulary.build(["azul fell-awen"]).counts

    def test_audit_names_what_the_target_lost(self) -> None:
        built = vocabulary.build([ctc_target("café 2026")], sources=["café 2026"])
        assert built.unexpected == {"é": 1, "2": 2, "0": 1, "6": 1}

    def test_no_sources_means_no_audit(self) -> None:
        assert vocabulary.build([ctc_target("café")]).unexpected == {}

    def test_empty_corpus(self) -> None:
        built = vocabulary.build([])
        assert built.counts == {}
        assert built.as_mapping() == {PAD_TOKEN: 0, UNK_TOKEN: 1, WORD_DELIMITER: 2}


class TestMapping:
    def test_pad_is_zero(self) -> None:
        """`Wav2Vec2ForCTC` reads `pad_token_id` as the blank; id 0 keeps them the same."""
        assert vocabulary.build(["azul"]).as_mapping()[PAD_TOKEN] == 0

    def test_ids_are_contiguous_and_unique(self) -> None:
        mapping = vocabulary.build(["azul fell-awen ɣef"]).as_mapping()
        assert sorted(mapping.values()) == list(range(len(mapping)))

    def test_space_is_exported_as_the_delimiter(self) -> None:
        mapping = vocabulary.build(["azul fell"]).as_mapping()
        assert WORD_DELIMITER in mapping
        assert " " not in mapping

    def test_every_character_becomes_a_class(self) -> None:
        built = vocabulary.build(["aɣbalu-nneɣ d azul"])
        mapping = built.as_mapping()
        for character in built.characters:
            if character != " ":
                assert character in mapping

    def test_written_file_round_trips(self, tmp_path: Path) -> None:
        built = vocabulary.build(["azul fell-awen ɣef tmurt"])
        path = tmp_path / "asr" / "vocab.json"
        built.write(path)
        assert json.loads(path.read_text(encoding="utf-8")) == built.as_mapping()


class TestUnnormalisedTranscriptsCorruptTheTargets:
    """What the homoglyph substitution costs, and where it is caught.

    The reference Common Voice recipe takes `set(transcripts)` as its vocabulary, so
    Greek epsilon and Latin open-e become two CTC classes and the model learns to
    emit both. Gating on `ALPHABET` refuses that — at the price of a second failure
    mode: the Greek letter is not a class, so it is deleted from the target and the
    model is taught that /ʕ/ is silent. `unexpected` is what makes it visible.
    """

    RAW = "Aεdawen n tmurt, aγrib d aεessas."

    def test_the_reference_recipe_would_make_two_classes_of_one_letter(self) -> None:
        naive = set(self.RAW.casefold()) | set(normalise(self.RAW).casefold())
        assert {"ε", "ɛ"} <= naive
        assert {"γ", "ɣ"} <= naive

    def test_gating_on_the_alphabet_keeps_the_homoglyph_out(self) -> None:
        built = vocabulary.build([ctc_target(self.RAW)])
        assert "ε" not in built.counts
        assert "γ" not in built.counts

    def test_but_the_letter_then_splits_the_word_in_two(self) -> None:
        """Worse than a lost phoneme: an unrepaired homoglyph becomes a boundary, so
        `Aεdawen` trains as two words and every WER over it is measured wrong."""
        assert ctc_target(self.RAW) == "a dawen n tmurt a rib d a essas"
        assert len(ctc_target(self.RAW).split()) == 9
        assert len(ctc_target(normalise(self.RAW)).split()) == 6

    def test_the_audit_names_it(self) -> None:
        built = vocabulary.build([ctc_target(self.RAW)], sources=[self.RAW])
        assert built.unexpected == {"ε": 2, "γ": 1}

    def test_normalising_first_is_what_keeps_the_phoneme(self) -> None:
        text = normalise(self.RAW)
        built = vocabulary.build([ctc_target(text)], sources=[text])
        assert built.counts["ɛ"] == 2
        assert built.counts["ɣ"] == 1
        assert built.unexpected == {}


class TestEncoder:
    def test_the_space_maps_onto_the_delimiter(self) -> None:
        """Without this the target loses every boundary and the model never emits one."""
        mapping = vocabulary.build(["azul fell"]).as_mapping()
        assert vocabulary.encoder(mapping)[" "] == mapping[WORD_DELIMITER]

    def test_special_tokens_are_not_characters(self) -> None:
        encode = vocabulary.encoder(vocabulary.build(["azul"]).as_mapping())
        assert PAD_TOKEN not in encode
        assert UNK_TOKEN not in encode

    def test_every_target_character_encodes(self) -> None:
        target = ctc_target("Aɣbalu-nneɣ d azul, ay amdan!")
        encode = vocabulary.encoder(vocabulary.build([target]).as_mapping())
        assert all(ch in encode for ch in target)

    def test_a_vocabulary_without_a_delimiter_is_refused(self) -> None:
        with pytest.raises(KeyError, match="delimited"):
            vocabulary.encoder({PAD_TOKEN: 0, "a": 1})


class TestDecode:
    def test_round_trips_a_target(self) -> None:
        """The emission is not the label sequence: a repeated class needs a blank
        between its two frames, which is what the model learns to emit."""
        target = "azul fell-awen ɣef tmurt"
        mapping = vocabulary.build([target]).as_mapping()
        encode = vocabulary.encoder(mapping)
        emission: list[int] = []
        for character in target:
            index = encode[character]
            if emission and emission[-1] == index:
                emission.append(mapping[PAD_TOKEN])
            emission.append(index)
        assert vocabulary.decode(emission, mapping) == target

    def test_blanks_are_dropped(self) -> None:
        mapping = vocabulary.build(["azul"]).as_mapping()
        ids = [mapping[PAD_TOKEN], mapping["a"], mapping[PAD_TOKEN], mapping["z"]]
        assert vocabulary.decode(ids, mapping) == "az"

    def test_repeats_are_collapsed(self) -> None:
        mapping = vocabulary.build(["azul"]).as_mapping()
        assert vocabulary.decode([mapping["a"]] * 5, mapping) == "a"

    def test_a_doubled_letter_needs_a_blank_between_it(self) -> None:
        """`ll` is emitted as `l blank l`; collapsing after dropping blanks loses one."""
        mapping = vocabulary.build(["fell"]).as_mapping()
        ids = [mapping["l"], mapping[PAD_TOKEN], mapping["l"]]
        assert vocabulary.decode(ids, mapping) == "ll"
        assert vocabulary.decode([mapping["l"], mapping["l"]], mapping) == "l"

    def test_empty_emission(self) -> None:
        mapping = vocabulary.build(["azul"]).as_mapping()
        assert vocabulary.decode([], mapping) == ""
        assert vocabulary.decode([mapping[PAD_TOKEN]] * 9, mapping) == ""

    def test_leading_and_trailing_delimiters_are_stripped(self) -> None:
        mapping = vocabulary.build(["azul fell"]).as_mapping()
        ids = [mapping[WORD_DELIMITER], mapping["a"], mapping[WORD_DELIMITER]]
        assert vocabulary.decode(ids, mapping) == "a"

    def test_an_unknown_id_does_not_crash_the_decode(self) -> None:
        mapping = vocabulary.build(["azul"]).as_mapping()
        assert vocabulary.decode([mapping["a"], 9999], mapping) == "a"


class TestVocabularyRecord:
    def test_as_dict_reports_the_class_count(self) -> None:
        built = vocabulary.build(["azul"])
        assert built.as_dict()["classes"] == len(built.as_mapping())

    def test_as_dict_names_the_space(self) -> None:
        payload = vocabulary.build(["a b"]).as_dict()
        counts = payload["counts"]
        assert isinstance(counts, dict)
        assert "<space>" in counts

    def test_characters_are_sorted(self) -> None:
        built = Vocabulary(counts={"z": 1, "a": 1}, unexpected={})
        assert built.characters == ("a", "z")
