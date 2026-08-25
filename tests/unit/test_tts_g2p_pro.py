"""The Pro front end: the letters of `agbalu.tts.g2p`, plus the marks v1 deleted.

Every assertion here is about something `Matoub-82M` got wrong. The base carries
`! " ( ) , . : ; ? — “ ” …` as trained rows and v1 fed it none of them, so the model was
fitted to sequences in which no utterance ends. The one thing that must never happen is the
one that already shipped: a character leaving the front end silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.tts import g2p, g2p_pro
from agbalu.tts.g2p import PhonemeError
from agbalu.tts.vocabulary import Vocabulary

TABLE = Path("resources/kokoro_vocabulary.json")


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load(TABLE)


class TestTheAlphabet:
    def test_every_symbol_the_front_end_can_emit_has_a_row_in_the_base(
        self, vocabulary: Vocabulary
    ) -> None:
        """The whole point of reusing Kokoro's marks rather than inventing tokens: nothing
        here needs a new embedding row, so the punctuation arrives already trained."""
        assert not sorted(g2p_pro.emitted() - set(vocabulary.symbols))

    def test_the_punctuation_it_emits_is_the_curly_pair_and_not_the_guillemets(self) -> None:
        marks = g2p_pro.punctuation()
        assert {"“", "”", "—"} <= marks
        assert "«" not in marks
        assert "»" not in marks

    def test_the_tie_bar_never_survives_the_fold(self) -> None:
        assert g2p_pro.TIE_BAR in g2p.inventory()
        assert g2p_pro.TIE_BAR not in g2p_pro.emitted()

    def test_the_letter_rules_are_imported_rather_than_copied(self) -> None:
        """Two tables that must agree are one table. `g2p_pro` overrides tokenisation and
        nothing else, so a rule fixed in one place is fixed in both."""
        assert g2p_pro.phonemize("azul") == g2p.phonemize("azul")


class TestPunctuationSurvives:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Azul.", "æzul."),
            ("Azul?", "æzul?"),
            ("Azul!", "æzul!"),
            ("Azul, azul", "æzul, æzul"),
            ("Azul; azul", "æzul; æzul"),
            ("Azul: azul", "æzul: æzul"),
            ("Azul…", "æzul…"),
            ("(azul)", "(æzul)"),
            ("«azul»", "“æzul”"),
        ],
    )
    def test_each_mark_binds_to_its_word_with_no_space(self, text: str, expected: str) -> None:
        assert g2p_pro.phonemize(text) == expected

    def test_a_question_mark_reaches_the_output_of_a_real_transcript(self) -> None:
        assert g2p_pro.phonemize("Zemreɣ ad am-d-ɛawdeɣ?").endswith("?")

    def test_v1_deleted_all_of_it(self) -> None:
        """The measurement this module exists for: the old front end emits no mark at all."""
        text = "Azul, amek i telliḍ?"
        assert not set(g2p.phonemize(text)) & g2p_pro.punctuation()
        assert set(g2p_pro.phonemize(text)) & {",", "?"}


class TestTheScanner:
    def test_a_mark_with_no_space_after_it_still_separates_two_words(self) -> None:
        """70 of the corpus's 121 unphonemisable transcripts were a missing space, not a
        defect. Stripping the ends of a token loses them; scanning it recovers them."""
        assert g2p_pro.phonemize("isental,asemres") == g2p_pro.phonemize("isental, asemres")

    def test_a_standalone_dash_opens_a_line_of_dialogue(self) -> None:
        assert g2p_pro.phonemize("- Azul") == f"{g2p_pro.DASH}æzul"

    def test_a_hyphen_inside_a_word_stays_a_clitic_boundary(self) -> None:
        assert g2p_pro.phonemize("am-d") == g2p.phonemize("am-d")

    def test_a_straight_quote_alternates_within_one_utterance(self) -> None:
        rendered = g2p_pro.phonemize('Yenna "azul" i "azul"')
        assert rendered.count("“") == 2
        assert rendered.count("”") == 2
        assert rendered.index("“") < rendered.index("”")

    def test_an_apostrophe_separates_rather_than_being_pronounced_or_dropped(self) -> None:
        assert g2p_pro.phonemize("l'azul") == g2p_pro.phonemize("l azul")


class TestItRefusesRatherThanDropping:
    @pytest.mark.parametrize("text", ["Zénith", "Saïd", "waḥ€d", "arraw_ines", "a&b"])
    def test_a_character_with_no_rule_raises(self, text: str) -> None:
        with pytest.raises(PhonemeError):
            g2p_pro.phonemize(text)

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_nothing_to_say_raises_rather_than_returning_an_empty_string(self, text: str) -> None:
        with pytest.raises(PhonemeError):
            g2p_pro.phonemize(text)

    def test_a_token_of_boundaries_alone_raises(self) -> None:
        with pytest.raises(PhonemeError):
            g2p_pro.phonemize("azul '' azul")

    @pytest.mark.parametrize("mark", ["﻿", " ", "\u200b"])
    def test_invisible_characters_are_refused_rather_than_pronounced(self, mark: str) -> None:
        """BOM, NBSP and ZWSP are 2.6–3.2% of this project's seed corpora. `str.split`
        consumes the NBSP as whitespace; the other two must not reach a phoneme."""
        text = f"az{mark}ul"
        if text.split() == ["az", "ul"]:
            assert g2p_pro.phonemize(text) == "æz ul"
            return
        with pytest.raises(PhonemeError):
            g2p_pro.phonemize(text)

    def test_the_pipe_the_filelist_splits_on_can_never_be_emitted(self) -> None:
        """`meldataset` splits every filelist line on `|` with `str.split`, so a phoneme
        string carrying one silently becomes a four-field row."""
        assert "|" not in g2p_pro.emitted()
        with pytest.raises(PhonemeError):
            g2p_pro.phonemize("azul|azul")


class TestAgainstTheTokenTable:
    def test_a_rendered_line_encodes_without_a_single_unmapped_symbol(
        self, vocabulary: Vocabulary
    ) -> None:
        ipa = g2p_pro.phonemize('Yenna-yas: "Ur ttagad ara" ; yeddem-itt.')
        assert not vocabulary.unmapped(ipa)
        assert len(vocabulary.encode(ipa)) == len(ipa)

    def test_the_table_on_disk_still_carries_the_marks_this_module_maps_onto(self) -> None:
        """Read from the vendored file rather than asserted: if Kokoro's own table ever loses
        a row, the front end must fail here and not at synthesis."""
        payload = json.loads(TABLE.read_text(encoding="utf-8"))
        base = set(payload["base"])
        assert {".", ",", "?", "!", ";", ":", "…", "(", ")", "“", "”", "—"} <= base
