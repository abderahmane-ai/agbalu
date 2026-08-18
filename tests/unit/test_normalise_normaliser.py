from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agbalu.normalise import NORMALISER_VERSION, Normaliser, normalise
from agbalu.normalise.normaliser import INVISIBLE
from agbalu.normalise.rules import ALPHABET, HYPHEN

KABYLE_LETTERS = "abcčdḍefgǧɣhḥijklmnqrsṣtṭuwxyzẓɛṛ"


@pytest.fixture(scope="module")
def n() -> Normaliser:
    return Normaliser()


@pytest.mark.parametrize(
    ("bad", "good"),
    [
        ("ε", "ɛ"),  # Greek small epsilon      -> latin open e
        ("Σ", "Ɛ"),  # Greek capital sigma      -> latin capital open e
        ("γ", "ɣ"),  # Greek small gamma        -> latin small gamma
        ("Γ", "Ɣ"),  # Greek capital gamma      -> latin capital gamma
        ("Ԑ", "Ɛ"),  # Cyrillic reversed ze     -> latin capital open e
        ("ԑ", "ɛ"),  # Cyrillic small reversed  -> latin small open e
    ],
)
def test_primary_homoglyphs_are_folded(n: Normaliser, bad: str, good: str) -> None:
    assert n.normalise(bad) == good


def test_the_epsilon_case_from_the_real_corpus(n: Normaliser) -> None:
    # 11.21% of open-e characters in Kabyle running text are this defect.
    assert n.normalise("Ayɣer i teεyiḍ akk annect-a?") == "Ayɣer i teɛyiḍ akk annect-a?"


@pytest.mark.parametrize(
    ("bad", "good"),
    [
        ("ğ", "ǧ"),
        ("ĝ", "ǧ"),
        ("ć", "č"),
        ("ť", "ṭ"),
        ("ț", "ṭ"),
        ("š", "ṣ"),
        ("ž", "ẓ"),
        ("ř", "ṛ"),
    ],
)
def test_secondary_homoglyphs_are_folded(n: Normaliser, bad: str, good: str) -> None:
    assert n.normalise(bad) == good


def test_t_cedilla_is_never_rewritten(n: Normaliser) -> None:
    """`ţ` is Dallet-tradition spirantised t, not emphatic ṭ and not Romanian.

    Corpus evidence: of 1,407 word types containing ţ, only 10 have a ţ->ṭ
    counterpart, against 479 for ţ->t and 603 for ţ->tt. Folding to ṭ would turn
    `neţţa` ("he") into a nonexistent emphatic form.
    """
    assert n.normalise("neţţa") == "neţţa"
    assert n.normalise("tideţ") == "tideţ"
    assert "ṭ" not in n.normalise("ţxil-k")


def test_t_cedilla_is_flagged_rather_than_silently_kept(n: Normaliser) -> None:
    flags = n.analyse("neţţa").flags
    assert [f.kind for f in flags] == ["legacy-t-cedilla", "legacy-t-cedilla"]


def test_capital_t_cedilla_is_also_preserved(n: Normaliser) -> None:
    assert n.normalise("Ţ") == "Ţ"


@pytest.mark.parametrize("ch", ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad", "\u2060"])
def test_invisible_characters_are_removed(n: Normaliser, ch: str) -> None:
    assert n.normalise(f"aɣ{ch}balu") == "aɣbalu"


@pytest.mark.parametrize("ch", [" ", " ", " ", " "])
def test_exotic_spaces_become_ascii_space(n: Normaliser, ch: str) -> None:
    assert n.normalise(f"a{ch}b") == "a b"


def test_runs_of_spaces_collapse(n: Normaliser) -> None:
    assert n.normalise("a     b") == "a b"


def test_leading_and_trailing_space_is_stripped(n: Normaliser) -> None:
    assert n.normalise("   aɣbalu   ") == "aɣbalu"


def test_newlines_survive_but_their_lines_are_stripped(n: Normaliser) -> None:
    assert n.normalise("a  \n   b  ") == "a\nb"


@pytest.mark.parametrize(("bad", "good"), [("–", "-"), ("—", "-"), ("‐", "-"), ("‑", "-")])
def test_dashes_become_ascii_hyphen(n: Normaliser, bad: str, good: str) -> None:
    assert n.normalise(bad) == good


def test_the_hyphen_itself_is_never_touched(n: Normaliser) -> None:
    # Clitics, coordination and compounds all depend on it.
    assert n.normalise("Tanemmirt-ik") == "Tanemmirt-ik"
    assert HYPHEN in n.normalise("ɣur-i")


def test_curly_quotes_become_straight(n: Normaliser) -> None:
    assert n.normalise("“a”") == '"a"'


def test_decomposed_forms_are_composed(n: Normaliser) -> None:
    # t + COMBINING DOT BELOW must become ṭ before any table lookup runs.
    assert n.normalise("ṭ") == "ṭ"


def test_output_is_always_nfc(n: Normaliser) -> None:
    out = n.normalise("ṭaṣ")
    assert unicodedata.normalize("NFC", out) == out


def test_a_bare_combining_mark_does_not_crash(n: Normaliser) -> None:
    assert isinstance(n.normalise("̣"), str)


def test_french_accents_are_kept_by_default(n: Normaliser) -> None:
    # Folding é->e destroys proper nouns; it must be requested explicitly.
    assert n.normalise("Aéroport") == "Aéroport"


def test_french_accents_fold_when_requested() -> None:
    folding = Normaliser(fold_diacritics=True)
    assert folding.normalise("Aéroport") == "Aeroport"


def test_folding_is_reported_as_a_change() -> None:
    folding = Normaliser(fold_diacritics=True)
    assert folding.analyse("é").count("diacritic-fold") == 1


def test_digraphs_are_flagged_not_substituted(n: Normaliser) -> None:
    result = n.analyse("ghef")
    assert result.text == "ghef"
    assert any(f.kind == "ascii-digraph" for f in result.flags)


def test_a_french_loanword_digraph_is_flagged_but_intact(n: Normaliser) -> None:
    assert n.normalise("chose") == "chose"


def test_rejected_characters_are_flagged_not_deleted(n: Normaliser) -> None:
    result = n.analyse("ça")
    assert "ç" in result.text
    assert any(f.kind == "rejected-character" for f in result.flags)


def test_out_of_inventory_letters_are_flagged(n: Normaliser) -> None:
    result = n.analyse("日本")
    assert any(f.kind == "out-of-inventory" for f in result.flags)


def test_empty_string(n: Normaliser) -> None:
    assert n.normalise("") == ""


def test_whitespace_only_string(n: Normaliser) -> None:
    assert n.normalise("      ") == ""


def test_a_string_of_only_invisibles_becomes_empty(n: Normaliser) -> None:
    assert n.normalise("\u200b\u200c\ufeff") == ""


def test_very_long_input_is_handled(n: Normaliser) -> None:
    assert n.normalise("ε" * 50_000) == "ɛ" * 50_000


def test_surrogate_free_astral_characters_survive(n: Normaliser) -> None:
    assert "𐒰" in n.normalise("aɣbalu 𐒰")


def test_analyse_reports_the_original(n: Normaliser) -> None:
    result = n.analyse("teεyiḍ")
    assert result.original == "teεyiḍ"
    assert result.changed
    assert result.count("homoglyph") == 1


def test_version_is_reported(n: Normaliser) -> None:
    """Asserts the wiring, not the number: a version bump is a deliberate act."""
    assert n.version == f"{NORMALISER_VERSION}+rules{n.rules.version}"


def test_module_level_helper_matches_the_class(n: Normaliser) -> None:
    assert normalise("teεyiḍ") == n.normalise("teεyiḍ")


kabyle_text = st.text(alphabet=st.sampled_from(KABYLE_LETTERS + " -.,?!"), max_size=120)
any_text = st.text(max_size=200)


@given(any_text)
@settings(max_examples=1000, deadline=None)
def test_normalisation_is_idempotent(text: str) -> None:
    """The exit criterion, over arbitrary Unicode rather than only the corpus."""
    once = normalise(text)
    assert normalise(once) == once


@pytest.mark.parametrize(
    "separator",
    [" ", " ", " ", " ", " ", " ", "\t", "\t\t", "  ", "\n"],
)
def test_any_whitespace_separates_the_guarded_token_from_the_next(separator: str) -> None:
    """The 1.2.0 defect: the guard split on U+0020 alone, so `À\xa0ð` normalised to
    `À ð` and then to `À ḍ` — shielded while the NBSP joined the two, unshielded once
    it became a space."""
    once = normalise(f"À{separator}ð")
    assert once == normalise(once)
    assert "ḍ" in once


def test_a_guarded_token_still_survives_a_plain_space(n: Normaliser) -> None:
    """The guard must keep doing its job; only its idea of a boundary changed."""
    assert n.normalise("Chişinău ε") == "Chişinău ɛ"


@given(kabyle_text)
@settings(max_examples=300, deadline=None)
def test_clean_kabyle_is_a_fixed_point(text: str) -> None:
    """Text already in the inventory must pass through untouched, modulo spacing."""
    expected = " ".join(text.split()) if text.strip() else ""
    assert normalise(text) == expected


@given(any_text)
@settings(max_examples=300, deadline=None)
def test_no_canonical_kabyle_letter_is_ever_destroyed(text: str) -> None:
    """Normalisation may add canonical letters; it may never remove one."""
    before = sum(text.count(c) for c in ALPHABET)
    after = sum(normalise(text).count(c) for c in ALPHABET)
    assert after >= before


@given(any_text)
@settings(max_examples=300, deadline=None)
def test_output_is_always_nfc_property(text: str) -> None:
    out = normalise(text)
    assert unicodedata.normalize("NFC", out) == out


@given(any_text)
@settings(max_examples=200, deadline=None)
def test_normalisation_never_introduces_an_invisible(text: str) -> None:
    assert not (set(normalise(text)) & INVISIBLE)


@given(any_text)
@settings(max_examples=200, deadline=None)
def test_flags_never_change_the_text(text: str) -> None:
    normaliser = Normaliser()
    result = normaliser.analyse(text)
    assert result.text == normaliser.normalise(text)


class TestForeignProperNouns:
    """Homoglyph rules must repair Kabyle without rewriting other languages."""

    def test_a_romanian_proper_noun_is_preserved(self) -> None:
        n = Normaliser()
        assert n.normalise("Tamanaɣt n Muldufa d Chişinău.") == "Tamanaɣt n Muldufa d Chişinău."

    def test_mojibake_capitals_do_not_block_a_real_repair(self) -> None:
        """`Ʃ` is atomic, so it is corruption rather than another alphabet.

        Treating any out-of-inventory letter as foreign blocked the legitimate
        `γ`→`ɣ` repair in `Ʃemdeγ-am` across 1,004 corpus lines.
        """
        assert Normaliser().normalise("Ʃemdeγ-am") == "Ʃemdeɣ-am"

    def test_a_protected_token_does_not_veto_its_neighbours(self) -> None:
        assert Normaliser().normalise("Nesεa Chişinău d leεmeṛ.") == "Nesɛa Chişinău d leɛmeṛ."

    def test_lowercase_is_never_protected(self) -> None:
        assert Normaliser().normalise("yesεa") == "yesɛa"

    @pytest.mark.parametrize("char", ["ă", "â", "î", "ë", "ü", "ñ", "ç"])
    def test_decomposing_european_letters_mark_foreign_orthography(self, char: str) -> None:
        assert Normaliser().is_foreign_orthography_mark(char)

    @pytest.mark.parametrize("char", ["Ʃ", "ð", "þ", "ŋ", "ɣ", "ḍ", "ṣ", "ş", "a", "1", " "])
    def test_atomic_kabyle_and_rule_letters_do_not(self, char: str) -> None:
        assert not Normaliser().is_foreign_orthography_mark(char)

    @pytest.mark.parametrize("char", ["š", "ž", "ş"])
    def test_a_rewrite_source_is_never_its_own_foreign_evidence(self, char: str) -> None:
        """`š`->`ṣ` is a deliberate rule, so `š` cannot prove a token is Czech.

        A proper noun whose only foreign letter is itself a rewrite source — `Škoda`
        — is therefore not protected. Distinguishing it from corrupted Kabyle needs
        a lexicon, not a character test.
        """
        assert not Normaliser().is_foreign_orthography_mark(char)

    def test_protection_is_reported_as_a_flag(self) -> None:
        result = Normaliser().analyse("Tamanaɣt n Muldufa d Chişinău.")
        assert [f.kind for f in result.flags].count("foreign-proper-noun") == 1
        assert not [c for c in result.changes if c.kind == "homoglyph"]

    def test_protection_is_idempotent(self) -> None:
        n = Normaliser()
        for text in ("Chişinău", "Ʃemdeγ-am", "Nesεa Chişinău d leεmeṛ."):
            once = n.normalise(text)
            assert n.normalise(once) == once

    @pytest.mark.parametrize("char", ["ţ", "Ţ"])
    def test_a_preserved_kabyle_letter_is_not_foreign_evidence(self, char: str) -> None:
        """`ţ` decomposes as `ă` does but is Kabyle — `docs/orthography.md` §4 keeps it
        on 21,058 occurrences across 6,851 word types."""
        assert not Normaliser().is_foreign_orthography_mark(char)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Yeţwaṛfaε weqcic-nni.", "Yeţwaṛfaɛ weqcic-nni."),
            ("Ḥess i Ṛebbi-Aţ-ţεiceḍ i lebda", "Ḥess i Ṛebbi-Aţ-ţɛiceḍ i lebda"),
            ("Ţeţţuɣ aεdaw-iw.", "Ţeţţuɣ aɛdaw-iw."),
        ],
    )
    def test_t_cedilla_does_not_block_a_homoglyph_repair(self, text: str, expected: str) -> None:
        """Left 89 unrepaired homoglyphs in AƔBALU-Text v1 under 1.1.0."""
        assert Normaliser().normalise(text) == expected

    def test_a_genuine_foreign_name_beside_t_cedilla_still_survives(self) -> None:
        """The 1.2.0 fix must not reopen what 1.1.0 closed."""
        n = Normaliser()
        text = "Ţeţţuɣ Chişinău akked Žižek, yesεa."
        assert n.normalise(text) == "Ţeţţuɣ Chişinău akked Žižek, yesɛa."
