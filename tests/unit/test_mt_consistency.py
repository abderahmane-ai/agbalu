"""Lexical stability of a document translation.

The measurement exists because chrF++ on FLORES+ cannot see it: every row there is one
sentence against one reference, so a term rendered three ways across a novel costs nothing.
Two versions of this module were wrong before the fixtures below existed — the first
reported `which`, `when` and `make` as the least stable terms in every document, and the
second counted `amcic` and `umcic`, one noun in two states, as two renderings.
"""

from __future__ import annotations

from agbalu.mt.consistency import (
    align,
    common_words,
    free_state,
    measure,
    renderings,
    tokens,
)


class TestTokens:
    def test_kabyle_diacritics_are_letters(self) -> None:
        assert tokens("Aɣerda ḥbes ṭṭraḍ") == ["aɣerda", "ḥbes", "ṭṭraḍ"]

    def test_digits_and_punctuation_are_not_tokens(self) -> None:
        assert tokens("100,000 men — and: 3 ships!") == ["men", "and", "ships"]

    def test_empty_text(self) -> None:
        assert tokens("") == []
        assert tokens("   \n\t  ") == []


class TestFreeState:
    """`amcic`/`umcic` is one noun. Counting the states apart reported a document that
    translated *cat* perfectly as 60% consistent."""

    def test_masculine_annexed_folds_onto_the_free_state(self) -> None:
        assert free_state("umcic") == free_state("amcic") == "amcic"
        assert free_state("uzrem") == free_state("azrem") == "azrem"
        assert free_state("wexxam") == "axxam"

    def test_feminine_annexed_folds_onto_the_free_state(self) -> None:
        assert free_state("temɣart") == free_state("tamɣart") == "tamɣart"

    def test_a_word_too_short_to_carry_a_prefix_is_left_alone(self) -> None:
        assert free_state("ur") == "ur"
        assert free_state("wa") == "wa"
        assert free_state("tal") == "tal"

    def test_a_word_with_no_state_prefix_is_left_alone(self) -> None:
        assert free_state("yenna") == "yenna"
        assert free_state("") == ""


class TestAlign:
    def test_a_line_whose_sentence_count_survives_is_paired(self) -> None:
        aligned = align("One cat. Two dogs.\n", "Yiwen umcic. Sin iqjan.\n")
        assert aligned.pairs == (("One cat.", "Yiwen umcic."), ("Two dogs.", "Sin iqjan."))
        assert aligned.skipped_lines == 0

    def test_a_line_whose_sentence_count_changed_is_skipped_not_guessed(self) -> None:
        """Pairing them anyway does not fail — it attributes one sentence's vocabulary to
        another, which is the exact defect this module reports."""
        aligned = align("One cat. Two dogs.\n", "Yiwen umcic d sin iqjan.\n")
        assert aligned.pairs == ()
        assert aligned.skipped_lines == 1

    def test_surplus_lines_on_either_side_are_counted_as_skipped(self) -> None:
        aligned = align("A.\nB.\nC.\n", "Ta.\nTb.\n")
        assert len(aligned.pairs) == 2
        assert aligned.skipped_lines == 1
        assert aligned.total_lines == 3

    def test_empty_documents(self) -> None:
        aligned = align("", "")
        assert aligned.pairs == ()
        assert aligned.skip_rate == 0.0


class TestRenderings:
    FILLER: int = 200
    """Segments with no `frog` in them.

    A term in *every* segment is a function word by `MAX_DOCUMENT_FRACTION`, and a fixture
    without filler tests the ceiling rather than the thing it means to test."""

    FRAMES: tuple[tuple[str, str], ...] = (
        ("sat upon the warm stone", "yeqqim ɣef ublaḍ yeḥman"),
        ("leapt across the wide river", "yezzegzew asif ehrin"),
        ("hid beneath the broken cart", "yeffer seddaw ukerrus yerrẓen"),
        ("sang until the pale morning", "yecna alamma d ṣṣbeḥ"),
        ("watched the sleeping merchant", "yemuqel amsaɣ iṭṭsen"),
        ("feared the circling bird", "yugad afrux yettezzin"),
        ("drank from the cold spring", "yeswa seg tala semmḍen"),
        ("counted the falling leaves", "yesseḥsab iferrawen yeɣlin"),
        ("waited for the returning king", "yerǧa agellid i d-yuɣalen"),
    )
    """A varied frame per segment.

    With one fixed frame the highest-Dice candidate is the frame's own verb — it co-occurs
    with the term in every segment and nowhere else — so the fixture measured the frame and
    reported perfect consistency whatever the noun did. Real documents vary; fixtures must."""

    @classmethod
    def _document(cls, target_words: list[str]) -> tuple[str, str]:
        """One sentence per line, `frog` on the source side of the first few."""
        source, target = "", ""
        for index, word in enumerate(target_words):
            english, kabyle = cls.FRAMES[index % len(cls.FRAMES)]
            source += f"The frog {english}.\n"
            target += f"{word.capitalize()} {kabyle}.\n"
        source += "".join(f"A heron flew over field number {n}.\n" for n in range(cls.FILLER))
        target += "".join(f"Afrux yufeg nnig yiger wis {n}.\n" for n in range(cls.FILLER))
        return source, target

    def test_a_term_rendered_one_way_is_fully_consistent(self) -> None:
        source, target = self._document(["amqerqur"] * 6)
        found = {term.term: term for term in renderings(align(source, target))}
        assert found["frog"].consistency == 1.0

    def test_a_term_rendered_three_ways_is_reported_with_all_of_them(self) -> None:
        source, target = self._document(["amqerqur"] * 4 + ["aɣerda"] * 3 + ["abrid"] * 2)
        found = {term.term: term for term in renderings(align(source, target))}
        assert found["frog"].occurrences == 9
        assert found["frog"].consistency == 4 / 9
        assert dict(found["frog"].renderings)["amqerqur"] == 4

    def test_the_two_states_of_one_noun_are_not_two_renderings(self) -> None:
        source, target = self._document(["amqerqur"] * 3 + ["umqerqur"] * 3)
        found = {term.term: term for term in renderings(align(source, target))}
        assert found["frog"].consistency == 1.0

    def test_a_common_source_word_is_excluded(self) -> None:
        source, target = self._document(["amqerqur"] * 6)
        found = {term.term for term in renderings(align(source, target), common={"frog"})}
        assert "frog" not in found

    def test_a_term_with_no_identifiable_rendering_is_not_reported_as_inconsistent(self) -> None:
        """A weak signal means the alignment found nothing, not that the model was
        inconsistent. Scoring the two the same put every document at 0.2.

        Nine distinct words, not `aqerqur0`–`aqerqur8`: digits are not letters to `tokens`,
        so the numbered variants collapse to one type and the fixture tested nothing."""
        nine = ["amqerqur", "aɣerda", "abrid", "afrux", "aslem", "izem", "ayaziḍ", "uccen", "izi"]
        source, target = self._document(nine)
        assert all(term.term != "frog" for term in renderings(align(source, target)))

    def test_a_term_below_the_occurrence_floor_is_not_scored(self) -> None:
        source, target = self._document(["amqerqur"] * 2)
        assert all(term.term != "frog" for term in renderings(align(source, target)))

    def test_no_pairs_yields_no_terms(self) -> None:
        assert renderings(align("", "")) == []


class TestCommonWords:
    def test_it_keeps_the_most_frequent_and_drops_the_rest(self) -> None:
        texts = ["the cat", "the dog", "the frog", "cat"]
        assert common_words(texts, keep=1) == frozenset({"the"})
        assert common_words(texts, keep=2) == frozenset({"the", "cat"})

    def test_an_empty_corpus_excludes_nothing(self) -> None:
        assert common_words([]) == frozenset()


class TestMeasure:
    def test_consistency_is_weighted_by_occurrences(self) -> None:
        """A term seen ten times must count for ten, or one stray rare term dominates."""
        frames = TestRenderings.FRAMES
        source = target = ""
        for n in range(8):
            english, kabyle = frames[n % len(frames)]
            source += f"The frog {english}.\n"
            target += f"Amqerqur {kabyle}.\n"
        for n, word in enumerate(["aslem", "aslem", "afrux", "afrux"]):
            english, kabyle = frames[(n + 2) % len(frames)]
            source += f"The heron {english}.\n"
            target += f"{word.capitalize()} {kabyle}.\n"
        source += "".join(f"A merchant walked to market number {n}.\n" for n in range(200))
        target += "".join(f"Amsaɣ yedda ɣer ssuq wis {n}.\n" for n in range(200))
        report = measure("fixture", source, target)
        terms = {term.term: term for term in report.terms}
        assert terms["frog"].consistency == 1.0
        assert terms["heron"].consistency == 0.5
        assert report.consistency == (8 + 2) / (8 + 4)

    def test_a_document_with_nothing_measurable_scores_zero_rather_than_failing(self) -> None:
        report = measure("empty", "", "")
        assert report.consistency == 0.0
        assert report.as_dict()["terms"] == 0

    def test_the_report_carries_the_skip_rate_it_was_measured_under(self) -> None:
        """A consistency over a third of a document is not a consistency over the document,
        and the number that says so has to travel with it.

        Whole words, not `A. B.` — a single capital before a full stop is an initial to the
        sentence splitter, so that fixture produced one unit and skipped nothing."""
        report = measure(
            "partial",
            "The cat slept. The dog ran.\nThe bird flew.\n",
            "Yeṭṭes umcic d weqjun yuzzel.\nAfrux yufeg.\n",
        )
        assert report.as_dict()["skip_rate"] == 0.5
