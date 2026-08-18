from __future__ import annotations

from collections.abc import Sequence

import pytest

from agbalu.bench.pos import (
    ABSTAIN,
    CORRUPTIONS,
    UNANNOTATED,
    Item,
    PosScoringError,
    Run,
    Unit,
    corrupt,
    items_for,
    run,
    score,
)
from agbalu.normalise.rules import load_rules
from agbalu.treebank import Sentence, Token, Word


def word(
    identifier: int, form: str, upos: str, *, lemma: str = "_", space_after: bool = True
) -> Word:
    return Word(
        id=identifier,
        form=form,
        lemma=lemma,
        upos=upos,
        feats="_",
        head=0,
        deprel="root",
        space_after=space_after,
    )


def single(item: Word) -> Token:
    return Token(form=item.form, space_after=item.space_after, words=(item,))


def sentence(
    *words: Word, sent_id: str = "s1", tokens: tuple[Token, ...] | None = None
) -> Sentence:
    return Sentence(
        sent_id=sent_id,
        text=" ".join(w.form for w in words),
        split="test",
        words=words,
        tokens=tokens if tokens is not None else tuple(single(w) for w in words),
    )


class Fixed:
    """Replays a canned answer per input token, so scoring is tested on its own."""

    def __init__(self, answers: list[list[str | None]], name: str = "fixed") -> None:
        self.answers = answers
        self._name = name
        self.seen: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return "r1"

    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]:
        self.seen = [list(s) for s in sentences]
        return self.answers


SIMPLE = sentence(word(1, "Azul", "INTJ"), word(2, "aɣerbaz", "NOUN"), word(3, ".", "PUNCT"))

CLITIC = sentence(
    word(1, "ɣur", "ADP"),
    word(2, "s", "PRON"),
    word(3, "aksum", "NOUN"),
    sent_id="s2",
    tokens=(
        Token(form="ɣur-s", space_after=True, words=(word(1, "ɣur", "ADP"), word(2, "s", "PRON"))),
        single(word(3, "aksum", "NOUN")),
    ),
)


class TestCorrupt:
    def test_canonical_letters_become_their_homoglyphs(self) -> None:
        assert corrupt("aɣerbaz teɛyiḍ", load_rules()) == "aγerbaz teεyiḍ"

    def test_capitals_are_covered(self) -> None:
        assert corrupt("Ɣef Ɛli", load_rules()) == "Γef Σli"

    def test_text_without_the_letters_is_untouched(self) -> None:
        assert corrupt("Azul fell-ak", load_rules()) == "Azul fell-ak"

    def test_an_empty_string_survives(self) -> None:
        assert corrupt("", load_rules()) == ""

    def test_every_corruption_is_one_the_normaliser_undoes(self) -> None:
        """Otherwise the corrupted condition measures an unrelated perturbation."""
        homoglyphs = load_rules().homoglyphs
        assert {homoglyphs[bad] for bad in CORRUPTIONS.values()} == set(CORRUPTIONS)

    def test_a_rules_table_that_does_not_reverse_the_change_is_an_error(self) -> None:
        rules = load_rules().model_copy(update={"homoglyphs": {}})
        with pytest.raises(PosScoringError, match="does not fold"):
            corrupt("aɣerbaz", rules)


class TestItems:
    def test_gold_words_gives_one_position_per_syntactic_word(self) -> None:
        [item] = items_for([CLITIC], "gold-words")
        assert item.forms() == ("ɣur", "s", "aksum")
        assert [unit.gold for unit in item.units] == ["ADP", "PRON", "NOUN"]

    def test_surface_gives_one_position_per_token(self) -> None:
        [item] = items_for([CLITIC], "surface")
        assert item.forms() == ("ɣur-s", "aksum")

    def test_a_multiword_token_has_no_single_gold_tag(self) -> None:
        [item] = items_for([CLITIC], "surface")
        assert [unit.gold for unit in item.units] == [None, "NOUN"]

    def test_the_corrupted_condition_rewrites_the_input_only(self) -> None:
        [canonical] = items_for([CLITIC], "surface", "canonical")
        [corrupted] = items_for([CLITIC], "surface", "corrupted", load_rules())
        assert corrupted.forms() == ("γur-s", "aksum")
        assert [u.gold for u in corrupted.units] == [u.gold for u in canonical.units]

    def test_the_corrupted_condition_needs_the_rules(self) -> None:
        with pytest.raises(PosScoringError, match="needs the homoglyph rules"):
            items_for([CLITIC], "surface", "corrupted")

    def test_no_sentences_gives_no_items(self) -> None:
        assert items_for([], "surface") == []


class TestRun:
    def test_every_scorable_position_becomes_a_prediction(self) -> None:
        items = items_for([SIMPLE], "gold-words")
        result = run(Fixed([["INTJ", "NOUN", "PUNCT"]]), items, "gold-words", "canonical")
        assert result.units == 3
        assert result.unscorable == 0
        assert [p.predicted for p in result.predictions] == ["INTJ", "NOUN", "PUNCT"]

    def test_the_tagger_is_asked_about_every_position(self) -> None:
        tagger = Fixed([["INTJ", "NOUN", "PUNCT"]])
        run(tagger, items_for([SIMPLE], "gold-words"), "gold-words", "canonical")
        assert tagger.seen == [["Azul", "aɣerbaz", "."]]

    def test_a_multiword_token_is_counted_unscorable_not_wrong(self) -> None:
        items = items_for([CLITIC], "surface")
        result = run(Fixed([["ADP", "NOUN"]]), items, "surface", "canonical")
        assert (result.units, result.unscorable) == (2, 1)
        assert result.unscorable_rate == pytest.approx(0.5)
        assert [p.form for p in result.predictions] == ["aksum"]

    def test_a_word_the_treebank_left_untagged_is_unscorable(self) -> None:
        untagged = sentence(word(1, "???", UNANNOTATED))
        result = run(
            Fixed([["NOUN"]]), items_for([untagged], "gold-words"), "gold-words", "canonical"
        )
        assert (result.units, result.unscorable, result.predictions) == (1, 1, ())

    def test_a_wrong_sentence_count_is_an_error(self) -> None:
        items = items_for([SIMPLE], "gold-words")
        with pytest.raises(PosScoringError, match="returned 2 sentences for 1"):
            run(Fixed([[], []]), items, "gold-words", "canonical")

    def test_a_wrong_label_count_is_an_error(self) -> None:
        """Scoring the overlap would give a plausible number from misaligned output."""
        items = items_for([SIMPLE], "gold-words")
        with pytest.raises(PosScoringError, match="returned 2 labels"):
            run(Fixed([["INTJ", "NOUN"]]), items, "gold-words", "canonical")

    def test_the_run_carries_the_system_identity(self) -> None:
        items = items_for([SIMPLE], "gold-words")
        result = run(
            Fixed([["INTJ", "NOUN", "PUNCT"]], name="sys"), items, "gold-words", "corrupted"
        )
        assert (result.tagger, result.revision, result.condition) == ("sys", "r1", "corrupted")

    def test_an_empty_run_has_no_unscorable_rate(self) -> None:
        assert run(Fixed([]), [], "surface", "canonical").unscorable_rate == 0.0


class TestScore:
    def _run(self, answers: list[str | None]) -> Run:
        return run(Fixed([answers]), items_for([SIMPLE], "gold-words"), "gold-words", "canonical")

    def test_all_correct_is_perfect(self) -> None:
        result = score(self._run(["INTJ", "NOUN", "PUNCT"]))
        assert (result.accuracy, result.coverage, result.macro_f1) == (1.0, 1.0, 1.0)

    def test_abstention_counts_as_wrong_but_not_as_an_answer(self) -> None:
        result = score(self._run(["INTJ", None, "PUNCT"]))
        assert result.accuracy == pytest.approx(2 / 3)
        assert result.coverage == pytest.approx(2 / 3)
        assert result.accuracy_when_answered == 1.0

    def test_abstention_is_recorded_against_the_gold_label(self) -> None:
        result = score(self._run(["INTJ", None, "PUNCT"]))
        assert result.confusions[("NOUN", ABSTAIN)] == 1

    def test_ignoring_a_label_removes_it_from_every_denominator(self) -> None:
        result = score(self._run(["INTJ", "NOUN", "VERB"]), ignore=frozenset({"PUNCT"}))
        assert (result.scored, result.accuracy) == (2, 1.0)
        assert "PUNCT" not in result.labels

    def test_precision_and_recall_are_computed_per_label(self) -> None:
        result = score(self._run(["NOUN", "NOUN", "PUNCT"]))
        noun = result.label_score("NOUN")
        assert (noun.support, noun.predicted) == (1, 2)
        assert (noun.precision, noun.recall, noun.f1) == pytest.approx((0.5, 1.0, 2 / 3))

    def test_a_label_the_system_never_predicts_scores_zero(self) -> None:
        result = score(self._run(["NOUN", "NOUN", "PUNCT"]))
        intj = result.label_score("INTJ")
        assert (intj.precision, intj.recall, intj.f1) == (0.0, 0.0, 0.0)

    def test_macro_f1_averages_over_labels_with_gold_support(self) -> None:
        result = score(self._run(["NOUN", "NOUN", "PUNCT"]))
        assert set(result.labels) == {"INTJ", "NOUN", "PUNCT"}
        assert result.macro_f1 == pytest.approx((0.0 + 2 / 3 + 1.0) / 3)

    def test_a_label_predicted_but_never_gold_is_not_averaged_over(self) -> None:
        result = score(self._run(["SYM", "NOUN", "PUNCT"]))
        assert "SYM" not in result.labels
        assert result.accuracy == pytest.approx(2 / 3)

    def test_labels_come_back_most_frequent_first(self) -> None:
        pair = sentence(word(1, "a", "NOUN"), word(2, "b", "NOUN"), word(3, "c", "VERB"))
        result = score(
            run(
                Fixed([["NOUN", "NOUN", "VERB"]]),
                items_for([pair], "gold-words"),
                "gold-words",
                "canonical",
            )
        )
        assert result.labels == ("NOUN", "VERB")

    def test_confusions_exclude_the_diagonal(self) -> None:
        result = score(self._run(["NOUN", "VERB", "PUNCT"]))
        assert result.top_confusions(5) == (("INTJ", "NOUN", 1), ("NOUN", "VERB", 1))

    def test_an_empty_score_divides_by_nothing(self) -> None:
        result = score(run(Fixed([]), [], "surface", "canonical"))
        assert (result.accuracy, result.coverage, result.accuracy_when_answered) == (0.0, 0.0, 0.0)
        assert (result.macro_f1, result.labels, result.label_scores()) == (0.0, (), ())

    def test_a_run_where_everything_is_unscorable_scores_nothing(self) -> None:
        items = [Item(sent_id="s", split="test", units=(Unit(form="x", gold=None),))]
        result = score(run(Fixed([["NOUN"]]), items, "surface", "canonical"))
        assert result.scored == 0
