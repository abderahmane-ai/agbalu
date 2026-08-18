from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.g2p.align import (
    Alignment,
    G2PError,
    Pronunciations,
    align,
    orthographic_tokens,
    read_pairs,
)
from agbalu.lexicon.analyser import Analyser
from agbalu.lexicon.coverage import scan, tokenise, totals, unknown_types
from agbalu.lexicon.models import Entry, LexiconError, Upos, features_of
from agbalu.lexicon.state import free_candidates
from agbalu.lexicon.validate import score

HUNSPELL = "hf.boffire.hunspell-kab"
VERBS = "hf.boffire.kabyle-verbs"
TOPONYMS = "hf.boffire.kabyle-toponyms"


def entry(
    form: str,
    lemma: str | None = None,
    upos: Upos | None = None,
    source: str = HUNSPELL,
    **features: str,
) -> Entry:
    return Entry(
        form=form,
        lemma=lemma,
        upos=upos,
        features=features_of(**features),
        glosses=(),
        source=source,
        licence="mit",
        redistribution="permissive",
    )


class TestRoutes:
    def test_an_exact_hit_wins(self) -> None:
        analyser = Analyser.from_entries([entry("axxam", "axxam", "NOUN")])
        analyses = analyser.analyse("axxam")
        assert [a.route for a in analyses] == ["exact"]
        assert analyses[0].lemma == "axxam"

    def test_a_clitic_is_stripped_to_its_head(self) -> None:
        """`axxam-nneɣ` is one orthographic word; the hyphen is the boundary."""
        analyser = Analyser.from_entries([entry("axxam", "axxam", "NOUN")])
        analyses = analyser.analyse("axxam-nneɣ")
        assert [a.route for a in analyses] == ["clitic"]
        assert analyses[0].lemma == "axxam"

    def test_a_one_character_head_is_not_treated_as_the_word(self) -> None:
        """`d-yusa` opens with a particle, not a lexeme."""
        analyser = Analyser.from_entries([entry("d")])
        assert analyser.analyse("d-yusa") == ()

    def test_a_sentence_initial_capital_is_folded(self) -> None:
        analyser = Analyser.from_entries([entry("azul", "azul")])
        assert [a.route for a in analyser.analyse("Azul")] == ["casefold"]

    def test_the_annexed_state_falls_back_to_the_free_form(self) -> None:
        analyser = Analyser.from_entries([entry("axxam", "axxam", "NOUN")])
        analyses = analyser.analyse("wexxam")
        assert [a.route for a in analyses] == ["state"]
        assert analyses[0].lemma == "axxam"

    def test_an_unknown_form_gets_no_guess(self) -> None:
        analyser = Analyser.from_entries([entry("axxam")])
        assert analyser.analyse("qqqqqq") == ()
        assert analyser.upos("qqqqqq") is None

    def test_an_empty_form_is_not_an_error(self) -> None:
        assert Analyser.from_entries([entry("a")]).analyse("") == ()

    def test_a_direct_route_is_not_diluted_by_a_speculative_one(self) -> None:
        """`wexxam` exists in its own right, so the state hypothesis must not be added."""
        analyser = Analyser.from_entries(
            [entry("wexxam", "axxam", "NOUN", State="Cons"), entry("axxam", "axxam", "NOUN")]
        )
        analyses = analyser.analyse("wexxam")
        assert [a.route for a in analyses] == ["exact"]


class TestPosDecision:
    def test_a_curated_tag_beats_many_gazetteer_tags(self) -> None:
        """Two OSM roads named `D` had outvoted a hand-annotated dictionary."""
        analyser = Analyser.from_entries(
            [
                entry("d", upos="PART", source=HUNSPELL),
                entry("d", "D", "PROPN", source=TOPONYMS),
                entry("d", "D", "PROPN", source=TOPONYMS),
            ]
        )
        assert analyser.upos("d") == "PART"

    def test_a_silent_curated_source_does_not_veto_a_lower_tier(self) -> None:
        """Hunspell's 12,348 unlabelled entries are silent through incompleteness, so
        silence must not suppress a tag another source has."""
        analyser = Analyser.from_entries(
            [
                entry("qsemṭina", upos=None, source=HUNSPELL),
                entry("qsemṭina", "Qsemṭina", "PROPN", source=TOPONYMS),
            ]
        )
        assert analyser.upos("qsemṭina") == "PROPN"

    def test_a_curated_tag_beats_the_verb_table(self) -> None:
        analyser = Analyser.from_entries(
            [entry("ur", upos="ADV", source=HUNSPELL)]
            + [entry("ur", "ur", "VERB", source=VERBS) for _ in range(50)]
        )
        assert analyser.upos("ur") == "ADV"

    def test_a_tie_inside_one_tier_returns_nothing(self) -> None:
        analyser = Analyser.from_entries(
            [entry("x", upos="NOUN", source=HUNSPELL), entry("x", upos="VERB", source=HUNSPELL)]
        )
        assert analyser.upos("x") is None

    def test_a_majority_inside_one_tier_wins(self) -> None:
        analyser = Analyser.from_entries(
            [
                entry("x", upos="NOUN", source=HUNSPELL),
                entry("x", upos="NOUN", source=HUNSPELL, Number="Plur"),
                entry("x", upos="VERB", source=HUNSPELL),
            ]
        )
        assert analyser.upos("x") == "NOUN"

    def test_lemmas_are_distinct_and_ordered(self) -> None:
        analyser = Analyser.from_entries(
            [entry("deg", "deg"), entry("deg", "ddu"), entry("deg", "deg")]
        )
        assert analyser.lemmas("deg") == ("deg", "ddu")


class TestState:
    @pytest.mark.parametrize(
        ("annexed", "free"),
        [
            ("wexxam", "axxam"),
            ("wazal", "azal"),
            ("umawal", "amawal"),
            ("yiseggasen", "iseggasen"),
            ("yemdanen", "imdanen"),
            ("wuzzal", "uzzal"),
        ],
    )
    def test_attested_alternations_are_reversed(self, annexed: str, free: str) -> None:
        assert free in set(free_candidates(annexed))

    def test_the_feminine_prefix_offers_both_sources(self) -> None:
        """`t-` collapses `ta-` and `ti-`, so both must be offered."""
        assert {"tamurt", "timurt"} <= set(free_candidates("tmurt"))

    def test_a_candidate_is_never_the_form_itself(self) -> None:
        assert all(c != "uzzal" for c in free_candidates("uzzal"))

    def test_an_empty_stem_yields_nothing(self) -> None:
        assert list(free_candidates("we")) == []
        assert list(free_candidates("")) == []

    def test_candidates_are_not_repeated(self) -> None:
        candidates = list(free_candidates("tmurt"))
        assert len(candidates) == len(set(candidates))


class TestCoverage:
    def corpus(self, tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
        path = tmp_path / "corpus.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for text, source in rows:
                row = {"text": text, "source": source}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_hyphenated_clitics_stay_one_token(self) -> None:
        """Splitting here would hide the clitic problem the analyser exists to solve."""
        assert tokenise("Yenna-yas axxam-nneɣ.") == ["Yenna-yas", "axxam-nneɣ"]

    def test_digits_are_not_tokens(self) -> None:
        assert tokenise("deg 1955 n") == ["deg", "n"]

    def test_coverage_is_measured_per_source(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        path = self.corpus(tmp_path, [("azul", "good"), ("qqq www", "bad")])
        per_source = scan(path, analyser)
        assert per_source["good"].token_coverage == 1.0
        assert per_source["bad"].token_coverage == 0.0

    def test_the_limit_applies_per_source_not_overall(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        rows = [("azul", "a")] * 5 + [("azul", "b")] * 5
        per_source = scan(self.corpus(tmp_path, rows), analyser, limit=2)
        assert per_source["a"].lines == 2
        assert per_source["b"].lines == 2

    def test_type_coverage_counts_each_word_once(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        path = self.corpus(tmp_path, [("azul azul azul qqq", "s")])
        report = scan(path, analyser)["s"]
        assert report.token_coverage == pytest.approx(0.75)
        assert report.type_coverage == pytest.approx(0.5)

    def test_totals_union_the_types_rather_than_summing_them(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        path = self.corpus(tmp_path, [("azul", "a"), ("azul", "b")])
        combined = totals(scan(path, analyser))
        assert combined.types == {"azul"}
        assert combined.tokens == 2

    def test_an_empty_corpus_reports_zero_not_a_crash(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        path = tmp_path / "corpus.jsonl"
        path.write_text("", encoding="utf-8")
        assert scan(path, analyser) == {}

    def test_a_missing_corpus_is_an_error(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        with pytest.raises(LexiconError, match="corpus not found"):
            scan(tmp_path / "absent.jsonl", analyser)

    def test_frequent_unknowns_are_ranked(self, tmp_path: Path) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        path = self.corpus(tmp_path, [("azul qqq qqq www", "s")])
        assert unknown_types(path, analyser, top=2) == [("qqq", 2), ("www", 1)]


class TestValidate:
    def test_punctuation_is_excluded_from_scoring(self) -> None:
        analyser = Analyser.from_entries([entry("azul", "azul", "NOUN")])
        report = score(analyser, [("azul", "azul", "NOUN"), (".", ".", "PUNCT")])
        assert report.tokens == 2
        assert report.scored == 1

    def test_agreement_is_over_unambiguous_predictions_only(self) -> None:
        analyser = Analyser.from_entries(
            [entry("x", upos="NOUN"), entry("x", upos="VERB"), entry("y", upos="NOUN")]
        )
        report = score(analyser, [("x", "x", "NOUN"), ("y", "y", "NOUN")])
        assert report.known == 2
        assert report.unambiguous == 1
        assert report.accuracy == 1.0

    def test_an_unknown_form_lowers_coverage_but_not_accuracy(self) -> None:
        analyser = Analyser.from_entries([entry("azul", "azul", "NOUN")])
        report = score(analyser, [("azul", "azul", "NOUN"), ("qqq", "qqq", "NOUN")])
        assert report.coverage == 0.5
        assert report.accuracy == 1.0

    def test_confusions_record_the_direction(self) -> None:
        analyser = Analyser.from_entries([entry("i", upos="NOUN")])
        report = score(analyser, [("i", "i", "ADP")])
        assert report.confusions[("ADP", "NOUN")] == 1

    def test_an_empty_treebank_gives_zero_not_a_division_error(self) -> None:
        analyser = Analyser.from_entries([entry("azul")])
        report = score(analyser, [])
        assert report.coverage == 0.0
        assert report.accuracy == 0.0
        assert report.lemma_agreement == 0.0


class TestG2PAlignment:
    def test_the_ipa_side_splits_clitics_so_the_orthography_must_too(self) -> None:
        """46.4% of sentences align on whitespace; 99.0% once hyphens split."""
        assert orthographic_tokens("Tanarit-a inu.") == ["Tanarit", "a", "inu"]

    def test_punctuation_is_stripped_from_tokens(self) -> None:
        assert orthographic_tokens("Bḍan.") == ["Bḍan"]

    def test_a_sentence_that_still_disagrees_is_dropped_not_guessed(self) -> None:
        pairs = iter([("a b c", "x y"), ("Tanarit-a inu", "θænæriθ æ inu")])
        _, report = align(pairs)
        assert report.sentences == 2
        assert report.aligned == 1
        assert report.skipped[1] == 1

    def test_pronunciations_are_keyed_case_insensitively(self) -> None:
        aligned = [Alignment(("Azul",), ("æzul",)), Alignment(("azul",), ("æzul",))]
        lexicon = Pronunciations.from_alignments(aligned)
        assert len(lexicon) == 1
        assert lexicon.best("AZUL") == "æzul"

    def test_an_unattested_word_has_no_pronunciation(self) -> None:
        lexicon = Pronunciations.from_alignments([Alignment(("azul",), ("æzul",))])
        assert lexicon.best("qqq") is None

    def test_a_word_with_two_transcriptions_is_reported_as_ambiguous(self) -> None:
        aligned = [Alignment(("tala",), ("θælæ",)), Alignment(("tala",), ("tælæ",))]
        lexicon = Pronunciations.from_alignments(aligned)
        assert set(lexicon.ambiguous()) == {"tala"}
        assert lexicon.best("tala") in {"θælæ", "tælæ"}

    def test_a_missing_source_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(G2PError, match="pronunciation data not found"):
            list(read_pairs(tmp_path / "absent.tsv"))
