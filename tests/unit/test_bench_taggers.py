from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.bench.taggers import (
    UNKNOWN_REVISION,
    LexiconTagger,
    MostFrequentTagger,
    NeuralTagger,
    TaggerError,
    manifest_revision,
)
from agbalu.lexicon.analyser import Analyser
from agbalu.lexicon.models import Entry, Upos
from agbalu.treebank import Sentence, Token, Word

MODEL = Path("data/raw/hf.boffire.kabyle-pos-v2")


def word(identifier: int, form: str, upos: str) -> Word:
    return Word(
        id=identifier,
        form=form,
        lemma=form,
        upos=upos,
        feats="_",
        head=0,
        deprel="root",
        space_after=True,
    )


def sentence(*words: Word) -> Sentence:
    return Sentence(
        sent_id="s",
        text=" ".join(w.form for w in words),
        split="train",
        words=words,
        tokens=tuple(Token(form=w.form, space_after=True, words=(w,)) for w in words),
    )


def write_lexicon(path: Path) -> Path:
    """One row in the shape `read_lexicon` expects."""
    row = {
        "form": "argaz",
        "lemma": "argaz",
        "upos": "NOUN",
        "feats": "_",
        "source": "amawal",
        "licence": "cc-by-sa-4.0",
        "redistribution": "share-alike",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def entry(form: str, upos: Upos, source: str = "amawal") -> Entry:
    return Entry(
        form=form,
        lemma=form,
        upos=upos,
        features=(),
        glosses=(),
        source=source,
        licence="cc-by-sa-4.0",
        redistribution="share-alike",
    )


class TestMostFrequentTagger:
    def test_a_form_takes_its_most_frequent_gold_tag(self) -> None:
        train = [
            sentence(word(1, "d", "PART")),
            sentence(word(1, "d", "PART")),
            sentence(word(1, "d", "ADP")),
        ]
        assert MostFrequentTagger.fit(train).tag([["d"]]) == [["PART"]]

    def test_an_unknown_form_falls_back_to_the_majority_tag(self) -> None:
        train = [sentence(word(1, "a", "NOUN"), word(2, "b", "NOUN"), word(3, "c", "VERB"))]
        assert MostFrequentTagger.fit(train).tag([["zzz"]]) == [["NOUN"]]

    def test_lookup_falls_back_to_case_folding(self) -> None:
        train = [sentence(word(1, "argaz", "NOUN"), word(2, "x", "VERB"), word(3, "y", "VERB"))]
        assert MostFrequentTagger.fit(train).tag([["Argaz"]]) == [["NOUN"]]

    def test_a_tie_breaks_on_the_tag_name_so_refitting_is_stable(self) -> None:
        train = [sentence(word(1, "d", "PART")), sentence(word(1, "d", "ADP"))]
        assert MostFrequentTagger.fit(train).tag([["d"]]) == [["ADP"]]

    def test_it_answers_at_every_position(self) -> None:
        train = [sentence(word(1, "a", "NOUN"))]
        tagged = MostFrequentTagger.fit(train).tag([["a", "b"], []])
        assert tagged == [["NOUN", "NOUN"], []]

    def test_the_revision_names_the_splits_it_was_fitted_on(self) -> None:
        assert MostFrequentTagger.fit([sentence(word(1, "a", "NOUN"))]).revision == "fit:train"

    def test_fitting_on_nothing_is_an_error(self) -> None:
        with pytest.raises(TaggerError, match="nothing to fit"):
            MostFrequentTagger.fit([])


class TestLexiconTagger:
    def test_a_known_form_gets_its_unambiguous_tag(self) -> None:
        tagger = LexiconTagger(Analyser.from_entries([entry("argaz", "NOUN")]), "lex", "1.0")
        assert tagger.tag([["argaz"]]) == [["NOUN"]]

    def test_an_unknown_form_is_an_abstention_not_a_guess(self) -> None:
        tagger = LexiconTagger(Analyser.from_entries([entry("argaz", "NOUN")]), "lex", "1.0")
        assert tagger.tag([["zzz"]]) == [[None]]

    def test_an_ambiguous_form_within_one_source_abstains(self) -> None:
        entries = [entry("d", "PART"), entry("d", "ADP")]
        tagger = LexiconTagger(Analyser.from_entries(entries), "lex", "1.0")
        assert tagger.tag([["d"]]) == [[None]]

    def test_a_missing_lexicon_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(TaggerError, match="lexicon not found"):
            LexiconTagger.load(tmp_path / "absent.jsonl")

    def test_the_revision_comes_from_the_sibling_stats_file(self, tmp_path: Path) -> None:
        path = write_lexicon(tmp_path / "lex.jsonl")
        path.with_suffix(".stats.json").write_text(
            json.dumps({"normaliser_version": "1.2.0+rules1.0.0"}), encoding="utf-8"
        )
        assert LexiconTagger.load(path).revision == "1.2.0+rules1.0.0"

    def test_a_lexicon_without_stats_reports_an_unknown_revision(self, tmp_path: Path) -> None:
        assert (
            LexiconTagger.load(write_lexicon(tmp_path / "lex.jsonl")).revision == UNKNOWN_REVISION
        )


class TestManifestRevision:
    def test_a_source_with_no_manifest_rows_is_unknown(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
        assert manifest_revision("nothing.here", tmp_path) == UNKNOWN_REVISION

    @pytest.mark.integration
    def test_the_model_revision_is_the_commit_it_was_fetched_at(self) -> None:
        if not Path("data/raw/manifest.jsonl").is_file():
            pytest.skip("no acquisition manifest on this machine")
        revision = manifest_revision("hf.boffire.kabyle-pos-v2")
        assert revision != UNKNOWN_REVISION
        assert len(revision) == 40


@pytest.mark.integration
@pytest.mark.slow
class TestNeuralTagger:
    @pytest.fixture(scope="class")
    def tagger(self) -> NeuralTagger:
        if not MODEL.is_dir():
            pytest.skip(f"model not present under {MODEL}")
        return NeuralTagger.load(MODEL)

    def test_it_returns_one_label_per_input_word(self, tagger: NeuralTagger) -> None:
        batch = [["Azul", "fell-awen"], ["Yella", "wergaz", "deg", "wexxam", "."]]
        assert [len(row) for row in tagger.tag(batch)] == [2, 5]

    def test_every_label_is_one_the_model_declares(self, tagger: NeuralTagger) -> None:
        labels = set(tagger.id2label.values())
        tagged = tagger.tag([["Yella", "wergaz", "deg", "wexxam", "."]])
        assert {label for row in tagged for label in row} <= labels

    def test_punctuation_is_the_easy_case_it_should_get_right(self, tagger: NeuralTagger) -> None:
        assert tagger.tag([[".", "?", ","]]) == [["PUNCT", "PUNCT", "PUNCT"]]

    def test_an_empty_sentence_yields_no_labels(self, tagger: NeuralTagger) -> None:
        """It must not be dropped from the batch: that would shift every later
        sentence onto the wrong gold."""
        empty, one = tagger.tag([[], ["Azul"]])
        assert empty == []
        assert len(one) == 1
        assert one[0] is not None

    def test_input_order_survives_length_sorted_batching(self, tagger: NeuralTagger) -> None:
        """Batching sorts by length; a lost permutation would score every sentence
        against another sentence's gold and still look plausible."""
        one = ["Azul"]
        many = ["Yella", "wergaz", "deg", "wexxam", "n", "baba-s", "."]
        forward = tagger.tag([one, many])
        backward = tagger.tag([many, one])
        assert forward == [backward[1], backward[0]]

    def test_batch_size_does_not_change_the_answer(self, tagger: NeuralTagger) -> None:
        batch = [["Azul"], ["Yella", "wergaz"], ["deg", "wexxam", "."], ["Tura"]]
        small = NeuralTagger(
            MODEL, tagger.name, tagger.revision, batch_size=1, device=tagger.device
        )
        assert small.tag(batch) == tagger.tag(batch)

    def test_a_missing_model_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(TaggerError, match="model directory not found"):
            NeuralTagger.load(tmp_path / "absent")
