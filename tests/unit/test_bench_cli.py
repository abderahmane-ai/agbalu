from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from agbalu.bench.audit import audit
from agbalu.bench.cli import _write_audit, _write_contamination, _write_pos
from agbalu.bench.contamination import scan
from agbalu.bench.flores import Sentence
from agbalu.bench.pos import SETTINGS, items_for, run
from agbalu.normalise import NORMALISER_VERSION, Normaliser
from agbalu.treebank import Sentence as TreebankSentence
from agbalu.treebank import Token, Word

CLEAN = "Deg wass n Arim imusnawen seg uɣerbaz n tujjya berrḥen-d s usnulfu n wallal amaynut"
CORRUPT = "Ayɣer i teεyiḍ akk annect-a?"


def sentence(text: str, ident: int = 1, split: str = "dev") -> Sentence:
    return Sentence(
        id=ident,
        text=text,
        split=split,  # type: ignore[arg-type]
        domain="wikinews",
        topic="News",
        url="",
        last_updated="1.0",
        iso_639_3="kab",
        iso_15924="Latn",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_audit_summary_stamps_the_normaliser_that_produced_it(tmp_path: Path) -> None:
    """Without it, a corrected reference from 1.0.0 — which rewrote foreign proper
    nouns — is indistinguishable on disk from one produced by 1.1.0."""
    engine = Normaliser()
    sentences = [sentence(CORRUPT)]
    report = audit(sentences, "kab_Latn", engine)
    _, _, summary = _write_audit(report, sentences, engine.version, tmp_path)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["normaliser_version"] == engine.version
    assert payload["normaliser_version"].startswith(f"{NORMALISER_VERSION}+rules")


def test_audit_writes_every_sentence_and_flags_only_the_changed(tmp_path: Path) -> None:
    engine = Normaliser()
    sentences = [sentence(CLEAN, 1), sentence(CORRUPT, 2)]
    report = audit(sentences, "kab_Latn", engine)
    corrected, diff, summary = _write_audit(report, sentences, engine.version, tmp_path)

    rows = read_jsonl(corrected)
    assert len(rows) == 2
    assert [r["corrected"] for r in rows] == [False, True]
    assert rows[0]["text"] == CLEAN
    repaired = str(rows[1]["text"])
    assert "ɛ" in repaired
    assert "ε" not in repaired

    assert len(read_jsonl(diff)) == 1
    assert json.loads(summary.read_text(encoding="utf-8"))["changed"] == 1


def test_audit_keeps_ids_from_two_splits_apart(tmp_path: Path) -> None:
    """FLORES+ ids restart at 0 per split; keying on `id` alone corrects the wrong rows."""
    engine = Normaliser()
    sentences = [sentence(CORRUPT, 7, "dev"), sentence(CLEAN, 7, "devtest")]
    report = audit(sentences, "kab_Latn", engine)
    corrected, _, _ = _write_audit(report, sentences, engine.version, tmp_path)

    rows = read_jsonl(corrected)
    assert [(r["split"], r["corrected"]) for r in rows] == [("dev", True), ("devtest", False)]


def test_audit_summary_names_codepoints_unambiguously(tmp_path: Path) -> None:
    """A bare `ε` key in a JSON report is itself ambiguous."""
    engine = Normaliser()
    sentences = [sentence(CORRUPT)]
    report = audit(sentences, "kab_Latn", engine)
    _, _, summary = _write_audit(report, sentences, engine.version, tmp_path)

    keys = list(json.loads(summary.read_text(encoding="utf-8"))["by_codepoint"])
    assert keys == ["ε U+03B5 GREEK SMALL LETTER EPSILON"]


def test_audit_of_a_clean_language_still_writes_all_three_files(tmp_path: Path) -> None:
    engine = Normaliser()
    sentences = [sentence(CLEAN)]
    report = audit(sentences, "kab_Latn", engine)
    corrected, diff, summary = _write_audit(report, sentences, engine.version, tmp_path)

    assert len(read_jsonl(corrected)) == 1
    assert read_jsonl(diff) == []
    assert json.loads(summary.read_text(encoding="utf-8"))["changed"] == 0


def test_contamination_summary_exists_when_nothing_leaked(tmp_path: Path) -> None:
    """An empty detail file looks like a run that never happened; the summary is the
    proof the corpus was scanned."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"text": "Azul fell-awen", "source": "s"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    engine = Normaliser()
    report = scan(corpus, [sentence(CLEAN)], engine)
    detail, summary = _write_contamination(report, "kab_Latn", corpus, engine.version, tmp_path)

    assert detail.read_text(encoding="utf-8") == ""
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["leaked"] == 0
    assert payload["corpus_lines"] == 1
    assert payload["benchmark_sentences"] == 1
    assert payload["normaliser_version"] == engine.version
    assert payload["corpus"] == str(corpus)


def test_contamination_summary_counts_a_real_leak(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"text": CLEAN, "source": "hf.some.source"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    engine = Normaliser()
    report = scan(corpus, [sentence(CLEAN)], engine)
    detail, summary = _write_contamination(report, "kab_Latn", corpus, engine.version, tmp_path)

    assert len(read_jsonl(detail)) == 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["leaked"] == 1
    assert payload["leak_rate"] == 1.0
    assert payload["by_source"] == {"hf.some.source": 1}


def treebank_sentence() -> TreebankSentence:
    words = (
        Word(
            id=1,
            form="ɣur",
            lemma="ɣur",
            upos="ADP",
            feats="_",
            head=3,
            deprel="case",
            space_after=True,
        ),
        Word(
            id=2,
            form="s",
            lemma="netta",
            upos="PRON",
            feats="_",
            head=1,
            deprel="nmod",
            space_after=True,
        ),
        Word(
            id=3,
            form="aksum",
            lemma="aksum",
            upos="NOUN",
            feats="_",
            head=0,
            deprel="root",
            space_after=True,
        ),
    )
    return TreebankSentence(
        sent_id="s1",
        text="ɣur-s aksum",
        split="test",
        words=words,
        tokens=(
            Token(form="ɣur-s", space_after=True, words=words[:2]),
            Token(form="aksum", space_after=True, words=words[2:]),
        ),
    )


class Constant:
    """Answers NOUN everywhere, so the payload is checked and not the tagger."""

    @property
    def name(self) -> str:
        return "constant"

    @property
    def revision(self) -> str:
        return "r1"

    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]:
        return [["NOUN"] * len(s) for s in sentences]


def pos_args(tmp_path: Path, split: str = "test") -> argparse.Namespace:
    return argparse.Namespace(
        treebank=Path("data/raw/git.ud-kabyle-adpt"), split=split, out=tmp_path
    )


def test_pos_summary_reports_both_views_of_every_run(tmp_path: Path) -> None:
    """A result quoted with punctuation included and one without are different
    numbers; the file has to carry both or the reader picks one at random."""
    sentences = [treebank_sentence()]
    items = items_for(sentences, "surface")
    runs = [run(Constant(), items, "surface", "canonical")]
    summary, detail = _write_pos(runs, pos_args(tmp_path), sentences, "1.2.0+rules1.0.0")

    payload = json.loads(summary.read_text(encoding="utf-8"))
    [entry] = payload["runs"]
    assert entry["all_tags"]["scored"] == 1
    assert entry["without_punct"]["scored"] == 1
    assert entry["system"] == "constant"
    assert entry["revision"] == "r1"
    assert payload["normaliser_version"] == "1.2.0+rules1.0.0"
    assert len(read_jsonl(detail)) == 1


def test_pos_summary_records_the_segmentation_that_cannot_be_scored(tmp_path: Path) -> None:
    sentences = [treebank_sentence()]
    runs = [run(Constant(), items_for(sentences, "surface"), "surface", "canonical")]
    summary, _ = _write_pos(runs, pos_args(tmp_path), sentences, "v")

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["words"] == 3
    assert payload["surface_tokens"] == 2
    assert payload["multiword_words"] == 2
    assert payload["multiword_rate"] == round(2 / 3, 6)
    assert payload["runs"][0]["unscorable"] == 1


def test_pos_predictions_carry_the_run_they_came_from(tmp_path: Path) -> None:
    sentences = [treebank_sentence()]
    runs = [
        run(Constant(), items_for(sentences, setting), setting, "canonical") for setting in SETTINGS
    ]
    _, detail = _write_pos(runs, pos_args(tmp_path), sentences, "v")

    rows = read_jsonl(detail)
    assert {row["setting"] for row in rows} == {"surface", "gold-words"}
    assert {row["system"] for row in rows} == {"constant"}
    assert sum(1 for row in rows if row["setting"] == "gold-words") == 3
