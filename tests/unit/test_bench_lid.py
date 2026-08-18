from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.bench.lid import (
    EVAL_LANGUAGES,
    KABYLE,
    ConfusionMatrix,
    Example,
    LidError,
    SourceAudit,
    audit_sources,
    build_evalset,
    candidates,
    is_usable,
    kabyle_side,
    latin_share,
    read_flores,
    score,
    sentences,
    strip_wikitext,
)


class StubIdentifier:
    """Returns a scripted label per input, in order."""

    def __init__(self, labels: Sequence[str], known: Sequence[str]) -> None:
        self._labels = list(labels)
        self._known = frozenset(known)

    @property
    def name(self) -> str:
        return "stub"

    @property
    def revision(self) -> str:
        return "0" * 12

    @property
    def labels(self) -> frozenset[str]:
        return self._known

    def identify(self, texts: Sequence[str]) -> list[str]:
        return self._labels[: len(texts)]


class ByPrefixIdentifier:
    """Labels each input from its own text, so the result is call-order independent."""

    def __init__(self, known: Sequence[str]) -> None:
        self._known = frozenset(known)

    @property
    def name(self) -> str:
        return "by-prefix"

    @property
    def revision(self) -> str:
        return "1" * 12

    @property
    def labels(self) -> frozenset[str]:
        return self._known

    def identify(self, texts: Sequence[str]) -> list[str]:
        return ["kab_Latn" if t.startswith("clean") else "rif_Latn" for t in texts]


def test_latin_share_of_empty_and_digit_only_text() -> None:
    assert latin_share("") == 0.0
    assert latin_share("12345 …") == 0.0


def test_latin_share_counts_berber_latin_letters_as_latin() -> None:
    assert latin_share("aɣbalu ameqqran") == 1.0
    assert latin_share("taqbaylit ḥ ḍ ṣ ṭ ẓ ṛ č ǧ") == 1.0


def test_latin_share_falls_with_tifinagh() -> None:
    assert latin_share("ⵜⴰⵎⴰⵣⵉⵖⵜ") == 0.0
    assert 0.0 < latin_share("abc ⵜⴰⵎ") < 1.0


def test_is_usable_rejects_short_and_overlong_lines() -> None:
    assert not is_usable("too short here")
    assert is_usable(" ".join(["awal"] * 5))
    assert not is_usable(" ".join(["awal"] * 500))


def test_is_usable_rejects_wiki_residue() -> None:
    assert not is_usable("vignette Aksel Aksel Taggayt:Imudaren s tacawit")
    assert not is_usable("d awal {{template}} n tutlayt tamaziɣt taqbaylit")
    assert not is_usable("see https://example.invalid for more of this text")


def test_is_usable_rejects_a_tifinagh_line() -> None:
    assert not is_usable("ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⴰⵡⴰⵢⵜ ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⴰⵡⴰⵢⵜ ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⴰⵡⴰⵢⵜ")


def test_sentences_splits_on_terminators_and_newlines() -> None:
    assert list(sentences("Amek i telliḍ? Labas. \n\n Ih ihi!")) == [
        "Amek i telliḍ?",
        "Labas.",
        "Ih ihi!",
    ]


def test_sentences_collapses_whitespace_and_drops_empties() -> None:
    assert list(sentences("  a\t\tb  \n\n\n   ")) == ["a b"]
    assert list(sentences("")) == []


def test_strip_wikitext_removes_templates_and_links() -> None:
    raw = "{{S|awal}} '''Aksel''' [[Afaylu:x.jpg|vignette]] d [[awal]] <ref>x</ref>"
    stripped = strip_wikitext(raw)
    assert "{{" not in stripped
    assert "[[" not in stripped
    assert "<ref>" not in stripped
    assert "awal" in stripped


def test_candidates_deduplicates_and_preserves_order() -> None:
    line = " ".join(["awal"] * 6)
    other = " ".join(["adlis"] * 6)
    assert candidates([f"{line}. {other}.", f"{line}."]) == [f"{line}.", f"{other}."]


def test_build_evalset_is_balanced_and_deterministic() -> None:
    pools = {lang: [f"{lang} line {i}" for i in range(50)] for lang in ("kab_Latn", "shi_Latn")}
    first = build_evalset(pools, per_language=10, seed=7)
    second = build_evalset(pools, per_language=10, seed=7)
    assert first == second
    assert len(first) == 20
    assert sum(1 for x in first if x.language == "kab_Latn") == 10


def test_build_evalset_changes_with_the_seed() -> None:
    pools = {"kab_Latn": [f"line {i}" for i in range(200)]}
    assert build_evalset(pools, per_language=20, seed=1) != build_evalset(
        pools, per_language=20, seed=2
    )


def test_build_evalset_refuses_an_unbalanced_draw() -> None:
    pools = {"kab_Latn": ["a"] * 10, "tzm_Latn": ["b"] * 3}
    with pytest.raises(LidError, match="tzm_Latn has 3"):
        build_evalset(pools, per_language=10, seed=1)


def test_build_evalset_refuses_a_non_positive_draw() -> None:
    with pytest.raises(LidError, match="must be positive"):
        build_evalset({"kab_Latn": ["a"]}, per_language=0, seed=1)


def matrix_of(pairs: dict[tuple[str, str], int], known: Sequence[str]) -> ConfusionMatrix:
    labels = tuple(x for x in EVAL_LANGUAGES if any(t == x for t, _ in pairs))
    return ConfusionMatrix(labels=labels, counts=pairs, system_labels=frozenset(known))


def test_perfect_matrix_scores_one() -> None:
    matrix = matrix_of(
        {("kab_Latn", "kab_Latn"): 10, ("shi_Latn", "shi_Latn"): 10},
        known=["kab_Latn", "shi_Latn"],
    )
    assert matrix.accuracy() == 1.0
    assert matrix.macro_f1() == 1.0
    assert matrix.absorption("shi_Latn") == 0.0


def test_macro_f1_ignores_languages_the_system_cannot_name() -> None:
    # rif is absent from the label set, so its recall is zero by construction.
    matrix = matrix_of(
        {("kab_Latn", "kab_Latn"): 10, ("rif_Latn", "kab_Latn"): 10},
        known=["kab_Latn"],
    )
    assert matrix.unnameable == ("rif_Latn",)
    assert matrix.discriminable == ("kab_Latn",)
    # Kabyle recall is perfect; precision is halved by the absorbed Tarifit.
    assert matrix.recall("kab_Latn") == 1.0
    assert matrix.precision("kab_Latn") == 0.5
    assert matrix.macro_f1() == pytest.approx(2 / 3)


def test_absorption_measures_the_share_sent_to_kabyle() -> None:
    matrix = matrix_of(
        {("tzm_Latn", "kab_Latn"): 9, ("tzm_Latn", "fra_Latn"): 1},
        known=["kab_Latn", "fra_Latn"],
    )
    assert matrix.absorption("tzm_Latn") == pytest.approx(0.9)
    assert matrix.absorption("tzm_Latn", into="fra_Latn") == pytest.approx(0.1)


def test_absorption_of_an_absent_language_is_zero_not_an_error() -> None:
    matrix = matrix_of({("kab_Latn", "kab_Latn"): 1}, known=["kab_Latn"])
    assert matrix.absorption("taq_Latn") == 0.0
    assert matrix.recall("taq_Latn") == 0.0
    assert matrix.precision("taq_Latn") == 0.0
    assert matrix.f1("taq_Latn") == 0.0


def test_empty_matrix_is_all_zero() -> None:
    matrix = ConfusionMatrix(labels=(), counts={}, system_labels=frozenset())
    assert matrix.total() == 0
    assert matrix.accuracy() == 0.0
    assert matrix.macro_f1() == 0.0


def test_confusions_are_heaviest_first_and_exclude_the_diagonal() -> None:
    matrix = matrix_of(
        {
            ("shi_Latn", "shi_Latn"): 5,
            ("shi_Latn", "kab_Latn"): 9,
            ("shi_Latn", "fra_Latn"): 2,
        },
        known=["kab_Latn", "shi_Latn", "fra_Latn"],
    )
    assert matrix.confusions("shi_Latn") == [("kab_Latn", 9), ("fra_Latn", 2)]


def test_predicted_labels_keep_off_set_answers_visible() -> None:
    matrix = matrix_of({("kab_Latn", "zxx_Latn"): 3}, known=["kab_Latn"])
    assert "zxx_Latn" in matrix.predicted_labels


def test_score_tabulates_true_against_predicted() -> None:
    examples = [
        Example(text="a", language="kab_Latn", source="s"),
        Example(text="b", language="shi_Latn", source="s"),
    ]
    system = StubIdentifier(["kab_Latn", KABYLE], known=["kab_Latn"])
    matrix = score(system, examples)
    assert matrix.counts[("shi_Latn", "kab_Latn")] == 1
    assert matrix.absorption("shi_Latn") == 1.0


def test_score_rejects_a_length_mismatch() -> None:
    examples = [Example(text="a", language="kab_Latn", source="s")] * 3
    with pytest.raises(LidError, match="returned 1 labels"):
        score(StubIdentifier(["kab_Latn"], known=["kab_Latn"]), examples)


def test_score_over_no_examples_is_empty_not_an_error() -> None:
    matrix = score(StubIdentifier([], known=["kab_Latn"]), [])
    assert matrix.total() == 0


def test_read_flores_reports_a_missing_split(tmp_path: Path) -> None:
    with pytest.raises(LidError, match="FLORES\\+ split not found"):
        list(read_flores(tmp_path / "absent.jsonl"))


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("kab-eng", "KAB"),
        ("eng-kab", "KAB"),
        ("kab-fra", "KAB"),
        ("fra-kab", "KAB"),
    ],
)
def test_kabyle_side_reads_the_direction(direction: str, expected: str) -> None:
    record = {"source": "KAB" if direction.startswith("kab") else "OTHER", "direction": direction}
    record["target"] = "OTHER" if direction.startswith("kab") else "KAB"
    assert kabyle_side(record) == expected


@pytest.mark.parametrize(
    "record",
    [
        {"source": "a", "target": "b", "direction": "eng-fra"},
        {"source": "a", "target": "b", "direction": "malformed"},
        {"source": "a", "target": "b"},
        {"source": "a", "target": "b", "direction": 7},
        {"direction": "kab-eng"},
        {"direction": "kab-eng", "source": 42},
    ],
)
def test_kabyle_side_returns_none_when_it_cannot_tell(record: dict[str, object]) -> None:
    assert kabyle_side(record) is None


def write_mt(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_audit_sources_groups_by_source_and_deduplicates(tmp_path: Path) -> None:
    line = "aql-iyi di temdint n Bgayet"
    corpus = write_mt(
        tmp_path / "mt.jsonl",
        [
            {"source": line, "target": "x", "direction": "kab-eng", "source_id": "a"},
            # The same pair in the other direction: one sentence, not two.
            {"source": "x", "target": line, "direction": "eng-kab", "source_id": "a"},
        ],
    )
    system = StubIdentifier(["kab_Latn"], known=["kab_Latn"])
    audits = audit_sources(corpus, identifier=system, per_source=10, seed=1)
    assert len(audits) == 1
    assert audits[0].distinct == 1
    assert audits[0].judged == 1
    assert audits[0].kabyle_share == 1.0


def test_audit_sources_counts_short_lines_without_judging_them(tmp_path: Path) -> None:
    corpus = write_mt(
        tmp_path / "mt.jsonl",
        [
            {"source": "too short", "target": "x", "direction": "kab-eng", "source_id": "a"},
            {
                "source": "aql-iyi di temdint n Bgayet",
                "target": "x",
                "direction": "kab-eng",
                "source_id": "a",
            },
        ],
    )
    audits = audit_sources(
        corpus, identifier=StubIdentifier(["kab_Latn"], known=["kab_Latn"]), per_source=10, seed=1
    )
    assert audits[0].distinct == 2
    assert audits[0].too_short == 1
    assert audits[0].judged == 1


def test_audit_sources_orders_by_kabyle_share_worst_first(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "source": f"{source_id} line number {i} here",
            "target": "x",
            "direction": "kab-eng",
            "source_id": source_id,
        }
        for source_id in ("clean", "dirty")
        for i in range(2)
    ]
    corpus = write_mt(tmp_path / "mt.jsonl", rows)
    audits = audit_sources(
        corpus, identifier=ByPrefixIdentifier(known=["kab_Latn"]), per_source=10, seed=1
    )
    assert [a.source_id for a in audits] == ["dirty", "clean"]
    assert audits[0].kabyle_share == 0.0
    assert audits[1].kabyle_share == 1.0


def test_audit_sources_reports_a_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(LidError, match="MT corpus not found"):
        audit_sources(
            tmp_path / "absent.jsonl",
            identifier=StubIdentifier([], known=[]),
            per_source=1,
            seed=1,
        )


def test_audit_sources_over_an_empty_corpus_is_empty(tmp_path: Path) -> None:
    corpus = write_mt(tmp_path / "mt.jsonl", [])
    assert (
        audit_sources(corpus, identifier=StubIdentifier([], known=[]), per_source=1, seed=1) == []
    )


def test_source_audit_share_of_nothing_is_zero_not_an_error() -> None:
    audit = SourceAudit(source_id="a", distinct=3, judged=0, too_short=3, labels={})
    assert audit.kabyle_share == 0.0
    assert audit.to_dict()["kabyle_share"] == 0.0
