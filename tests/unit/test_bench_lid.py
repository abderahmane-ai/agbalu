from __future__ import annotations

import json
from collections import Counter
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
    kabyle_side,
    read_evalset,
    score,
    write_evalset,
)
from agbalu.bench.lid_sources import (
    SourceError,
    candidates,
    is_usable,
    latin_share,
    read_flores,
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


def pool(language: str, source: str, count: int, domain: str = "web") -> list[Example]:
    return [
        Example(text=f"{source} line {index}", language=language, source=source, domain=domain)
        for index in range(count)
    ]


def test_build_evalset_is_balanced_and_deterministic() -> None:
    pools = {lang: pool(lang, f"{lang}.a", 50) for lang in ("kab_Latn", "shi_Latn")}
    first = build_evalset(pools, per_language=10, seed=7)
    assert first == build_evalset(pools, per_language=10, seed=7)
    assert len(first) == 20
    assert sum(1 for example in first if example.language == "kab_Latn") == 10


def test_build_evalset_changes_with_the_seed() -> None:
    pools = {"kab_Latn": pool("kab_Latn", "kab.a", 200)}
    assert build_evalset(pools, per_language=20, seed=1) != build_evalset(
        pools, per_language=20, seed=2
    )


def test_build_evalset_draws_evenly_across_sources_not_proportionally() -> None:
    """The whole point of the rebuild: one source of 5,000 lines and one of 40 must not
    give a 125:1 evaluation split, because that measures the big source."""
    pools = {"rif_Latn": pool("rif_Latn", "big", 5_000) + pool("rif_Latn", "small", 40)}
    drawn = build_evalset(pools, per_language=60, seed=3)
    counts = Counter(example.source for example in drawn)
    assert counts["big"] == 30
    assert counts["small"] == 30


def test_build_evalset_spends_a_small_source_then_takes_the_rest_from_the_large() -> None:
    pools = {"shy_Latn": pool("shy_Latn", "big", 100) + pool("shy_Latn", "small", 4)}
    drawn = build_evalset(pools, per_language=20, seed=3)
    counts = Counter(example.source for example in drawn)
    assert counts == {"big": 16, "small": 4}


def test_build_evalset_refuses_an_unbalanced_draw() -> None:
    pools = {"kab_Latn": pool("kab_Latn", "a", 10), "tzm_Latn": pool("tzm_Latn", "b", 3)}
    with pytest.raises(LidError, match="tzm_Latn has 3"):
        build_evalset(pools, per_language=10, seed=1)


def test_build_evalset_refuses_a_non_positive_draw() -> None:
    with pytest.raises(LidError, match="must be positive"):
        build_evalset({"kab_Latn": pool("kab_Latn", "a", 1)}, per_language=0, seed=1)


def test_evalset_round_trips_its_provenance(tmp_path: Path) -> None:
    """The published set carried the language in the `source` field, so nothing could be
    told about where a row came from. Provenance survives the write or the set is a
    list of strings again."""
    examples = build_evalset(
        {"rif_Latn": pool("rif_Latn", "tatoeba.rif", 20)}, per_language=5, seed=1
    )
    destination = tmp_path / "lid-eval.jsonl"
    write_evalset(examples, destination)
    assert read_evalset(destination) == examples
    assert {example.source for example in read_evalset(destination)} == {"tatoeba.rif"}


def test_read_evalset_refuses_a_missing_or_empty_file(tmp_path: Path) -> None:
    with pytest.raises(LidError, match="not found"):
        read_evalset(tmp_path / "absent.jsonl")
    (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
    with pytest.raises(LidError, match="carried no records"):
        read_evalset(tmp_path / "empty.jsonl")


def test_score_ignores_a_class_public_identifiers_cannot_be_asked_about() -> None:
    """`NOT_AMAZIGH` is a decision the trained model makes; GlotLID has no such label,
    so scoring it against one would report the label inventory."""
    examples = [
        Example("a", "kab_Latn", "s", "web"),
        Example("b", "NOT_AMAZIGH", "s", "negative-latin"),
    ]
    matrix = score(StubIdentifier(["kab_Latn"], EVAL_LANGUAGES), examples)
    assert matrix.total() == 1
    assert matrix.labels == ("kab_Latn",)


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
        Example(text="a", language="kab_Latn", source="s", domain="web"),
        Example(text="b", language="shi_Latn", source="s", domain="web"),
    ]
    system = StubIdentifier(["kab_Latn", KABYLE], known=["kab_Latn"])
    matrix = score(system, examples)
    assert matrix.counts[("shi_Latn", "kab_Latn")] == 1
    assert matrix.absorption("shi_Latn") == 1.0


def test_score_rejects_a_length_mismatch() -> None:
    examples = [Example(text="a", language="kab_Latn", source="s", domain="web")] * 3
    with pytest.raises(LidError, match="returned 1 labels"):
        score(StubIdentifier(["kab_Latn"], known=["kab_Latn"]), examples)


def test_score_over_no_examples_is_empty_not_an_error() -> None:
    matrix = score(StubIdentifier([], known=["kab_Latn"]), [])
    assert matrix.total() == 0


def test_read_flores_reports_a_missing_language(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="no FLORES\\+ split for taq_Latn"):
        list(read_flores(tmp_path, "taq_Latn"))


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
