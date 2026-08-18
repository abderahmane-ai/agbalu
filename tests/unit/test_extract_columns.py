from __future__ import annotations

from collections.abc import Mapping

from agbalu.extract.columns import choose_field, explicit_kab_field, sample

KAB = "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a."
KAB2 = "Ɣef waya i d-nusa ɣer da, imi nezmer ad nemmeslay akken ilaq."
ENG = "I am at home and I will not go to the city today at all."
ENG2 = "This is the reason we came here, because we are able to speak."


def rows(**columns: list[str]) -> list[Mapping[str, str]]:
    length = len(next(iter(columns.values())))
    return [{k: v[i] for k, v in columns.items()} for i in range(length)]


def test_an_explicit_name_contradicted_by_its_content_loses() -> None:
    """A field name is a claim, not evidence.

    Trusting the name unconditionally is what admitted 758,579 Tifinagh lines via
    `kab_tfng` and the count field `lang_distribution.kab`.
    """
    records = rows(kab=[ENG, ENG2], other=[KAB, KAB2])
    assert choose_field(records)[0] == "other"


def test_explicit_name_matches_nested_leaf() -> None:
    assert explicit_kab_field(["translation.kab", "translation.en"]) == "translation.kab"


def test_unnamed_columns_are_chosen_by_content() -> None:
    records = rows(col0=[KAB, KAB2], col1=[ENG, ENG2])
    name, score = choose_field(records)
    assert name == "col0"
    assert score > 0.18


def test_all_foreign_columns_are_rejected() -> None:
    records = rows(a=[ENG, ENG2], b=["Bonjour tout le monde ici", "Je suis alle en ville"])
    name, _ = choose_field(records)
    assert name is None


def test_metadata_columns_are_never_chosen() -> None:
    records = rows(
        glot_lang=["kab", "kab"], berber_status=["HIGH_CONF", "HIGH_CONF"], t=[KAB, KAB2]
    )
    name, _ = choose_field(records)
    assert name == "t"


def test_numeric_columns_are_skipped() -> None:
    records = rows(n=["1", "2"], t=[KAB, KAB2])
    assert choose_field(records)[0] == "t"


def test_empty_input_returns_none() -> None:
    assert choose_field([]) == (None, 0.0)


def test_ragged_records_do_not_crash() -> None:
    records: list[Mapping[str, str]] = [{"a": KAB}, {"b": ENG}, {"a": KAB2, "b": ENG2}]
    name, _ = choose_field(records)
    assert name == "a"


def test_sample_stops_at_the_limit() -> None:
    def endless() -> Mapping[str, str]:
        return {"t": KAB}

    assert len(sample((endless() for _ in range(10_000)), limit=25)) == 25


def test_sample_of_short_input_returns_everything() -> None:
    assert len(sample([{"t": KAB}], limit=400)) == 1


def test_tifinagh_field_name_is_not_treated_as_kabyle_latin() -> None:
    tifinagh = ["ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ", "ⴰⵔ ⵜⵉⵎⵍⵉⵍⵉⵜ"]
    records = rows(kab_tfng=tifinagh, kab_latn=[KAB, KAB2])
    assert choose_field(records)[0] == "kab_latn"


def test_a_tifinagh_only_source_yields_no_field() -> None:
    records = rows(kab_tfng=["ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ", "ⴰⵔ ⵜⵉⵎⵍⵉⵍⵉⵜ"])
    assert choose_field(records)[0] is None


def test_explicit_name_on_numeric_metadata_is_rejected() -> None:
    # `lang_distribution.kab` holds a count, not a sentence.
    records = rows(**{"lang_distribution.kab": ["2", "3"], "raw_text": [KAB, KAB2]})
    assert choose_field(records)[0] == "raw_text"


def test_explicit_name_wins_when_its_content_confirms_it() -> None:
    records = rows(kab=[KAB, KAB2], other=[ENG, ENG2])
    name, score = choose_field(records)
    assert name == "kab"
    assert score >= 0.18
