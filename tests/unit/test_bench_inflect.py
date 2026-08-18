"""The inflection floor, and the split it is read against."""

from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.bench.inflect import (
    CONFIGS,
    Entry,
    InflectionError,
    copy_floor,
    read_split,
    score_forms,
)


def entries() -> list[Entry]:
    return [
        Entry(lemma="ɛedder", feats="Person=1|Number=Plur", form="nɛedder"),
        Entry(lemma="ɛedder", feats="Mood=Imp", form="ɛedder"),
        Entry(lemma="ddu", feats="Person=1|Number=Sing", form="ddiɣ"),
    ]


def test_the_floor_is_not_zero_and_that_is_why_it_is_measured() -> None:
    """The imperative singular is the citation form, so a copy is right in one cell of
    almost every paradigm. Reading a headline exact match against zero would flatter it."""
    floor = copy_floor(entries())
    assert floor.exact == 1
    assert floor.exact_match == pytest.approx(1 / 3)


def test_the_floor_charges_the_edits_a_copy_did_not_make() -> None:
    """`ɛedder`→`nɛedder` is one insertion, `ɛedder`→`ɛedder` none, `ddu`→`ddiɣ` two."""
    floor = copy_floor(entries())
    assert floor.errors == 1 + 0 + 2
    assert floor.characters == len("nɛedder") + len("ɛedder") + len("ddiɣ")


def test_a_perfect_system_scores_one() -> None:
    rows = entries()
    assert score_forms(rows, [row.form for row in rows]).exact_match == 1.0


def test_an_unknown_config_names_the_ones_that_exist() -> None:
    with pytest.raises(InflectionError, match="expected one of"):
        read_split("test", config="conjugation")


def test_a_missing_split_says_how_to_build_it(tmp_path: Path) -> None:
    for config in CONFIGS:
        with pytest.raises(InflectionError, match="build_kabinflect"):
            read_split("test", config=config, directory=tmp_path)
