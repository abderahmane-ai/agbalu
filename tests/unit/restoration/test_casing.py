"""Verify case components preserve Kabyle clitics and proper names."""

from agbalu.restoration.annotations import apply_case
from agbalu.restoration.casing import case_units, component_bytes


def test_clitic_and_name_have_separate_targets() -> None:
    units = case_units("d-Taqbaylit")
    assert [(unit.char_start, unit.char_end, unit.label) for unit in units] == [
        (0, 1, 0),
        (2, 11, 1),
    ]
    assert [unit.label for unit in case_units("Ait-Menguellet")] == [1, 1]


def test_byte_offsets_account_for_multibyte_kabyle_letters() -> None:
    word = "ɣer-Ɣer"
    units = case_units(word)
    assert [component_bytes(word, unit) for unit in units] == [(0, 4), (5, 9)]
    assert [unit.label for unit in units] == [0, 1]


def test_uncased_and_unrepresentable_components_remain_explicit() -> None:
    assert [unit.label for unit in case_units("2026-taQbaylit")] == [-100, -100]
    assert case_units("") == ()


def test_component_predictions_restore_hyphenated_names_without_spelling_edits() -> None:
    assert apply_case("d-taqbaylit", (0, 1)) == "d-Taqbaylit"
    assert apply_case("ait-menguellet", (1, 1)) == "Ait-Menguellet"
    assert apply_case("n-lezzayer", (0, 1)) == "n-Lezzayer"
    assert apply_case("s-tmaziɣt", (0, 1)) == "s-Tmaziɣt"
