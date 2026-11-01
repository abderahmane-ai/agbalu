"""Verify constrained rendering preserves words and punctuation order."""

from agbalu.restoration.annotations import annotate
from agbalu.restoration.decoding import StructureState, render


def test_ordered_quotes_and_terminal_punctuation_render_exactly() -> None:
    result = render(("yenna", "azul"), (0, 5, 4), (1, 1), ("@", "@«", "@»"))
    assert result == "Yenna: «Azul!»"
    assert annotate(result).words == ("yenna", "azul")


def test_illegal_closer_is_counted_without_removing_primary_mark() -> None:
    state = StructureState()
    assert render(("azul",), (0, 2), (1,), ("@", "@»"), state) == "Azul."
    assert state.suppressed == 1


def test_unclosed_quotes_are_not_completed_by_guessing() -> None:
    assert render(("azul",), (0, 0), (0,), ("@«", "@")) == "«azul"
