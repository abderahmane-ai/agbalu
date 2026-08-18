from __future__ import annotations

import pytest

from agbalu.extract.detect import (
    column_score,
    function_rate,
    kabyle_score,
    specific_rate,
)

KABYLE = [
    "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a.",
    "Tameṭṭut-nni tenna-yas belli acu i yellan d ayen ilaqen.",
    "Ɣef waya i d-nusa ɣer da, imi nezmer ad nemmeslay.",
]
ENGLISH = [
    "I am at home and I will not go to the city today.",
    "That woman told him what was needed was the right thing.",
    "This is why we came here, because we can talk.",
]
FRENCH = [
    "Je suis a la maison et je n irai pas en ville aujourd hui.",
    "Cette femme lui a dit ce qu il fallait faire.",
]


def test_kabyle_scores_above_english_and_french() -> None:
    assert column_score(KABYLE) > column_score(ENGLISH)
    assert column_score(KABYLE) > column_score(FRENCH)


def test_score_is_bounded() -> None:
    for text in [*KABYLE, *ENGLISH, "", "   ", "123", "!!!", "ⵜⴰⵎⴰⵣⵉⵖⵜ"]:
        assert 0.0 <= kabyle_score(text) <= 1.0


@pytest.mark.parametrize("text", ["", "   ", "12345", "...", "\n\t"])
def test_empty_and_non_letter_input_scores_zero(text: str) -> None:
    assert kabyle_score(text) == 0.0
    assert specific_rate(text) == 0.0
    assert function_rate(text) == 0.0


def test_non_latin_script_scores_zero() -> None:
    assert kabyle_score("ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵇⴱⴰⵢⵍⵉⵜ") == 0.0
    assert kabyle_score("هذه جملة عربية كاملة") == 0.0
    assert kabyle_score("это предложение на русском") == 0.0


def test_specific_letters_drive_the_letter_signal() -> None:
    assert specific_rate("aɣ") > specific_rate("ag")
    assert specific_rate("abcdef") == 0.0


def test_homoglyph_text_still_scores_as_kabyle() -> None:
    # Greek epsilon for ɛ is 3.2% of the corpus; detection must survive it because
    # extraction runs before normalisation on some paths.
    corrupted = "Aql-i deg wexxam, ur ttruγuɣ ara ɣer temdint ass-a."
    assert kabyle_score(corrupted) > 0.0


def test_column_score_ignores_blank_values() -> None:
    assert column_score(["", "   ", *KABYLE]) == pytest.approx(column_score(KABYLE))


def test_column_score_of_empty_column_is_zero() -> None:
    assert column_score([]) == 0.0
    assert column_score(["", "  "]) == 0.0
