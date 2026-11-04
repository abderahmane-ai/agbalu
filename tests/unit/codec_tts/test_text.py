"""Preserve meaningful Kabyle characters through script indexing."""

import unicodedata

from agbalu.codec_tts.waveforms import text_key


def test_equivalent_unicode_and_spacing() -> None:
    text = "ţ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ɛ ɣ"
    assert text_key(unicodedata.normalize("NFD", text)) == text
    assert text_key("\tţ\u00a0 ɣ \n") == "ţ ɣ"


def test_invisible_marks_and_homoglyphs_are_not_silently_deleted() -> None:
    text = "\ufeffɛ ε\u200b\u200f"
    assert text_key(text) == text
    assert text_key("[whispering] Azul!") == "[whispering] Azul!"
