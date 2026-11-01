"""Verify source-family exclusions and prose admission rules."""

from agbalu.restoration.configuration import Settings
from agbalu.restoration.corpus import SOURCES, paragraphs, rejection, split_for


def test_mined_text_and_model_punctuated_audio_are_not_supervision() -> None:
    assert "opus.nllb-kab" not in SOURCES
    assert "taqbaylit.rradyu-tis-snat" not in SOURCES
    assert "taqbaylit.tughalin" not in SOURCES
    assert split_for("taqbaylit.bouira") == split_for("hf.imsidag.kabyle-corpus-ubouira") == 0
    assert split_for("hf.imsidag.kabyle-corpus-hca") == 1


def test_paragraphs_are_never_joined_across_removed_boundaries() -> None:
    assert paragraphs("Azul.\n\nAbrid.\n \nTaddart.") == ["Azul.", "Abrid.", "Taddart."]
    assert rejection("A | b | c | d | e | f | g | h |.", Settings()) == "markup_or_table"
