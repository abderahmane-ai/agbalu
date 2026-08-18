"""The KenLM build arguments, and the CTC decoder they are fused into."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from agbalu.speech import cli
from agbalu.speech.lm import (
    LM_ORDER,
    LM_PRUNE,
    arpa_command,
    binary_command,
    build_ctc_decoder,
    decoder_labels,
    extract_unigrams,
)


def test_unigrams_are_the_distinct_words_of_the_targets_in_a_fixed_order() -> None:
    """Beam search constrains itself to these, so an unordered set would make two runs of
    one decoder produce different beams."""
    unigrams = extract_unigrams(["azul amek i tettiliḍ", "ur d-yettaɣ ara aɣṛum"])
    assert unigrams == sorted(unigrams)
    assert len(unigrams) == 8
    assert "aɣṛum" in unigrams


def test_a_word_repeated_across_sentences_appears_once() -> None:
    assert extract_unigrams(["azul azul", "azul"]) == ["azul"]


def test_an_empty_corpus_yields_no_unigrams_rather_than_raising() -> None:
    assert extract_unigrams([]) == []
    assert extract_unigrams(["", "   "]) == []


def test_the_build_command_is_the_one_that_produced_the_published_binary() -> None:
    """`5gram.klm` was built with `--prune 0 1 2 3`. A default that disagrees with the
    artifact means one of the two is undocumented."""
    command = arpa_command(Path("text.txt"), Path("out.arpa"))
    assert command[0] == "lmplz"
    assert command[command.index("--order") + 1] == str(LM_ORDER)
    start = command.index("--prune") + 1
    assert command[start : start + 4] == LM_PRUNE.split()


def test_the_corpus_carries_sentence_boundary_tokens_so_they_must_be_skipped() -> None:
    """Without `--skip_symbols`, `lmplz` aborts on a corpus containing `<s>` or `</s>`."""
    assert "--skip_symbols" in arpa_command(Path("a"), Path("b"))


def test_the_binary_is_a_trie_rather_than_a_probing_hash() -> None:
    """186 MB against several hundred. The model is loaded once per container, and the
    array-compressed trie is what makes that affordable beside the acoustic model."""
    assert binary_command(Path("a.arpa"), Path("b.klm")) == [
        "build_binary",
        "trie",
        "a.arpa",
        "b.klm",
    ]


def test_the_blank_is_the_pad_class_and_the_delimiter_is_a_space() -> None:
    """Passed through verbatim, `[PAD]` and `|` appear in every hypothesis and every
    error rate computed from one is wrong."""
    assert decoder_labels({"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}) == ["", "[UNK]", " ", "a"]


def test_the_labels_are_ordered_by_id_not_by_insertion() -> None:
    assert decoder_labels({"a": 3, "[PAD]": 0, "|": 2, "[UNK]": 1}) == ["", "[UNK]", " ", "a"]


def test_a_decoder_builds_without_a_language_model() -> None:
    pytest.importorskip("pyctcdecode")
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4, "c": 5}
    assert build_ctc_decoder(vocabulary) is not None


def test_a_missing_language_model_decodes_without_one_rather_than_raising() -> None:
    """The binary lives on a volume the container may not have."""
    pytest.importorskip("pyctcdecode")
    assert (
        build_ctc_decoder({"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3}, kenlm_path=Path("/no"))
        is not None
    )


def test_the_plain_corpus_reports_lines_not_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`lmplz` reads a sentence per line, and 204,185 records of AƔBALU-Text v1 carry an
    internal newline — so counting records understates what the model is built from by
    6.7%, and the card carried that number as the corpus size."""
    corpus = tmp_path / "text.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"text": text}, ensure_ascii=False) + "\n"
            for text in ("azul", "aman\nd tudert", "  ", "tamurt\nn\nleqbayel")
        ),
        encoding="utf-8",
    )
    plain = tmp_path / "plain.txt"
    # The toolchain is not the subject and is absent from the gate's machine.
    monkeypatch.setattr(cli, "build_arpa", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "compile_binary", lambda *_args, **_kwargs: None)

    assert (
        cli.command_lm(
            argparse.Namespace(
                text=corpus,
                plain=plain,
                arpa=tmp_path / "x.arpa",
                binary=tmp_path / "x.klm",
                order=LM_ORDER,
                keep_arpa=True,
            )
        )
        == 0
    )
    assert plain.read_text(encoding="utf-8").splitlines() == [
        "azul",
        "aman",
        "d tudert",
        "tamurt",
        "n",
        "leqbayel",
    ]
    assert "6 lines from 3 records" in capsys.readouterr().out
