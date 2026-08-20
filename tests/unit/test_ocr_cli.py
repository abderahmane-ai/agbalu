"""The two GPU-free command-line surfaces, held to what the Makefile promises of them.

A subcommand the Makefile documents and the parser does not have is a target that cannot
run, so the two lists are checked against each other. A missing corpus, image or directory
comes back as an exit code with a message rather than a traceback, because these are the
commands an operator runs by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.ocr import cli as ocr_cli
from agbalu.standardise import cli as standardise_cli

OCR_COMMANDS = ("generate", "evaluate", "infer", "transcribe-book")
STANDARDISE_COMMANDS = ("standardise", "evaluate")


@pytest.mark.parametrize("command", OCR_COMMANDS)
def test_every_ocr_subcommand_the_makefile_documents_is_reachable(command: str) -> None:
    assert command in ocr_cli.HANDLERS


@pytest.mark.parametrize("command", STANDARDISE_COMMANDS)
def test_every_standardise_subcommand_the_makefile_documents_is_reachable(command: str) -> None:
    assert command in standardise_cli.HANDLERS


def test_the_ocr_parser_rejects_a_subcommand_that_does_not_exist() -> None:
    with pytest.raises(SystemExit):
        ocr_cli.build_parser().parse_args(["train"])


def test_a_missing_corpus_is_an_exit_code_and_not_a_traceback(tmp_path: Path) -> None:
    assert ocr_cli.main(["evaluate", "--input", str(tmp_path / "absent.jsonl")]) == 1


def test_a_missing_image_is_an_exit_code_and_not_a_traceback(tmp_path: Path) -> None:
    assert ocr_cli.main(["infer", "--image", str(tmp_path / "absent.png")]) == 1


def test_a_missing_book_directory_is_an_exit_code_and_not_a_traceback(tmp_path: Path) -> None:
    assert ocr_cli.main(["transcribe-book", "--book-dir", str(tmp_path / "absent")]) == 1


def test_the_renderer_refuses_a_corpus_that_is_not_there(tmp_path: Path) -> None:
    """Answering with a handful of sentences repeated to length renders a plausible dataset
    out of nothing."""
    with pytest.raises(ocr_cli.CorpusError, match="not found"):
        ocr_cli._read_corpus_lines(tmp_path / "absent.jsonl")


def test_the_evaluation_draw_starts_past_what_the_release_read() -> None:
    """The training loader took its sentences from the head of a source-ordered file, so a
    held-out draw has to begin after them or it is scoring the training set."""
    assert ocr_cli.TRAINED_PREFIX >= 80_000


def test_the_dual_script_evaluation_reads_the_test_split_not_the_train_split() -> None:
    assert ocr_cli.HELDOUT_TIFINAGH.name == "test.parquet"
