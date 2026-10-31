"""Verify explicit encoder CLI overrides and run identities."""

import pytest

from agbalu.encoder.cli import configuration, parser


def test_overrides_reach_the_typed_recipe() -> None:
    recipe, _ = configuration(parser().parse_args(["train", "--tokens", "400000", "--lr", "0.001"]))
    assert recipe.total_tokens == 400000
    assert recipe.peak_lr == 0.001


def test_paths_cannot_escape_the_run_directory() -> None:
    with pytest.raises(ValueError, match="invalid run identifier"):
        configuration(parser().parse_args(["train", "--run", "../elsewhere"]))
