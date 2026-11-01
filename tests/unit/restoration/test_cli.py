"""Verify restoration command defaults and override validation."""

import pytest

from agbalu.restoration.cli import configuration, parser


def test_train_defaults_need_no_operator_recipe_decisions() -> None:
    settings, locations = configuration(parser().parse_args(["train", "--run", "belaid"]))
    assert settings.epochs == 2
    assert settings.require_smoke
    assert str(locations.encoder) == "artifacts/runs/encoder-pretrain"


def test_path_like_run_names_are_refused() -> None:
    with pytest.raises(ValueError, match="identifier"):
        configuration(parser().parse_args(["train", "--run", "../other"]))
