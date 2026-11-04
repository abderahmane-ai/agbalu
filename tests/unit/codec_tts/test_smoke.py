"""Reject invalid smoke inputs without touching compute targets."""

import pytest

from agbalu.codec_tts.cli import build_parser, run_directory
from agbalu.codec_tts.smoke import PARTS


@pytest.mark.parametrize("part", PARTS)
def test_every_smoke_part_has_cli_support(part: str) -> None:
    assert build_parser().parse_args(["smoke", "--part", part, "--run", "r"]).part == part


@pytest.mark.parametrize("run", [None, "../outside", "/outside", ".", ".."])
def test_invalid_run_paths_fail(run: str | None) -> None:
    with pytest.raises(ValueError, match="--run"):
        run_directory(run)
