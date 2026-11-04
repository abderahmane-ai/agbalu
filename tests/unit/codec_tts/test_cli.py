"""Validate preparation command parsing."""

import pytest

from agbalu.codec_tts.cli import build_parser


@pytest.mark.parametrize("action", ["inventory", "prepare", "partition"])
def test_phase_command(action: str) -> None:
    assert build_parser().parse_args([action]).action == action


def test_unknown_action_fails() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["oops"])
