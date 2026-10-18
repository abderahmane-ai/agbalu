"""The checkpoints Jugurtha starts from, defined once and imported by every consumer."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agbalu.llm.bases import BASE, INSTRUCT, PILOT, REVISIONS, TEXT_TOWER_PARAMETERS

CONSUMERS = (
    Path("src/agbalu/llm/cli.py"),
    Path("src/agbalu/llm/smoke.py"),
    Path("src/agbalu/llm/train.py"),
)


class TestNames:
    def test_the_base_is_the_4b_base_checkpoint(self) -> None:
        assert BASE == "Qwen/Qwen3.5-4B-Base"

    def test_the_merge_target_is_the_base_without_its_suffix(self) -> None:
        assert BASE.removesuffix("-Base") == INSTRUCT

    def test_the_pilot_is_a_base_checkpoint_of_the_same_family(self) -> None:
        assert PILOT.startswith("Qwen/Qwen3.5-")
        assert PILOT.endswith("-Base")


class TestRevisions:
    def test_both_merged_checkpoints_are_pinned_to_a_full_commit(self) -> None:
        assert set(REVISIONS) == {BASE, INSTRUCT}
        assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in REVISIONS.values())


class TestTextTower:
    def test_every_checkpoint_has_a_count(self) -> None:
        assert set(TEXT_TOWER_PARAMETERS) == {BASE, INSTRUCT, PILOT}

    def test_the_twin_shares_the_base_count_tensor_for_tensor(self) -> None:
        assert TEXT_TOWER_PARAMETERS[BASE] == TEXT_TOWER_PARAMETERS[INSTRUCT]

    def test_the_pilot_is_about_a_fifth_of_the_base(self) -> None:
        ratio = TEXT_TOWER_PARAMETERS[PILOT] / TEXT_TOWER_PARAMETERS[BASE]
        assert 0.15 < ratio < 0.2


@pytest.mark.parametrize("path", CONSUMERS)
class TestConsumersImportIt:
    def test_imports_from_the_shared_module(self, path: Path) -> None:
        assert "from agbalu.llm.bases import " in path.read_text(encoding="utf-8")

    def test_does_not_redefine_the_name(self, path: Path) -> None:
        assert "BASE: Final =" not in path.read_text(encoding="utf-8")
