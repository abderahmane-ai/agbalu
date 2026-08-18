"""The base is one definition, and every consumer reads it (task 11.3).

The corpus is counted in one tokenizer and scored on another model's held-out sets. If those
two names drift apart the token budget stops describing the run it sizes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.llm.bases import BASE, TEXT_TOWER_PARAMETERS
from agbalu.llm.cli import BASE_MODEL

CONSUMERS = (Path("modal_app/llm.py"), Path("src/agbalu/llm/cli.py"))


class TestBase:
    def test_is_the_model_the_corpus_is_counted_in(self) -> None:
        assert BASE_MODEL == BASE

    def test_names_an_owner_and_a_repo(self) -> None:
        owner, _, name = BASE.partition("/")
        assert owner
        assert name
        assert BASE.strip() == BASE

    def test_is_a_qwen3_5_checkpoint(self) -> None:
        assert BASE.startswith("Qwen/Qwen3.5-")


class TestTextTower:
    def test_excludes_the_vision_tower_and_the_mtp_head(self) -> None:
        # 2,274,069,824 total, of which 331,416,576 vision and 60,828,160 MTP.
        assert TEXT_TOWER_PARAMETERS == 2_274_069_824 - 331_416_576 - 60_828_160

    def test_sizes_one_epoch_of_full_continued_pretraining(self) -> None:
        """`6 * N * D` over 230,407,971 tokens is 2.60 EFLOP, which is the run's cost."""
        eflop = 6 * TEXT_TOWER_PARAMETERS * 230_407_971 / 1e18
        assert eflop == pytest.approx(2.60, abs=0.01)


@pytest.mark.parametrize("path", CONSUMERS)
class TestConsumersImportIt:
    def test_imports_from_the_shared_module(self, path: Path) -> None:
        assert "from agbalu.llm.bases import " in path.read_text(encoding="utf-8")

    def test_does_not_redefine_the_name(self, path: Path) -> None:
        assert "BASE: Final =" not in path.read_text(encoding="utf-8")
