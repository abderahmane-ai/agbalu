"""Replay: budgets from shares, seeded shard order, and a sample that accounts for every row."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.llm import replay
from agbalu.llm.decontaminate import Decontaminator

EVAL = "the committee met on tuesday to decide the budget for next year in the capital"


class WordCounter:
    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


class LocalFetcher:
    """`ShardFetcher`'s surface over files already on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.revision = "abc123"

    def files(self, prefix: str, suffix: str) -> list[str]:
        names = (p.name for p in self.root.iterdir())
        return sorted(n for n in names if n.startswith(prefix) and n.endswith(suffix))

    def fetch(self, name: str) -> Path:
        return self.root / name


def shards(root: Path, count: int, rows: int) -> None:
    for shard in range(count):
        lines = [json.dumps({"text": f"word {shard} {i} " * 5}) for i in range(rows)]
        lines.append(json.dumps({"text": f"So {EVAL}."}))
        lines.append(json.dumps({"text": "  "}))
        (root / f"{shard}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


SOURCE = replay.ReplaySource("code", "x/y", "", "jsonl", "text", "odc-by", 1.0, 1.0)


class TestBudgets:
    def test_each_source_gets_its_share(self) -> None:
        got = replay.budgets(1000)
        assert got == {s.name: int(1000 * s.share) for s in replay.REPLAY_SOURCES}

    def test_shares_that_do_not_sum_to_one_are_refused(self) -> None:
        with pytest.raises(replay.ReplayError, match="sum to"):
            replay.budgets(10, [SOURCE, SOURCE])

    def test_a_zero_budget_is_refused(self) -> None:
        with pytest.raises(replay.ReplayError):
            replay.budgets(0)


class TestShardOrder:
    def test_is_seeded_and_independent_of_listing_order(self) -> None:
        names = [f"{i}.parquet" for i in range(20)]
        assert replay.shard_order(names, 3) == replay.shard_order(list(reversed(names)), 3)
        assert replay.shard_order(names, 3) != replay.shard_order(names, 4)


class TestSample:
    def test_stops_once_the_budget_is_met_and_accounts_for_every_row(self, tmp_path: Path) -> None:
        shards(tmp_path, 3, 20)
        sink = io.StringIO()
        tally = replay.sample(
            SOURCE, 50, LocalFetcher(tmp_path), Decontaminator.build([EVAL]), WordCounter(), sink
        )
        assert tally.tokens >= 50
        assert tally.read == tally.kept + tally.skipped + sum(tally.rejected.values())
        assert tally.revision == "abc123"
        assert len(sink.getvalue().splitlines()) == tally.kept

    def test_contaminated_and_empty_rows_are_rejected(self, tmp_path: Path) -> None:
        shards(tmp_path, 1, 2)
        tally = replay.sample(
            SOURCE,
            30,
            LocalFetcher(tmp_path),
            Decontaminator.build([EVAL]),
            WordCounter(),
            io.StringIO(),
        )
        assert (tally.kept, tally.tokens) == (2, 30)
        assert tally.rejected == {"contaminated": 1, "empty": 1}

    def test_a_budget_the_shards_cannot_fill_is_refused(self, tmp_path: Path) -> None:
        shards(tmp_path, 1, 2)
        with pytest.raises(replay.ReplayError, match="after every shard"):
            replay.sample(
                SOURCE,
                10_000,
                LocalFetcher(tmp_path),
                Decontaminator.build([EVAL]),
                WordCounter(),
                io.StringIO(),
            )
