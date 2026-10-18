"""SFT rows from conversations and replay: parsing, the replay budget, every row accounted."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.llm import sft_build, sft_rows
from agbalu.llm.chat import Turn
from agbalu.llm.sft_build import Conversation
from tests.unit.test_llm_chat import MiniTemplate

EOS = 7


class WordTokenizer:
    """One token per word, ids from a growing table, `EOS` reserved."""

    eos = EOS

    def __init__(self) -> None:
        self.table: dict[str, int] = {}

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [[self.table.setdefault(w, 100 + len(self.table)) for w in t.split()] for t in texts]


def conversation(index: int, split: str = "train", *, think: bool = False) -> Conversation:
    answer = Turn("assistant", f"tiririt {index} " * 4, reasoning="axemmem" if think else None)
    return Conversation(f"c{index}", split, (Turn("user", f"asteqsi {index}"), answer))


def replay_file(path: Path, documents: int, words: int) -> tuple[Path, int]:
    lines = [
        json.dumps({"text": " ".join(f"w{d}x{i}" for i in range(words))}) for d in range(documents)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, documents * words


def sources(tmp_path: Path) -> dict[str, tuple[Path, int]]:
    return {
        kind: replay_file(tmp_path / f"{kind}.jsonl", 400, 3) for kind in sft_build.REPLAY_SHARES
    }


def outputs(tmp_path: Path) -> sft_build.Outputs:
    return sft_build.Outputs(tmp_path / "rows.npz", tmp_path / "validation.npz")


class TestReading:
    def test_a_conversation_file_round_trips(self, tmp_path: Path) -> None:
        row = {
            "id": "c1",
            "split": "train",
            "turns": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning": "r"},
            ],
        }
        path = tmp_path / "sft.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        (found,) = sft_build.read_conversations(path)
        assert found.turns[1] == Turn("assistant", "a", "r")

    @pytest.mark.parametrize(
        ("row", "match"),
        [
            ({"id": 1, "split": "train", "turns": []}, "not usable"),
            ({"id": "a", "split": "test", "turns": []}, "not usable"),
            ({"id": "a", "split": "train", "turns": {}}, "not a list"),
            ({"id": "a", "split": "train", "turns": [{"role": "tool", "content": "x"}]}, "role"),
            (
                {
                    "id": "a",
                    "split": "train",
                    "turns": [{"role": "user", "content": "x", "reasoning": 3}],
                },
                "reasoning",
            ),
        ],
    )
    def test_malformed_rows_are_named(
        self, tmp_path: Path, row: dict[str, object], match: str
    ) -> None:
        path = tmp_path / "sft.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(sft_build.BuildError, match=match):
            sft_build.read_conversations(path)

    def test_a_repeated_id_is_refused(self, tmp_path: Path) -> None:
        row = json.dumps({"id": "a", "split": "train", "turns": []})
        path = tmp_path / "sft.jsonl"
        path.write_text(f"{row}\n{row}\n", encoding="utf-8")
        with pytest.raises(sft_build.BuildError, match="repeats"):
            sft_build.read_conversations(path)


class TestBudget:
    def test_replay_is_a_tenth_of_the_supervised_tokens(self) -> None:
        assert sft_build.replay_budget(900) == 100

    def test_no_conversation_tokens_is_refused(self) -> None:
        with pytest.raises(sft_build.BuildError):
            sft_build.replay_budget(0)

    def test_a_draw_spans_the_whole_file_not_a_prefix(self, tmp_path: Path) -> None:
        path, _ = replay_file(tmp_path / "r.jsonl", 1000, 1)
        kept = list(sft_build.bernoulli_lines(path, 0.05, seed=1))
        indices = sorted(int(text.split("x")[0][1:]) for text in kept)
        assert indices[-1] > 900
        assert 20 < len(indices) < 90


class TestBuild:
    def test_rows_hold_every_kept_turn_and_the_replay_share(self, tmp_path: Path) -> None:
        found = [conversation(i) for i in range(30)] + [conversation(99, "validation")]
        found.append(conversation(50, think=True))
        stats = sft_build.build(
            found, MiniTemplate(), WordTokenizer(), sources(tmp_path), outputs(tmp_path)
        )
        assert stats["conversations"] == {"train": 31, "validation": 1}
        assert stats["rejected"] == {}
        share = stats["replay_share"]
        assert isinstance(share, float)
        assert abs(share - 0.10) <= sft_build.SHARE_TOLERANCE
        rows = sft_rows.Rows.read(tmp_path / "rows.npz")
        kinds = set(rows.kinds.tolist())
        assert kinds == {sft_rows.NO_THINK, sft_rows.THINK, sft_rows.REPLAY}

    def test_an_unrenderable_conversation_is_rejected_and_counted(self, tmp_path: Path) -> None:
        broken = Conversation("bad", "train", (Turn("user", "q"),))
        found = [conversation(i) for i in range(30)] + [conversation(99, "validation"), broken]
        stats = sft_build.build(
            found, MiniTemplate(), WordTokenizer(), sources(tmp_path), outputs(tmp_path)
        )
        assert stats["rejected"] == {"unrenderable": 1}

    def test_without_validation_turns_nothing_is_written(self, tmp_path: Path) -> None:
        found = [conversation(i) for i in range(5)]
        with pytest.raises(sft_build.BuildError, match="no turn"):
            sft_build.build(
                found, MiniTemplate(), WordTokenizer(), sources(tmp_path), outputs(tmp_path)
            )
        assert not (tmp_path / "rows.npz").exists()

    def test_a_replay_file_too_small_for_its_budget_is_refused(self, tmp_path: Path) -> None:
        small = sources(tmp_path)
        small["aligned"] = replay_file(tmp_path / "tiny.jsonl", 1, 1)
        found = [conversation(i) for i in range(30)] + [conversation(99, "validation")]
        with pytest.raises(sft_build.BuildError, match="replay aligned: 1 tokens in all of"):
            sft_build.build(found, MiniTemplate(), WordTokenizer(), small, outputs(tmp_path))

    def test_long_documents_are_cut_to_the_budget_exactly(self, tmp_path: Path) -> None:
        path, _ = replay_file(tmp_path / "long.jsonl", 50, 40)
        drawn = sft_build.draw_documents(path, 1.0, seed=3, target=55, tokenizer=WordTokenizer())
        assert drawn.tokens == 55
        assert [len(ids) for ids in drawn.documents] == [40, 15]

    def test_a_short_draw_is_redrawn_until_the_budget_is_met(self, tmp_path: Path) -> None:
        path, _ = replay_file(tmp_path / "r.jsonl", 200, 2)
        replay = dict.fromkeys(sft_build.REPLAY_SHARES, (path, 10**9))
        tallies = sft_build.Tallies()
        drawn = sft_build.replay_segments(replay, 100, WordTokenizer(), tallies)
        assert all(
            tallies.replay_tokens[kind] >= round(100 * share)
            for kind, share in sft_build.REPLAY_SHARES.items()
        )
        assert drawn
